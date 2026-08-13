/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/io/experimental/deletion_vectors.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/iterator>
#include <thrust/sequence.h>

#include <roaring/roaring.h>
#include <roaring/roaring64.h>

#include <utility>

namespace {

std::vector<char> write_parquet(cudf::table_view const& input_table, std::size_t rows_per_row_group)
{
  CUDF_EXPECTS(rows_per_row_group > 0, "Parquet writer must at least write one row per row group");

  auto parquet_buffer           = std::vector<char>{};
  auto const row_group_num_rows = static_cast<cudf::size_type>(rows_per_row_group);

  // Explicitly set all columns to non-nullable as input
  auto metadata = cudf::io::table_input_metadata{input_table};
  for (auto& col_meta : metadata.column_metadata) {
    col_meta.set_nullability(false);
  }

  auto out_opts =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&parquet_buffer))
      .metadata(std::move(metadata))
      .build();

  auto writer    = cudf::io::chunked_parquet_writer(out_opts);
  auto start_row = cudf::size_type{0};
  while (start_row < input_table.num_rows()) {
    auto const end_row =
      std::min<cudf::size_type>(start_row + row_group_num_rows, input_table.num_rows());
    writer.write(cudf::slice(input_table, {start_row, end_row}).front());
    start_row = end_row;
  }
  writer.close();
  return parquet_buffer;
}

/**
 * @brief Builds a cudf column from a span of host data
 *
 * @param host_data Span of host data
 * @param data_type The data type of the column
 * @param stream The stream to use for the operation
 * @param mr The memory resource to use for the operation
 *
 * @return A unique pointer to a column containing the row mask
 */
template <typename T>
auto build_column_from_host_data(cudf::host_span<T const> host_data,
                                 cudf::type_id data_type,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not host_data.empty(), "Host data vector must not be empty");

  auto const num_rows = host_data.size();
  rmm::device_buffer buffer{num_rows * sizeof(T), stream, mr};
  cudf::detail::cuda_memcpy_async<T>(
    cudf::device_span<T>{static_cast<T*>(buffer.data()), num_rows}, host_data, stream);
  return std::make_unique<cudf::column>(
    cudf::data_type{data_type}, num_rows, std::move(buffer), rmm::device_buffer{}, 0);
}

/**
 * @brief Builds a host vector of expected row indices from the specified row group offsets and
 * row counts
 *
 * @param row_group_offsets Row group offsets
 * @param row_group_num_rows Number of rows in each row group
 * @param num_rows Total number of table rows
 *
 * @return Host vector of expected row indices
 */
auto build_expected_row_indices(cudf::host_span<std::size_t const> row_group_offsets,
                                cudf::host_span<cudf::size_type const> row_group_num_rows,
                                cudf::size_type num_rows)
{
  auto const num_row_groups = static_cast<cudf::size_type>(row_group_num_rows.size());

  // Row span offsets
  auto row_group_span_offsets = std::vector<cudf::size_type>(num_row_groups + 1);
  row_group_span_offsets[0]   = 0;
  std::inclusive_scan(
    row_group_num_rows.begin(), row_group_num_rows.end(), row_group_span_offsets.begin() + 1);

  // Expected row indices data
  auto expected_row_indices = std::vector<std::size_t>(num_rows);
  // Initialize all row indices to 1 so that we can do a segmented inclusive scan
  std::fill(expected_row_indices.begin(), expected_row_indices.end(), 1);

  // Scatter row group offsets to expected row indices
  thrust::scatter(thrust::host,
                  row_group_offsets.begin(),
                  row_group_offsets.end(),
                  row_group_span_offsets.begin(),
                  expected_row_indices.begin());

  // Inclusive scan to compute the rest of the row indices
  std::for_each(cuda::counting_iterator(0), cuda::counting_iterator(num_row_groups), [&](auto i) {
    auto start_row_index = row_group_span_offsets[i];
    auto end_row_index   = row_group_span_offsets[i + 1];
    std::inclusive_scan(expected_row_indices.begin() + start_row_index,
                        expected_row_indices.begin() + end_row_index,
                        expected_row_indices.begin() + start_row_index);
  });

  return expected_row_indices;
}

/**
 * @brief Builds a roaring64 deletion vector and a (host) row mask vector based on the specified
 * probability of a row being deleted
 *
 * @param num_rows Number of rows in the table
 * @param deletion_probability The probability of a row being deleted
 * @param row_indices Host vector of row indices
 * @param are_retention_vectors Whether to add retained, rather than deleted, row indices to the
 * bitmap
 *
 * @return A pair of a roaring bitmap and a host row mask vector
 */
auto build_roaring_bitmap_and_expected_row_mask(cudf::size_type num_rows,
                                                float deletion_probability,
                                                cudf::host_span<std::size_t const> row_indices,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr,
                                                bool are_retention_vectors = false)
{
  static constexpr auto seed = 0xbaLL;
  std::mt19937 engine{seed};
  std::bernoulli_distribution dist(deletion_probability);

  CUDF_EXPECTS(std::cmp_equal(row_indices.size(), num_rows),
               "Row indices vector must have the same number of rows as the table");

  auto expected_row_mask = thrust::host_vector<bool>(num_rows);
  std::generate(expected_row_mask.begin(), expected_row_mask.end(), [&]() { return dist(engine); });

  auto deletion_vector = roaring::api::roaring64_bitmap_create();

  // Context for the roaring64 bitmap for faster (bulk) add operations
  auto roaring64_context =
    roaring::api::roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0}, .leaf = nullptr};

  std::for_each(cuda::counting_iterator<size_t>(0),
                cuda::counting_iterator<size_t>(num_rows),
                [&](auto row_idx) {
                  // For retention vectors, retain rows selected in the row mask. For deletion
                  // vectors, delete the remaining rows.
                  if (expected_row_mask[row_idx] == are_retention_vectors) {
                    roaring::api::roaring64_bitmap_add_bulk(
                      deletion_vector, &roaring64_context, row_indices[row_idx]);
                  }
                });

  auto const num_bytes = roaring::api::roaring64_bitmap_portable_size_in_bytes(deletion_vector);
  EXPECT_GT(num_bytes, 0);
  auto serialized_roaring_bitmap = std::vector<cuda::std::byte>(num_bytes);
  auto const bytes_written       = roaring::api::roaring64_bitmap_portable_serialize(
    deletion_vector, reinterpret_cast<char*>(serialized_roaring_bitmap.data()));
  EXPECT_EQ(bytes_written, num_bytes);
  auto num_buckets = uint64_t{0};
  std::memcpy(
    &num_buckets, reinterpret_cast<uint64_t*>(serialized_roaring_bitmap.data()), sizeof(uint64_t));
  EXPECT_LT(num_buckets, std::numeric_limits<cudf::size_type>::max());
  roaring::api::roaring64_bitmap_free(deletion_vector);

  return std::make_pair(
    std::move(serialized_roaring_bitmap),
    build_column_from_host_data<bool>(expected_row_mask, cudf::type_id::BOOL8, stream, mr));
}

/**
 * @brief Builds the expected cudf table by prepending the input table columns with the specified
 * row index column and applying the specified row mask column
 */
std::unique_ptr<cudf::table> build_expected_table(
  cudf::table_view const& input_table_view,
  cudf::column_view const& expected_row_index_column,
  cudf::column_view const& row_mask_column,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = input_table_view.num_rows();

  CUDF_EXPECTS(expected_row_index_column.size() == num_rows,
               "Index column and table must have the same number of rows");
  CUDF_EXPECTS(row_mask_column.size() == num_rows,
               "Row mask column and table must have the same number of rows");

  auto index_and_columns = std::vector<cudf::column_view>{};
  index_and_columns.reserve(input_table_view.num_columns() + 1);
  index_and_columns.push_back(expected_row_index_column);
  std::transform(cuda::counting_iterator(0),
                 cuda::counting_iterator(input_table_view.num_columns()),
                 std::back_inserter(index_and_columns),
                 [&](auto col_idx) { return input_table_view.column(col_idx); });
  return cudf::apply_boolean_mask(cudf::table_view{index_and_columns}, row_mask_column, stream, mr);
}

/**
 * @brief Constructs a resultant table by applying the specified deletion vector to the input table
 * and compares it to the table read via the `cudf::io::parquet::experimental::read_parquet` API as
 * well as the `cudf::io::parquet::experimental::chunked_parquet_reader`
 */
template <cudf::size_type num_concat = 1>
void test_read_parquet_and_apply_mask(
  cudf::host_span<char const> parquet_buffer,
  cudf::io::parquet::experimental::deletion_vector_info const& deletion_vector_info,
  cudf::table_view const& input_table_view,
  cudf::column_view const& expected_row_mask_column,
  cudf::column_view const& expected_row_index_column,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static_assert(std::cmp_greater_equal(num_concat, 1),
                "num_concat must be greater than or equal to 1");

  CUDF_EXPECTS(num_concat == 1 or not deletion_vector_info.serialized_roaring_bitmaps.empty(),
               "serialized_roaring_bitmaps must be non-empty if num_concat is greater than 1");

  auto const num_input_rows = input_table_view.num_rows();

  // Initialize a deletion vector information struct
  auto final_deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
    .serialized_roaring_bitmaps = deletion_vector_info.serialized_roaring_bitmaps,
    .deletion_vector_row_counts = std::vector(num_concat, input_table_view.num_rows()),
    .row_group_offsets          = {},
    .row_group_num_rows         = {},
    .are_retention_vectors      = deletion_vector_info.are_retention_vectors,
  };

  // Vector to hold the Parquet buffer spans
  auto parquet_buffers = std::vector<cudf::host_span<char const>>(num_concat, parquet_buffer);

  // Fill in row index column information
  auto& row_group_row_offsets = final_deletion_vector_info.row_group_offsets;
  auto& row_group_num_counts  = final_deletion_vector_info.row_group_num_rows;
  row_group_row_offsets.reserve(num_concat * deletion_vector_info.row_group_offsets.size());
  row_group_num_counts.reserve(num_concat * deletion_vector_info.row_group_num_rows.size());

  for (auto i = 0; i < num_concat; ++i) {
    row_group_row_offsets.insert(row_group_row_offsets.end(),
                                 deletion_vector_info.row_group_offsets.begin(),
                                 deletion_vector_info.row_group_offsets.end());
    row_group_num_counts.insert(row_group_num_counts.end(),
                                deletion_vector_info.row_group_num_rows.begin(),
                                deletion_vector_info.row_group_num_rows.end());
  }

  // Vector to store the extra serialized roaring bitmap data
  auto serialized_roaring_bitmaps_data = std::vector<std::vector<cuda::std::byte>>{};
  serialized_roaring_bitmaps_data.reserve(num_concat - 1);

  // Build the expected table
  auto expected_table = [&]() {
    auto tables      = std::vector<std::unique_ptr<cudf::table>>{};
    auto table_views = std::vector<cudf::table_view>{};
    tables.reserve(num_concat);
    table_views.reserve(num_concat);
    // Insert the first table
    tables.emplace_back(build_expected_table(
      input_table_view, expected_row_index_column, expected_row_mask_column, stream, mr));
    table_views.emplace_back(tables.back()->view());

    if constexpr (num_concat == 1) {
      return std::move(tables.back());
    } else {
      constexpr auto deletion_probability = 0.45;

      for (auto i = 0; i < num_concat - 1; ++i) {
        // Build deletion vector and the expected row mask column
        auto local_expected_row_indices =
          build_expected_row_indices(deletion_vector_info.row_group_offsets,
                                     deletion_vector_info.row_group_num_rows,
                                     num_input_rows);

        auto local_expected_row_index_column = build_column_from_host_data<std::size_t>(
          local_expected_row_indices, cudf::type_id::UINT64, stream, mr);

        auto [local_deletion_vector, local_expected_row_mask_column] =
          build_roaring_bitmap_and_expected_row_mask(
            num_input_rows,
            deletion_probability,
            local_expected_row_indices,
            stream,
            mr,
            final_deletion_vector_info.are_retention_vectors);

        // Insert the expected table, the corresponding deletion vector and its data span
        tables.emplace_back(build_expected_table(input_table_view,
                                                 local_expected_row_index_column->view(),
                                                 local_expected_row_mask_column->view(),
                                                 stream,
                                                 mr));
        table_views.emplace_back(tables.back()->view());
        serialized_roaring_bitmaps_data.emplace_back(std::move(local_deletion_vector));
        final_deletion_vector_info.serialized_roaring_bitmaps.emplace_back(
          serialized_roaring_bitmaps_data.back());
      }

      // Final expected table is the concatenation of all the expected tables
      return cudf::concatenate(table_views, stream, mr);
    }
  }();

  // Read parquet and apply deletion vector
  auto const in_opts =
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{cudf::host_span<cudf::host_span<char const>>{parquet_buffers}})
      .build();

  auto const table_with_deletion_vector =
    cudf::io::parquet::experimental::read_parquet(in_opts, final_deletion_vector_info, stream, mr)
      .tbl;

  // Check
  CUDF_TEST_EXPECT_TABLES_EQUAL(table_with_deletion_vector->view(), expected_table->view());

  // Read using the chunked reader
  auto const test_chunked_table_with_deletion_vector = [&](std::size_t chunk_read_limit,
                                                           std::size_t pass_read_limit) {
    auto const reader = std::make_unique<cudf::io::parquet::experimental::chunked_parquet_reader>(
      chunk_read_limit, pass_read_limit, in_opts, final_deletion_vector_info, stream, mr);

    auto table_chunks      = std::vector<std::unique_ptr<cudf::table>>{};
    auto table_chunk_views = std::vector<cudf::table_view>{};
    while (reader->has_next()) {
      table_chunks.emplace_back(reader->read_chunk().tbl);
      table_chunk_views.emplace_back(table_chunks.back()->view());
    }
    auto chunked_table_with_deletion_vector = cudf::concatenate(table_chunk_views, stream, mr);
    // Check
    CUDF_TEST_EXPECT_TABLES_EQUAL(chunked_table_with_deletion_vector->view(),
                                  expected_table->view());
  };

  test_chunked_table_with_deletion_vector(512, 2048);
  test_chunked_table_with_deletion_vector(1024, 10240);
  test_chunked_table_with_deletion_vector(10240, 102400);
  test_chunked_table_with_deletion_vector(102400, 0);
  test_chunked_table_with_deletion_vector(0, 0);
}

}  // namespace

// Base test fixture for API tests
struct ParquetDeletionVectorsTest : public cudf::test::BaseFixture {};

TEST_F(ParquetDeletionVectorsTest, NoRowIndexColumn)
{
  auto constexpr num_rows             = 5'000;
  auto constexpr rows_per_row_group   = 1'000;
  auto constexpr num_columns          = 8;
  auto constexpr include_validity     = false;
  auto constexpr deletion_probability = 0.5;  ///< 50% of the rows are deleted
  static_assert(num_rows % rows_per_row_group == 0,
                "num_rows must be a multiple of rows_per_row_group");

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto input_table = create_random_fixed_table<float>(num_columns, num_rows, include_validity);

  // Write table to parquet buffer
  auto parquet_buffer = write_parquet(input_table->view(), rows_per_row_group);

  // Build the expected row index column
  auto expected_row_indices = thrust::host_vector<std::size_t>(num_rows);
  std::iota(expected_row_indices.begin(), expected_row_indices.end(), std::size_t{0});
  auto expected_row_index_column = build_column_from_host_data<std::size_t>(
    expected_row_indices, cudf::type_id::UINT64, stream, mr);

  // Use num_concat = 1 here since the row index column is simply a sequence and input table
  // concatenation won't properly reset it.
  auto constexpr num_concat = 1;

  // Build deletion vector and the expected row mask column
  {
    auto [deletion_vector, expected_row_mask_column] = build_roaring_bitmap_and_expected_row_mask(
      num_rows, deletion_probability, expected_row_indices, stream, mr);
    auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
      .serialized_roaring_bitmaps = {deletion_vector},
      .deletion_vector_row_counts = {input_table->view().num_rows()}};
    test_read_parquet_and_apply_mask<num_concat>(parquet_buffer,
                                                 deletion_vector_info,
                                                 input_table->view(),
                                                 expected_row_mask_column->view(),
                                                 expected_row_index_column->view(),
                                                 stream,
                                                 mr);
  }

  // Retention vectors retain the rows present in the bitmap.
  {
    auto constexpr are_retention_vectors               = true;
    auto [retention_vector, retention_row_mask_column] = build_roaring_bitmap_and_expected_row_mask(
      num_rows, deletion_probability, expected_row_indices, stream, mr, are_retention_vectors);
    auto retention_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
      .serialized_roaring_bitmaps = {retention_vector},
      .deletion_vector_row_counts = {input_table->view().num_rows()},
      .are_retention_vectors      = are_retention_vectors};
    test_read_parquet_and_apply_mask<num_concat>(parquet_buffer,
                                                 retention_vector_info,
                                                 input_table->view(),
                                                 retention_row_mask_column->view(),
                                                 expected_row_index_column->view(),
                                                 stream,
                                                 mr);
  }

  // Test no row index column and no deletion vector
  {
    // Build expected row mask column containing all true values
    auto expected_row_mask = thrust::host_vector<bool>(num_rows, true);
    auto expected_row_mask_column =
      build_column_from_host_data<bool>(expected_row_mask, cudf::type_id::BOOL8, stream, mr);
    auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{};
    test_read_parquet_and_apply_mask<num_concat>(parquet_buffer,
                                                 deletion_vector_info,
                                                 input_table->view(),
                                                 expected_row_mask_column->view(),
                                                 expected_row_index_column->view(),
                                                 stream,
                                                 mr);
  }
}

TEST_F(ParquetDeletionVectorsTest, CustomRowIndexColumn)
{
  auto constexpr num_rows             = 5'000;
  auto constexpr num_columns          = 4;
  auto constexpr include_validity     = false;
  auto constexpr deletion_probability = 0.4;  ///< 40% of the rows are deleted
  auto constexpr rows_per_row_group   = 1'000;
  auto constexpr num_row_groups       = num_rows / rows_per_row_group;
  static_assert(num_rows % rows_per_row_group == 0,
                "num_rows must be a multiple of rows_per_row_group");

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto input_table = create_random_fixed_table<float>(num_columns, num_rows, include_validity);

  // Write table to parquet buffer
  auto parquet_buffer = write_parquet(input_table->view(), rows_per_row_group);

  // Row offsets for each row group - arbitrary, only used to build the UINT64 `index` column
  auto row_group_offsets = std::vector<size_t>(num_row_groups);
  row_group_offsets[0]   = static_cast<size_t>(std::llround(1e9));
  std::transform(
    cuda::counting_iterator(1),
    cuda::counting_iterator(num_row_groups),
    row_group_offsets.begin() + 1,
    [&](auto i) { return static_cast<size_t>(std::llround(row_group_offsets[i - 1] + 0.5e9)); });

  // Split the `num_rows` into `num_row_groups` spans
  auto row_group_splits = std::vector<cudf::size_type>(num_row_groups - 1);
  {
    std::mt19937 engine{0xf00d};
    std::uniform_int_distribution<cudf::size_type> dist{1, num_rows};
    std::generate(row_group_splits.begin(), row_group_splits.end(), [&]() { return dist(engine); });
    std::sort(row_group_splits.begin(), row_group_splits.end());
  }

  // Number of rows in each row group span
  auto row_group_num_rows = std::vector<cudf::size_type>{};
  {
    row_group_num_rows.reserve(num_row_groups);
    auto previous_split = cudf::size_type{0};
    std::transform(row_group_splits.begin(),
                   row_group_splits.end(),
                   std::back_inserter(row_group_num_rows),
                   [&](auto current_split) {
                     auto current_split_size = current_split - previous_split;
                     previous_split          = current_split;
                     return current_split_size;
                   });
    row_group_num_rows.push_back(num_rows - row_group_splits.back());
  }

  // Build the expected row index column
  auto expected_row_indices =
    build_expected_row_indices(row_group_offsets, row_group_num_rows, num_rows);
  auto expected_row_index_column = build_column_from_host_data<std::size_t>(
    expected_row_indices, cudf::type_id::UINT64, stream, mr);

  // Build deletion vector and the expected row mask column
  auto [deletion_vector, expected_row_mask_column] = build_roaring_bitmap_and_expected_row_mask(
    num_rows, deletion_probability, expected_row_indices, stream, mr);

  // Don't concatenate the input table and test with single deletion vector
  auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
    .serialized_roaring_bitmaps = {deletion_vector},
    .deletion_vector_row_counts = {input_table->view().num_rows()},
    .row_group_offsets          = row_group_offsets,
    .row_group_num_rows         = row_group_num_rows};
  test_read_parquet_and_apply_mask<1>(parquet_buffer,
                                      deletion_vector_info,
                                      input_table->view(),
                                      expected_row_mask_column->view(),
                                      expected_row_index_column->view(),
                                      stream,
                                      mr);

  // Concatenate input table and test with multiple deletion vectors
  test_read_parquet_and_apply_mask<4>(parquet_buffer,
                                      deletion_vector_info,
                                      input_table->view(),
                                      expected_row_mask_column->view(),
                                      expected_row_index_column->view(),
                                      stream,
                                      mr);

  // Concatenate input table and test with many deletion vectors (>= stream fork threshold of 8)
  test_read_parquet_and_apply_mask<8>(parquet_buffer,
                                      deletion_vector_info,
                                      input_table->view(),
                                      expected_row_mask_column->view(),
                                      expected_row_index_column->view(),
                                      stream,
                                      mr);
  test_read_parquet_and_apply_mask<16>(parquet_buffer,
                                       deletion_vector_info,
                                       input_table->view(),
                                       expected_row_mask_column->view(),
                                       expected_row_index_column->view(),
                                       stream,
                                       mr);

  // Retention vectors with custom row indices.
  {
    auto constexpr are_retention_vectors               = true;
    auto [retention_vector, retention_row_mask_column] = build_roaring_bitmap_and_expected_row_mask(
      num_rows, deletion_probability, expected_row_indices, stream, mr, are_retention_vectors);
    deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
      .serialized_roaring_bitmaps = {retention_vector},
      .deletion_vector_row_counts = {input_table->view().num_rows()},
      .row_group_offsets          = row_group_offsets,
      .row_group_num_rows         = row_group_num_rows,
      .are_retention_vectors      = are_retention_vectors};
    test_read_parquet_and_apply_mask<1>(parquet_buffer,
                                        deletion_vector_info,
                                        input_table->view(),
                                        retention_row_mask_column->view(),
                                        expected_row_index_column->view(),
                                        stream,
                                        mr);
  }

  // Test custom row index column and no deletion vector
  {
    // Build expected row mask column containing all true values
    auto expected_row_mask = thrust::host_vector<bool>(num_rows, true);
    auto expected_row_mask_column =
      build_column_from_host_data<bool>(expected_row_mask, cudf::type_id::BOOL8, stream, mr);

    auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
      .row_group_offsets = row_group_offsets, .row_group_num_rows = row_group_num_rows};
    test_read_parquet_and_apply_mask<1>(parquet_buffer,
                                        deletion_vector_info,
                                        input_table->view(),
                                        expected_row_mask_column->view(),
                                        expected_row_index_column->view(),
                                        stream,
                                        mr);
  }
}

// Test fixture for deletion vectors delete count API
struct DeletionVectorsCountTests : public ParquetDeletionVectorsTest {};

TEST_F(DeletionVectorsCountTests, EmptyDeletionVector)
{
  auto const stream = cudf::get_default_stream();

  auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{};
  auto const result         = cudf::io::parquet::experimental::compute_num_deleted_rows(
    deletion_vector_info, std::numeric_limits<cudf::size_type>::max(), stream);
  EXPECT_EQ(result, 0);
}

TEST_F(DeletionVectorsCountTests, NoRowIndex)
{
  auto constexpr num_rows             = 50'000;
  auto constexpr deletion_probability = 0.5;

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto row_indices = thrust::host_vector<size_t>(num_rows);
  std::iota(row_indices.begin(), row_indices.end(), size_t{0});

  for (auto const are_retention_vectors : {false, true}) {
    auto [deletion_vector, expected_row_mask_column] = build_roaring_bitmap_and_expected_row_mask(
      num_rows, deletion_probability, row_indices, stream, mr, are_retention_vectors);

    auto const expected_row_mask = cudf::detail::make_host_vector(
      cudf::device_span<bool const>(expected_row_mask_column->view().data<bool>(), num_rows),
      stream);
    auto const expected_deleted =
      std::count(expected_row_mask.begin(), expected_row_mask.end(), false);

    auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
      .serialized_roaring_bitmaps = {deletion_vector},
      .deletion_vector_row_counts = {num_rows},
      .row_group_offsets          = {},
      .row_group_num_rows         = {},
      .are_retention_vectors      = are_retention_vectors};

    for (auto chunk_size : {num_rows, num_rows / 2}) {
      auto const result = cudf::io::parquet::experimental::compute_num_deleted_rows(
        deletion_vector_info, chunk_size, stream);
      EXPECT_EQ(result, expected_deleted);
    }
  }
}

TEST_F(DeletionVectorsCountTests, CustomRowIndex)
{
  auto constexpr num_rows             = 25'000;
  auto constexpr num_row_groups       = 5;
  auto constexpr deletion_probability = 0.4;

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Arbitrary row group offsets
  auto row_group_offsets = std::vector<size_t>(num_row_groups);
  row_group_offsets[0]   = static_cast<size_t>(std::llround(1e9));
  std::transform(
    cuda::counting_iterator(1),
    cuda::counting_iterator(num_row_groups),
    row_group_offsets.begin() + 1,
    [&](auto i) { return static_cast<size_t>(std::llround(row_group_offsets[i - 1] + 0.5e9)); });

  // Split num_rows into spans
  auto row_group_splits = std::vector<cudf::size_type>(num_row_groups - 1);
  {
    std::mt19937 engine{0xdeaL};
    std::uniform_int_distribution<cudf::size_type> dist{1, num_rows};
    std::generate(row_group_splits.begin(), row_group_splits.end(), [&]() { return dist(engine); });
    std::sort(row_group_splits.begin(), row_group_splits.end());
  }

  auto row_group_num_rows = std::vector<cudf::size_type>{};
  {
    row_group_num_rows.reserve(num_row_groups);
    auto previous_split = cudf::size_type{0};
    std::transform(row_group_splits.begin(),
                   row_group_splits.end(),
                   std::back_inserter(row_group_num_rows),
                   [&](auto current_split) {
                     auto current_split_size = current_split - previous_split;
                     previous_split          = current_split;
                     return current_split_size;
                   });
    row_group_num_rows.push_back(num_rows - row_group_splits.back());
  }

  auto expected_row_indices =
    build_expected_row_indices(row_group_offsets, row_group_num_rows, num_rows);

  auto [deletion_vector, expected_row_mask_column] = build_roaring_bitmap_and_expected_row_mask(
    num_rows, deletion_probability, expected_row_indices, stream, mr);

  auto const expected_row_mask = cudf::detail::make_host_vector(
    cudf::device_span<bool const>(expected_row_mask_column->view().data<bool>(), num_rows), stream);
  size_t const expected_deleted =
    std::count(expected_row_mask.begin(), expected_row_mask.end(), false);

  auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
    .serialized_roaring_bitmaps = {deletion_vector},
    .deletion_vector_row_counts = {num_rows},
    .row_group_offsets          = row_group_offsets,
    .row_group_num_rows         = row_group_num_rows};

  for (auto chunk_size :
       {num_rows, static_cast<cudf::size_type>(std::llround(row_group_splits.front() * 1.2))}) {
    auto const num_deleted_rows = cudf::io::parquet::experimental::compute_num_deleted_rows(
      deletion_vector_info, chunk_size, stream);
    EXPECT_EQ(num_deleted_rows, expected_deleted);
  }
}

TEST_F(DeletionVectorsCountTests, MultipleDeletionVectors)
{
  auto constexpr num_rows_per_dv      = 10'000;
  auto constexpr num_deletion_vectors = 5;
  auto constexpr num_rows             = num_rows_per_dv * num_deletion_vectors;
  auto constexpr deletion_probability = 0.3;

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Each deletion vector covers a separate row group with a distinct offset
  auto row_group_offsets  = std::vector<size_t>(num_deletion_vectors);
  auto row_group_num_rows = std::vector<cudf::size_type>(num_deletion_vectors, num_rows_per_dv);
  std::transform(cuda::counting_iterator(0),
                 cuda::counting_iterator(num_deletion_vectors),
                 row_group_offsets.begin(),
                 [](auto i) { return static_cast<size_t>(i) * 100'000; });

  auto expected_row_indices =
    build_expected_row_indices(row_group_offsets, row_group_num_rows, num_rows);

  size_t total_expected_deleted = 0;
  auto serialized_bitmaps       = std::vector<std::vector<cuda::std::byte>>{};
  auto bitmap_spans             = std::vector<cudf::host_span<cuda::std::byte const>>{};
  auto dv_row_counts            = std::vector<cudf::size_type>{};

  for (int i = 0; i < num_deletion_vectors; ++i) {
    auto span_start = i * num_rows_per_dv;
    auto local_indices =
      cudf::host_span<size_t const>(expected_row_indices.data() + span_start, num_rows_per_dv);

    auto [dv, mask_col] = build_roaring_bitmap_and_expected_row_mask(
      num_rows_per_dv, deletion_probability, local_indices, stream, mr);

    auto const host_mask = cudf::detail::make_host_vector(
      cudf::device_span<bool const>(mask_col->view().data<bool>(), num_rows_per_dv), stream);
    total_expected_deleted += std::count(host_mask.begin(), host_mask.end(), false);

    serialized_bitmaps.emplace_back(std::move(dv));
    dv_row_counts.push_back(num_rows_per_dv);
  }

  bitmap_spans.reserve(num_deletion_vectors);
  for (auto const& bm : serialized_bitmaps) {
    bitmap_spans.emplace_back(bm);
  }

  auto deletion_vector_info = cudf::io::parquet::experimental::deletion_vector_info{
    .serialized_roaring_bitmaps = bitmap_spans,
    .deletion_vector_row_counts = dv_row_counts,
    .row_group_offsets          = row_group_offsets,
    .row_group_num_rows         = row_group_num_rows};

  // All rows in one chunk
  for (auto chunk_size :
       {num_rows, static_cast<cudf::size_type>(std::llround(num_rows_per_dv * 1.2))}) {
    auto const result = cudf::io::parquet::experimental::compute_num_deleted_rows(
      deletion_vector_info, chunk_size, stream);
    EXPECT_EQ(result, total_expected_deleted);
  }
}
