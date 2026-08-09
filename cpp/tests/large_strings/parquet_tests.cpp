/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/iterator>

namespace {

cudf::test::TempDirTestEnvironment* const g_temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

}  // namespace

struct ParquetStringsTest : public cudf::test::StringsLargeTest {};

TEST_F(ParquetStringsTest, ReadLargeStrings)
{
  // need to create a string column larger than `threshold`
  auto const col0        = this->long_column();
  auto const column_size = cudf::strings_column_view(col0).chars_size(cudf::get_default_stream());
  auto const threshold   = column_size - 1;
  auto const expected    = cudf::table_view{{col0, col0, col0}};

  auto expected_metadata = cudf::io::table_input_metadata{expected};
  expected_metadata.column_metadata[1].set_encoding(
    cudf::io::column_encoding::DELTA_LENGTH_BYTE_ARRAY);
  expected_metadata.column_metadata[2].set_encoding(cudf::io::column_encoding::DELTA_BYTE_ARRAY);

  // set smaller threshold to reduce file size and execution time
  setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1);

  auto const filepath = g_temp_env->get_temp_filepath("ReadLargeStrings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD)
      .stats_level(cudf::io::STATISTICS_NONE)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result      = cudf::io::read_parquet(default_in_opts);
  auto const result_view = result.tbl->view();
  for (auto cv : result_view) {
    auto const offsets = cudf::strings_column_view(cv).offsets();
    EXPECT_EQ(offsets.type(), cudf::data_type{cudf::type_id::INT64});
  }
  CUDF_TEST_EXPECT_TABLES_EQUAL(result_view, expected);

  // go back to normal threshold
  unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
}

// Disabled as the test is too brittle and depends on empirically set `pass_read_limit`,
// encoding type, and the currently used `ZSTD` scratch space size.
TEST_F(ParquetStringsTest, DISABLED_ChunkedReadLargeStrings)
{
  // Construct a table with one large strings column > 2GB
  auto const wide = this->wide_column();
  auto input      = cudf::concatenate(std::vector<cudf::column_view>(120000, wide));  ///< 230MB

  int constexpr multiplier = 12;
  std::vector<cudf::column_view> input_cols(multiplier, input->view());
  auto col0 = cudf::concatenate(input_cols);  ///< 2.70GB

  // Expected table
  auto const expected    = cudf::table_view{{col0->view()}};
  auto expected_metadata = cudf::io::table_input_metadata{expected};

  // Needed to get exactly 2 Parquet subpasses: first with large-strings and the second with
  // regualar ones. This may change in the future and lead to false failures.
  expected_metadata.column_metadata[0].set_encoding(
    cudf::io::column_encoding::DELTA_LENGTH_BYTE_ARRAY);

  // Host buffer to write Parquet
  std::vector<char> buffer;

  // Writer options
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, expected)
      .metadata(expected_metadata);

  // Needed to get exactly 2 Parquet subpasses: first with large-strings and the second with
  // regualar ones. This may change in the future and lead to false failures.
  out_opts.set_compression(cudf::io::compression_type::ZSTD);

  // Write to Parquet
  cudf::io::write_parquet(out_opts);

  // Empirically set pass_read_limit of 8GB so we read almost entire table (>2GB strings) in the
  // first subpass and only a small amount in the second subpass. This may change in the future
  // and lead to false failures.
  size_t constexpr pass_read_limit = size_t{8} * 1024 * 1024 * 1024;

  // Reader options
  cudf::io::parquet_reader_options default_in_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}});

  // Chunked parquet reader
  auto reader = cudf::io::chunked_parquet_reader(0, pass_read_limit, default_in_opts);

  // Read chunked
  auto tables = std::vector<std::unique_ptr<cudf::table>>{};
  while (reader.has_next()) {
    tables.emplace_back(reader.read_chunk().tbl);
  }
  auto table_views = std::vector<cudf::table_view>{};
  std::transform(tables.begin(), tables.end(), std::back_inserter(table_views), [](auto& tbl) {
    return tbl->view();
  });
  auto result            = cudf::concatenate(table_views);
  auto const result_view = result->view();

  // Verify offsets
  for (auto const& cv : result_view) {
    auto const offsets = cudf::strings_column_view(cv).offsets();
    EXPECT_EQ(offsets.type(), cudf::data_type{cudf::type_id::INT64});
  }

  // Verify tables to be equal
  CUDF_TEST_EXPECT_TABLES_EQUAL(result_view, expected);

  // Verify that we read exactly two table chunks
  EXPECT_EQ(tables.size(), 2);
}

TEST_F(ParquetStringsTest, ChunkedReadNestedLargeStrings)
{
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using strings_col = cudf::test::strings_column_wrapper;
  using structs_col = cudf::test::structs_column_wrapper;

  auto constexpr num_rows = 100'000;

  std::vector<std::unique_ptr<cudf::column>> input_columns;
  auto const int_iter = cuda::counting_iterator<int32_t>{0};
  input_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());

  auto const str_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](int32_t i) { return std::to_string(i) + std::to_string(i) + std::to_string(i); });
  input_columns.emplace_back(strings_col{str_iter, str_iter + num_rows}.release());

  auto offsets = std::vector<cudf::size_type>{};
  offsets.reserve(num_rows * 2);
  cudf::size_type num_structs = 0;
  for (int i = 0; i < num_rows; ++i) {
    offsets.push_back(num_structs);
    auto const new_list_size = i % 4;
    num_structs += new_list_size;
  }
  offsets.push_back(num_structs);

  auto const make_structs_col = [=] {
    auto child1 = int32s_col(int_iter, int_iter + num_structs);
    auto child2 = int32s_col(int_iter + num_structs, int_iter + num_structs * 2);
    auto child3 = strings_col{str_iter, str_iter + num_structs};

    return structs_col{{child1, child2, child3}}.release();
  };

  input_columns.emplace_back(
    cudf::make_lists_column(static_cast<cudf::size_type>(offsets.size() - 1),
                            cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                              offsets.begin(), offsets.end())
                              .release(),
                            make_structs_col(),
                            0,
                            rmm::device_buffer{}));

  // Input table
  auto const table    = cudf::table{std::move(input_columns)};
  auto const expected = table.view();

  auto const child3_view = expected.column(2).child(1).child(2);  // list<struct<int,int,string>>
  auto const column_size =
    cudf::strings_column_view(child3_view).chars_size(cudf::get_default_stream());
  // set smaller threshold to reduce file size and execution time
  auto const threshold =
    column_size / 16;  // Empirically set to get a mix of 32 and 64 bit string col chunks.
  setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1);

  // Host buffer to write Parquet
  auto buffer = std::vector<char>{};
  // Writer options
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, expected)
      .max_page_size_bytes(512 * 1024)
      .max_page_size_rows(20000)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .write_v2_headers(false);

  // Write to Parquet
  cudf::io::write_parquet(out_opts);

  // Reader options
  cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}});

  auto constexpr chunk_read_limit = size_t{1} * 1024 * 1024;
  auto constexpr pass_read_limit  = 0;
  // Chunked parquet reader
  auto reader = cudf::io::chunked_parquet_reader(chunk_read_limit, pass_read_limit, in_opts);

  // Read chunked
  auto tables = std::vector<std::unique_ptr<cudf::table>>{};
  while (reader.has_next()) {
    tables.emplace_back(reader.read_chunk().tbl);
  }
  auto table_views = std::vector<cudf::table_view>{};
  std::transform(tables.begin(), tables.end(), std::back_inserter(table_views), [](auto& tbl) {
    return tbl->view();
  });
  auto result = cudf::concatenate(table_views);

  // Verify tables to be equal
  CUDF_TEST_EXPECT_TABLES_EQUAL(result->view(), expected);

  // go back to normal threshold
  unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
}
