/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "deletion_vectors_helpers.hpp"
#include "io/parquet/reader_impl_helpers.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/io/experimental/deletion_vectors.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/roaring_bitmap.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>

#include <numeric>

namespace cudf::io::parquet::experimental {

namespace {

// We are only working with 64-bit roaring bitmaps here
auto constexpr roaring_bitmap_type = cudf::roaring_bitmap_type::BITS_64;

}  // namespace

namespace detail {

[[nodiscard]] table_with_metadata read_parquet(parquet_reader_options const& options,
                                               deletion_vector_info const& deletion_vector_info,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  auto const& serialized_roaring_bitmaps = deletion_vector_info.serialized_roaring_bitmaps;
  auto const& deletion_vector_row_counts = deletion_vector_info.deletion_vector_row_counts;
  auto const& row_group_offsets          = deletion_vector_info.row_group_offsets;
  auto const& row_group_num_rows         = deletion_vector_info.row_group_num_rows;

  CUDF_EXPECTS(
    row_group_offsets.size() == row_group_num_rows.size(),
    "Encountered a mismatch in the number of row group offsets and row group row counts");
  CUDF_EXPECTS(
    not options.get_filter().has_value(),
    "Encountered a non-empty AST filter expression. Use a roaring64 bitmap deletion vector to "
    "filter the table instead");
  CUDF_EXPECTS(serialized_roaring_bitmaps.empty() or
                 serialized_roaring_bitmaps.size() == deletion_vector_row_counts.size(),
               "Encountered a mismatch in the number of deletion vectors and the number of rows "
               "per deletion vector");

  // Use default mr to read parquet table and build row index column if we will be applying the
  // deletion vector to produce a new table later
  auto const table_mr =
    serialized_roaring_bitmaps.empty() ? mr : cudf::get_current_device_resource_ref();

  auto [table, metadata] = cudf::io::read_parquet(options, stream, table_mr);
  auto const num_rows    = table->num_rows();

  CUDF_EXPECTS(
    row_group_num_rows.empty() or
      std::cmp_equal(metadata.num_input_row_groups, row_group_num_rows.size()),
    "Encountered a mismatch in the number of row groups in parquet file and the specified "
    "row group offsets/row counts vectors");

  // Compute a row index column from the specified row group offsets and counts
  auto row_index_column = compute_row_index_column(
    row_group_offsets, row_group_num_rows, std::nullopt, num_rows, stream, table_mr);

  // Prepend row index column to the table columns
  auto table_with_index =
    prepend_index_column_to_table(std::move(table), std::move(row_index_column));

  prepend_index_column_to_table_metadata(metadata);

  if (serialized_roaring_bitmaps.empty()) {
    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  }

  // Construct all deletion vectors and store their references
  auto deletion_vectors     = std::vector<cudf::roaring_bitmap>{};
  auto deletion_vector_refs = std::vector<std::reference_wrapper<cudf::roaring_bitmap const>>{};
  // Must reserve enough space for the deletion vectors and references to avoid implicit
  // re-allocations in emplace_back leading to dangling references
  deletion_vectors.reserve(serialized_roaring_bitmaps.size());
  deletion_vector_refs.reserve(serialized_roaring_bitmaps.size());
  std::transform(serialized_roaring_bitmaps.begin(),
                 serialized_roaring_bitmaps.end(),
                 std::back_inserter(deletion_vector_refs),
                 [&](auto const& serialized_roaring_bitmap) {
                   deletion_vectors.emplace_back(roaring_bitmap_type, serialized_roaring_bitmap);
                   return std::ref(deletion_vectors.back());
                 });
  auto row_mask = compute_row_mask_column(table_with_index->get_column(0).view(),
                                          deletion_vector_refs,
                                          deletion_vector_row_counts,
                                          stream,
                                          cudf::get_current_device_resource_ref());

  auto const mask_type = deletion_vector_info.are_retention_vectors
                           ? cudf::detail::mask_type::RETENTION
                           : cudf::detail::mask_type::DELETION;

  // Filter the table using the deletion vector
  return cudf::io::table_with_metadata{
    // Supply user-provided mr to apply deletion mask to allocate output table's memory
    cudf::detail::apply_mask(table_with_index->view(), row_mask->view(), mask_type, stream, mr),
    std::move(metadata)};
}

[[nodiscard]] size_t compute_num_deleted_rows(deletion_vector_info const& deletion_vector_info,
                                              cudf::size_type max_chunk_rows,
                                              rmm::cuda_stream_view stream)
{
  auto const& serialized_roaring_bitmaps = deletion_vector_info.serialized_roaring_bitmaps;
  auto const& deletion_vector_row_counts = deletion_vector_info.deletion_vector_row_counts;
  auto const& row_group_offsets          = deletion_vector_info.row_group_offsets;
  auto const& row_group_num_rows         = deletion_vector_info.row_group_num_rows;

  if (serialized_roaring_bitmaps.empty()) { return 0; }

  // Validate input
  CUDF_EXPECTS(std::cmp_equal(serialized_roaring_bitmaps.size(), deletion_vector_row_counts.size()),
               "Encountered a mismatch in the number of deletion vector data spans and the number "
               "of rows per deletion vector");
  CUDF_EXPECTS(
    row_group_offsets.size() == row_group_num_rows.size(),
    "Encountered a mismatch in the number of row group offsets and row group row counts");
  CUDF_EXPECTS(max_chunk_rows > 0, "Encountered an invalid chunk size");

  auto const is_row_group_data_unspecified = row_group_offsets.empty();

  auto const num_rows = [&]() {
    auto const rows_in_dvs = std::accumulate(
      deletion_vector_row_counts.begin(), deletion_vector_row_counts.end(), size_t{0});
    if (not is_row_group_data_unspecified) {
      auto const rows_in_rgs =
        std::accumulate(row_group_num_rows.begin(), row_group_num_rows.end(), size_t{0});
      CUDF_EXPECTS(std::cmp_equal(rows_in_dvs, rows_in_rgs),
                   "Encountered a mismatch in the number of rows across deletion vectors and the "
                   "number of rows across row groups");
    }
    return rows_in_dvs;
  }();

  std::queue<size_t> rg_offsets_queue;
  std::queue<size_type> rg_counts_queue;
  for (size_t i = 0; i < row_group_offsets.size(); ++i) {
    rg_offsets_queue.push(row_group_offsets[i]);
    rg_counts_queue.push(row_group_num_rows[i]);
  }

  std::queue<cudf::roaring_bitmap> dv_queue;
  std::queue<cudf::size_type> dv_row_counts_queue;
  for (size_t i = 0; i < serialized_roaring_bitmaps.size(); ++i) {
    dv_queue.emplace(roaring_bitmap_type, serialized_roaring_bitmaps[i]);
    dv_row_counts_queue.push(deletion_vector_row_counts[i]);
  }

  size_t matched_rows   = 0;
  size_t remaining_rows = num_rows;
  size_t start_row      = 0;

  while (remaining_rows > 0) {
    // Maximum number of rows to process in this chunk
    auto const chunk_rows = std::min<size_t>(remaining_rows, max_chunk_rows);
    auto row_index_column =
      compute_partial_row_index_column(rg_offsets_queue,
                                       rg_counts_queue,
                                       start_row,
                                       static_cast<size_type>(chunk_rows),
                                       is_row_group_data_unspecified,
                                       stream,
                                       cudf::get_current_device_resource_ref());
    matched_rows += compute_partial_deleted_row_count(
      row_index_column->view(), dv_queue, dv_row_counts_queue, stream);

    start_row += chunk_rows;
    remaining_rows -= chunk_rows;
  }

  // Bitmap hits are deleted rows for deletion vectors, retained rows for retention vectors
  return deletion_vector_info.are_retention_vectors ? num_rows - matched_rows : matched_rows;
}

}  // namespace detail

/**
 * @copydoc
 * cudf::io::parquet::experimental::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(std::size_t chunk_read_limit,
                                               std::size_t pass_read_limit,
                                               parquet_reader_options const& options,
                                               deletion_vector_info const& deletion_vector_info,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
  : _start_row{0},
    _is_unspecified_row_group_data{deletion_vector_info.row_group_offsets.empty()},
    _are_retentions{deletion_vector_info.are_retention_vectors},
    _stream{stream},
    _mr{mr},
    // Use default mr for the internal chunked reader and row index column if we will
    // be applying the deletion vector to produce the output table
    _table_mr{deletion_vector_info.serialized_roaring_bitmaps.empty()
                ? mr
                : cudf::get_current_device_resource_ref()}
{
  auto const& serialized_roaring_bitmaps = deletion_vector_info.serialized_roaring_bitmaps;
  auto const& deletion_vector_row_counts = deletion_vector_info.deletion_vector_row_counts;
  auto const& row_group_offsets          = deletion_vector_info.row_group_offsets;
  auto const& row_group_num_rows         = deletion_vector_info.row_group_num_rows;

  CUDF_EXPECTS(
    row_group_offsets.size() == row_group_num_rows.size(),
    "Encountered a mismatch in the number of row group offsets and row group row counts");
  CUDF_EXPECTS(
    not options.get_filter().has_value(),
    "Encountered a non-empty AST filter expression. Use a roaring64 bitmap deletion vector to "
    "filter the table instead");
  CUDF_EXPECTS(serialized_roaring_bitmaps.empty() or
                 serialized_roaring_bitmaps.size() == deletion_vector_row_counts.size(),
               "Encountered a mismatch in the number of deletion vectors and the number of rows "
               "per deletion vector");

  // Initialize the internal chunked parquet reader
  _reader = std::make_unique<cudf::io::chunked_parquet_reader>(
    chunk_read_limit, pass_read_limit, options, _stream, _table_mr);

  // Push row group offsets and counts to the internal queues
  if (not row_group_offsets.empty()) {
    auto iter = cuda::make_zip_iterator(row_group_offsets.begin(), row_group_num_rows.begin());
    std::for_each(iter, iter + row_group_offsets.size(), [&](auto const& elem) {
      _row_group_row_offsets.push(cuda::std::get<0>(elem));
      _row_group_row_counts.push(cuda::std::get<1>(elem));
    });
  }

  // Push deletion vector data spans and row counts to the internal queues
  if (not serialized_roaring_bitmaps.empty()) {
    auto iter = cuda::make_zip_iterator(serialized_roaring_bitmaps.begin(),
                                        deletion_vector_row_counts.begin());
    std::for_each(iter, iter + serialized_roaring_bitmaps.size(), [&](auto const& elem) {
      _deletion_vectors.emplace(roaring_bitmap_type, cuda::std::get<0>(elem));
      _deletion_vector_row_counts.push(cuda::std::get<1>(elem));
    });
  }
}

/**
 * @copydoc
 * cudf::io::parquet::experimental::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(std::size_t chunk_read_limit,
                                               parquet_reader_options const& options,
                                               deletion_vector_info const& deletion_vector_info,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
  : chunked_parquet_reader(chunk_read_limit,
                           parquet::detail::derive_pass_read_limit(chunk_read_limit),
                           options,
                           deletion_vector_info,
                           stream,
                           mr)
{
}

/**
 * @copydoc cudf::io::parquet::experimental::chunked_parquet_reader::~chunked_parquet_reader
 */
chunked_parquet_reader::~chunked_parquet_reader() = default;

/**
 * @copydoc cudf::io::parquet::experimental::chunked_parquet_reader::has_next
 */
bool chunked_parquet_reader::has_next() const { return _reader->has_next(); }

/**
 * @copydoc cudf::io::parquet::experimental::chunked_parquet_reader::read_chunk
 */
table_with_metadata chunked_parquet_reader::read_chunk()
{
  CUDF_FUNC_RANGE();

  // Read a chunk of the parquet table
  auto [table, metadata] = _reader->read_chunk();
  auto const num_rows    = table->num_rows();

  // Compute a chunk of the row index column from the specified row group offsets and counts
  auto row_index_column = compute_partial_row_index_column(_row_group_row_offsets,
                                                           _row_group_row_counts,
                                                           _start_row,
                                                           num_rows,
                                                           _is_unspecified_row_group_data,
                                                           _stream,
                                                           _table_mr);
  // Update the start row index for the next chunk
  _start_row += num_rows;

  // Prepend row index column to the table columns
  auto table_with_index =
    prepend_index_column_to_table(std::move(table), std::move(row_index_column));

  // Also prepend the row index column's metadata to the table schema
  prepend_index_column_to_table_metadata(metadata);

  // Return early if deletion vector is not present
  if (_deletion_vectors.empty()) {
    return table_with_metadata{std::move(table_with_index), std::move(metadata)};
  }

  // Filter the table using the deletion vectors
  auto row_mask = compute_partial_row_mask_column(table_with_index->get_column(0).view(),
                                                  _deletion_vectors,
                                                  _deletion_vector_row_counts,
                                                  _stream,
                                                  cudf::get_current_device_resource_ref());
  auto const mask_type =
    _are_retentions ? cudf::detail::mask_type::RETENTION : cudf::detail::mask_type::DELETION;
  return cudf::io::table_with_metadata{
    // Supply user-provided mr to apply deletion mask to allocate output table's memory
    cudf::detail::apply_mask(table_with_index->view(), row_mask->view(), mask_type, _stream, _mr),
    std::move(metadata)};
}

/**
 * @copydoc cudf::io::parquet::experimental::read_parquet
 */
table_with_metadata read_parquet(parquet_reader_options const& options,
                                 deletion_vector_info const& deletion_vector_info,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::read_parquet(options, deletion_vector_info, stream, mr);
}

/**
 * @copydoc cudf::io::parquet::experimental::compute_num_deleted_rows
 */
size_t compute_num_deleted_rows(deletion_vector_info const& deletion_vector_info,
                                cudf::size_type max_chunk_rows,
                                rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::compute_num_deleted_rows(deletion_vector_info, max_chunk_rows, stream);
}

}  // namespace cudf::io::parquet::experimental
