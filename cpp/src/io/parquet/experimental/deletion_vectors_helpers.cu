/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "deletion_vectors_helpers.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

namespace cudf::io::parquet::experimental {

void prepend_index_column_to_table_metadata(table_metadata& metadata)
{
  auto updated_schema_info = std::vector<cudf::io::column_name_info>{};
  updated_schema_info.reserve(metadata.schema_info.size() + 1);
  updated_schema_info.emplace_back("index");
  updated_schema_info.insert(updated_schema_info.end(),
                             std::make_move_iterator(metadata.schema_info.begin()),
                             std::make_move_iterator(metadata.schema_info.end()));
  metadata.schema_info = std::move(updated_schema_info);
}

std::unique_ptr<cudf::table> prepend_index_column_to_table(
  std::unique_ptr<cudf::table>&& table, std::unique_ptr<cudf::column>&& row_index_column)
{
  auto index_and_table_columns = std::vector<std::unique_ptr<cudf::column>>{};
  index_and_table_columns.reserve(table->num_columns() + 1);
  index_and_table_columns.push_back(std::move(row_index_column));
  auto table_columns = table->release();
  index_and_table_columns.insert(index_and_table_columns.end(),
                                 std::make_move_iterator(table_columns.begin()),
                                 std::make_move_iterator(table_columns.end()));
  return std::make_unique<cudf::table>(std::move(index_and_table_columns));
}

std::unique_ptr<cudf::column> compute_row_index_column(
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  std::optional<size_t> start_row,
  size_type num_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_row_groups = static_cast<size_type>(row_group_num_rows.size());

  if (row_group_offsets.empty()) {
    auto row_indices      = rmm::device_buffer(num_rows * sizeof(size_t), stream, mr);
    auto row_indices_iter = static_cast<size_t*>(row_indices.data());
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     row_indices_iter,
                     row_indices_iter + num_rows,
                     start_row.value_or(size_t{0}));
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
                                          num_rows,
                                          std::move(row_indices),
                                          rmm::device_buffer{0, stream, mr},
                                          0);
  }

  auto row_group_span_offsets =
    cudf::detail::make_host_vector<size_type>(num_row_groups + 1, stream);
  row_group_span_offsets[0] = 0;
  std::inclusive_scan(
    row_group_num_rows.begin(), row_group_num_rows.end(), row_group_span_offsets.begin() + 1);

  CUDF_EXPECTS(row_group_span_offsets.back() == num_rows,
               "Encountered a mismatch in the number of rows in the row index column and the "
               "number of rows in the row group(s)");

  auto row_indices      = rmm::device_buffer(num_rows * sizeof(size_t), stream, mr);
  auto row_indices_iter = static_cast<size_t*>(row_indices.data());
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               row_indices_iter,
               row_indices_iter + num_rows,
               1);

  auto row_group_keys = rmm::device_uvector<size_type>(num_rows, stream);
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               row_group_keys.begin(),
               row_group_keys.end(),
               0);

  auto d_row_group_offsets = cudf::detail::make_device_uvector_async(
    row_group_offsets, stream, cudf::get_current_device_resource_ref());
  auto d_row_group_span_offsets = cudf::detail::make_device_uvector_async(
    row_group_span_offsets, stream, cudf::get_current_device_resource_ref());
  auto in_iter =
    cuda::make_zip_iterator(d_row_group_offsets.begin(), cuda::counting_iterator<size_type>(0));
  auto out_iter = cuda::make_zip_iterator(row_indices_iter, row_group_keys.begin());
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  in_iter,
                  in_iter + num_row_groups,
                  d_row_group_span_offsets.begin(),
                  out_iter);

  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         row_group_keys.begin(),
                         row_group_keys.end(),
                         row_group_keys.begin(),
                         cuda::maximum<cudf::size_type>());

  thrust::inclusive_scan_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    row_group_keys.begin(),
    row_group_keys.end(),
    row_indices_iter,
    row_indices_iter);

  stream.sync();
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
                                        num_rows,
                                        std::move(row_indices),
                                        rmm::device_buffer{0, stream, mr},
                                        0);
}

std::unique_ptr<cudf::column> compute_partial_row_index_column(
  std::queue<size_t>& row_group_offsets,
  std::queue<size_type>& row_group_num_rows,
  size_t start_row,
  size_type num_rows,
  bool is_unspecified_row_group_data,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (is_unspecified_row_group_data) {
    return compute_row_index_column({}, {}, start_row, num_rows, stream, mr);
  }

  std::vector<size_t> row_offsets;
  std::vector<size_type> row_counts;
  size_type rows_filled = 0;
  while (std::cmp_less(rows_filled, num_rows)) {
    CUDF_EXPECTS(
      not(row_group_offsets.empty() or row_group_num_rows.empty()),
      "Unable to compute the row index column from the specified row group offsets and counts");

    auto const row_count = std::min<size_type>(num_rows - rows_filled, row_group_num_rows.front());
    row_counts.emplace_back(row_count);
    row_offsets.emplace_back(row_group_offsets.front());

    if (std::cmp_less(row_count, row_group_num_rows.front())) {
      row_group_offsets.front()  = row_group_offsets.front() + row_count;
      row_group_num_rows.front() = row_group_num_rows.front() - row_count;
    } else {
      row_group_offsets.pop();
      row_group_num_rows.pop();
    }

    rows_filled += row_count;
  }

  return compute_row_index_column(row_offsets, row_counts, std::nullopt, num_rows, stream, mr);
}

namespace {

/**
 * @brief Queries the deletion vectors and writes the membership result into the specified BOOL8
 * mutable column view
 *
 * @param row_index_column Row index column
 * @param deletion_vector_refs Deletion vector refs
 * @param rows_per_deletion_vector Rows per deletion vector
 * @param output BOOL8 mutable column view to receive membership results
 * @param stream CUDA stream to launch the query kernel
 */
void query_deletion_vectors(
  cudf::column_view const& row_index_column,
  cudf::host_span<std::reference_wrapper<cudf::roaring_bitmap const> const> deletion_vector_refs,
  cudf::host_span<size_type const> rows_per_deletion_vector,
  cudf::mutable_column_view const& output,
  cuda::stream_ref stream)
{
  auto const num_rows             = row_index_column.size();
  auto const num_deletion_vectors = static_cast<cudf::size_type>(deletion_vector_refs.size());

  if (num_deletion_vectors == 1) {
    CUDF_EXPECTS(
      std::cmp_greater_equal(rows_per_deletion_vector.front(), num_rows),
      "Encountered insufficient deletion vector size to query the entire row index column");
    deletion_vector_refs.front().get().contains_async(row_index_column, output, stream);
    return;
  }

  auto deletion_vector_row_offsets = std::vector<size_type>(deletion_vector_refs.size() + 1);
  deletion_vector_row_offsets[0]   = 0;
  std::inclusive_scan(rows_per_deletion_vector.begin(),
                      rows_per_deletion_vector.end(),
                      deletion_vector_row_offsets.begin() + 1);

  CUDF_EXPECTS(deletion_vector_row_offsets.back() == num_rows,
               "Encountered a mismatch in the number of rows in the row index column and the "
               "number of rows in the deletion vector(s)");

  constexpr auto stream_fork_threshold = 8;
  if (num_deletion_vectors >= stream_fork_threshold) {
    auto streams = cudf::detail::fork_streams(stream, num_deletion_vectors);
    std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                  cuda::counting_iterator{num_deletion_vectors},
                  [&](auto const dv_idx) {
                    auto const begin = deletion_vector_row_offsets[dv_idx];
                    auto const end   = deletion_vector_row_offsets[dv_idx + 1];
                    deletion_vector_refs[dv_idx].get().contains_async(
                      cudf::detail::slice(row_index_column, begin, end, streams[dv_idx]),
                      cudf::detail::slice(output, begin, end, streams[dv_idx]),
                      streams[dv_idx]);
                  });
    cudf::detail::join_streams(streams, stream);
  } else {
    std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                  cuda::counting_iterator{num_deletion_vectors},
                  [&](auto const dv_idx) {
                    auto const begin = deletion_vector_row_offsets[dv_idx];
                    auto const end   = deletion_vector_row_offsets[dv_idx + 1];
                    deletion_vector_refs[dv_idx].get().contains_async(
                      cudf::detail::slice(row_index_column, begin, end, stream),
                      cudf::detail::slice(output, begin, end, stream),
                      stream);
                  });
  }
}

}  // namespace

std::unique_ptr<cudf::column> compute_row_mask_column(
  cudf::column_view const& row_index_column,
  cudf::host_span<std::reference_wrapper<cudf::roaring_bitmap const> const> deletion_vector_refs,
  cudf::host_span<size_type const> rows_per_deletion_vector,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows  = row_index_column.size();
  auto row_mask_column = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto const row_mask_view = row_mask_column->mutable_view();

  query_deletion_vectors(
    row_index_column, deletion_vector_refs, rows_per_deletion_vector, row_mask_view, stream);

  return row_mask_column;
}

size_t compute_deleted_row_count(
  cudf::column_view const& row_index_column,
  cudf::host_span<std::reference_wrapper<cudf::roaring_bitmap const> const> deletion_vector_refs,
  cudf::host_span<size_type const> deletion_vector_row_counts,
  cuda::stream_ref stream)
{
  auto row_mask_column = compute_row_mask_column(row_index_column,
                                                 deletion_vector_refs,
                                                 deletion_vector_row_counts,
                                                 stream,
                                                 cudf::get_current_device_resource_ref());
  return static_cast<size_t>(
    thrust::count(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  row_mask_column->view().begin<bool>(),
                  row_mask_column->view().end<bool>(),
                  true));
}

namespace {

/**
 * @brief Consumes a specified number of rows from a queue of deletion vectors into vectors of
 * currently active deletion vectors and their corresponding references and row counts
 *
 * Fully consumed deletion vectors are moved out of the input queue into a returned owning vector
 * so that references into them remain valid until the caller is done. References are taken
 * against the stable storage of either the input queue (for partially-consumed vectors) or the
 * returned owning vector (for fully-consumed vectors). The caller must reserve enough space in
 * the returned vectors to avoid implicit reallocations that would invalidate references.
 *
 * @param num_rows The number of rows to consume
 * @param[in,out] deletion_vectors Queue of deletion vectors to consume from
 * @param[in,out] deletion_vector_row_counts Queue of per-deletion-vector row counts
 * @return Tuple of (owning storage for fully-consumed deletion vectors, references to active
 * deletion vectors, row counts per active deletion vector)
 */
auto consume_deletion_vectors(size_type num_rows,
                              std::queue<cudf::roaring_bitmap>& deletion_vectors,
                              std::queue<size_type>& deletion_vector_row_counts)
{
  std::vector<size_type> row_counts;
  std::vector<cudf::roaring_bitmap> temp_deletion_vectors;
  std::vector<std::reference_wrapper<cudf::roaring_bitmap const>> deletion_vector_refs;
  // Must reserve enough space for the temporary deletion vectors and references to avoid implicit
  // re-allocations in emplace_back leading to dangling references
  row_counts.reserve(deletion_vectors.size());
  temp_deletion_vectors.reserve(deletion_vectors.size());
  deletion_vector_refs.reserve(deletion_vectors.size());
  size_type rows_filled = 0;

  while (std::cmp_less(rows_filled, num_rows)) {
    CUDF_EXPECTS(
      not(deletion_vector_row_counts.empty() or deletion_vectors.empty()),
      "Encountered insufficient number of deletion vector row counts or deletion vectors: " +
        std::to_string(deletion_vector_row_counts.size()) + " " +
        std::to_string(deletion_vectors.size()));

    // Compute how many rows can be queried from the current roaring bitmap
    auto const row_count =
      std::min<size_type>(num_rows - rows_filled, deletion_vector_row_counts.front());
    row_counts.emplace_back(row_count);

    // If we still have remaining rows to query from the current roaring bitmap, update its row
    // count and take a reference into the queue's stable storage
    if (std::cmp_less(row_count, deletion_vector_row_counts.front())) {
      deletion_vector_row_counts.front() = deletion_vector_row_counts.front() - row_count;
      deletion_vector_refs.emplace_back(std::ref(deletion_vectors.front()));
    } else {
      // Else if the deletion vector is fully queried, move it to the temporary vector and pop it
      // from the queues so that the reference remains valid for the caller
      temp_deletion_vectors.emplace_back(std::move(deletion_vectors.front()));
      deletion_vector_refs.emplace_back(std::ref(temp_deletion_vectors.back()));
      deletion_vectors.pop();
      deletion_vector_row_counts.pop();
    }

    rows_filled += row_count;
  }

  return std::tuple{
    std::move(temp_deletion_vectors), std::move(deletion_vector_refs), std::move(row_counts)};
}

}  // namespace

std::unique_ptr<cudf::column> compute_partial_row_mask_column(
  cudf::column_view const& row_index_column,
  std::queue<cudf::roaring_bitmap>& deletion_vectors,
  std::queue<size_type>& deletion_vector_row_counts,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto [_, dv_refs, dv_row_counts] =
    consume_deletion_vectors(row_index_column.size(), deletion_vectors, deletion_vector_row_counts);
  return compute_row_mask_column(row_index_column, dv_refs, dv_row_counts, stream, mr);
}

size_t compute_partial_deleted_row_count(cudf::column_view const& row_index_column,
                                         std::queue<cudf::roaring_bitmap>& deletion_vectors,
                                         std::queue<size_type>& deletion_vector_row_counts,
                                         cuda::stream_ref stream)
{
  auto [_, dv_refs, dv_row_counts] =
    consume_deletion_vectors(row_index_column.size(), deletion_vectors, deletion_vector_row_counts);
  return compute_deleted_row_count(row_index_column, dv_refs, dv_row_counts, stream);
}

}  // namespace cudf::io::parquet::experimental
