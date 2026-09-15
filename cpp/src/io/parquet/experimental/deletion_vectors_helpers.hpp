/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/io/experimental/deletion_vectors.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/roaring_bitmap.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <functional>
#include <optional>
#include <queue>

namespace cudf::io::parquet::experimental {

/**
 * @brief Prepends the index column information to the table metadata
 *
 * @param[in,out] metadata Table metadata to update with the index column name
 */
void prepend_index_column_to_table_metadata(table_metadata& metadata);

/**
 * @brief Prepends the index column to the table columns
 *
 * @param table Input table to prepend the index column to
 * @param row_index_column Row index column to prepend
 * @return A new table with the index column prepended to the input table columns
 */
[[nodiscard]] std::unique_ptr<cudf::table> prepend_index_column_to_table(
  std::unique_ptr<cudf::table>&& table, std::unique_ptr<cudf::column>&& row_index_column);

/**
 * @brief Computes a row index column from the specified row group row offsets and counts
 *
 * @param row_group_offsets Host span of row offsets of each row group
 * @param row_group_num_rows Host span of number of rows in each row group
 * @param start_row Starting row index when row group offsets are empty
 * @param num_rows Total number of rows in the output column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to allocate the output column
 *
 * @return UINT64 column containing row indices
 */
[[nodiscard]] std::unique_ptr<cudf::column> compute_row_index_column(
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  std::optional<size_t> start_row,
  size_type num_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes a chunk of the row index column by consuming row group data from queues
 *
 * @param[in,out] row_group_offsets Queue of per-row-group starting row offsets
 * @param[in,out] row_group_num_rows Queue of per-row-group row counts
 * @param start_row Starting row index when row group data is unspecified
 * @param num_rows Number of rows in the output column
 * @param is_unspecified_row_group_data Whether to ignore queues and produce a simple sequence
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to allocate the output column
 *
 * @return UINT64 column containing row indices for this chunk
 */
[[nodiscard]] std::unique_ptr<cudf::column> compute_partial_row_index_column(
  std::queue<size_t>& row_group_offsets,
  std::queue<size_type>& row_group_num_rows,
  size_t start_row,
  size_type num_rows,
  bool is_unspecified_row_group_data,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes a BOOL8 row mask column from the specified row index column and deletion vectors
 *
 * @param row_index_column View of the row index column
 * @param deletion_vector_refs Host span of cudf::roaring_bitmap references
 * @param rows_per_deletion_vector Host span of number of rows per deletion vector
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to allocate the output column
 *
 * @return BOOL8 column where `true` indicates a non-deleted row
 */
[[nodiscard]] std::unique_ptr<cudf::column> compute_row_mask_column(
  cudf::column_view const& row_index_column,
  cudf::host_span<std::reference_wrapper<cudf::roaring_bitmap const> const> deletion_vector_refs,
  cudf::host_span<size_type const> rows_per_deletion_vector,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes the number of rows deleted by the deletion vectors
 *
 * @param row_index_column View of the row index column
 * @param deletion_vector_refs Host span of cudf::roaring_bitmap references
 * @param deletion_vector_row_counts Host span of number of rows per deletion vector
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Number of rows marked as deleted
 */
[[nodiscard]] size_t compute_deleted_row_count(
  cudf::column_view const& row_index_column,
  cudf::host_span<std::reference_wrapper<cudf::roaring_bitmap const> const> deletion_vector_refs,
  cudf::host_span<size_type const> deletion_vector_row_counts,
  cuda::stream_ref stream);

/**
 * @brief Computes a chunk of the BOOL8 row mask column by consuming deletion vectors from queues
 *
 * @param row_index_column View of the row index column for the current chunk
 * @param[in,out] deletion_vectors Queue of roaring bitmaps to consume from
 * @param[in,out] deletion_vector_row_counts Queue of per-deletion-vector row counts
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to allocate the output column
 *
 * @return BOOL8 column where `true` indicates a non-deleted row
 */
[[nodiscard]] std::unique_ptr<cudf::column> compute_partial_row_mask_column(
  cudf::column_view const& row_index_column,
  std::queue<cudf::roaring_bitmap>& deletion_vectors,
  std::queue<size_type>& deletion_vector_row_counts,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes the number of deleted rows for a chunk by consuming deletion vectors from queues
 *
 * @param row_index_column View of the row index column for the current chunk
 * @param[in,out] deletion_vectors Queue of roaring bitmaps to consume from
 * @param[in,out] deletion_vector_row_counts Queue of per-deletion-vector row counts
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Number of deleted rows in this chunk
 */
[[nodiscard]] size_t compute_partial_deleted_row_count(
  cudf::column_view const& row_index_column,
  std::queue<cudf::roaring_bitmap>& deletion_vectors,
  std::queue<size_type>& deletion_vector_row_counts,
  cuda::stream_ref stream);

}  // namespace cudf::io::parquet::experimental
