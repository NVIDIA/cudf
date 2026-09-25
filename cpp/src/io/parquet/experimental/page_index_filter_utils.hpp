/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/parquet/reader_impl_helpers.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

using metadata_base = parquet::detail::metadata;

/**
 * @brief Compact page statistics for a logical filter column.
 */
struct page_statistics_input {
  size_type column_index;  ///< Index of the logical filter column
  cudf::table statistics;  ///< Compacted page-level min, max, and all-null stats columns
  std::vector<size_type> page_row_offsets;  ///< Row offset of each page, spanning all rows
};

/**
 * @brief Compute page row offsets and column chunk page (count) offsets for a given column schema
 * index
 *
 * @param per_file_metadata Span of parquet footer metadata
 * @param row_group_indices Span of input row group indices
 * @param schema_idx Column's schema index
 * @param stream CUDA stream
 * @return Pair of page row offsets and column chunk page (count) offsets
 */
[[nodiscard]] std::pair<cudf::detail::host_vector<size_type>, cudf::detail::host_vector<size_type>>
compute_page_row_offsets_and_colchunk_page_offsets(
  std::span<metadata_base const> per_file_metadata,
  std::span<std::vector<size_type> const> row_group_indices,
  size_type schema_idx,
  cuda::stream_ref stream);

/**
 * @brief Computes page row offsets and the size (number of rows) of the largest page for a given
 * column schema index
 *
 * @param per_file_metadata Span of parquet footer metadata
 * @param row_group_indices Span of input row group indices
 * @param schema_idx Column's schema index
 * @return A pair of page row offsets and the size of the largest page in this column
 */
[[nodiscard]] std::pair<std::vector<size_type>, size_type> compute_page_row_offsets(
  std::span<metadata_base const> per_file_metadata,
  std::span<std::vector<size_type> const> row_group_indices,
  size_type schema_idx);

/**
 * @brief Checks whether every row is retained by the boolean row mask
 *
 * Null entries in the row mask are treated as surviving rows
 *
 * @param retention_mask Boolean column indicating retained rows
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Boolean indicating whether every row is retained
 */
[[nodiscard]] bool are_all_rows_retained(cudf::column_view const& retention_mask,
                                         cuda::stream_ref stream);

/**
 * @brief Computes a mask indicating which row ranges contain at least one selected row
 *
 * Null entries in the row mask are treated as surviving rows
 *
 * @param row_mask Boolean column indicating selected rows
 * @param page_row_offsets Page row offsets defining the row ranges
 * @param max_page_size Size of the largest page row range
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Boolean vector with one entry for each consecutive row range
 */
[[nodiscard]] thrust::host_vector<bool> compute_row_range_selection_mask(
  cudf::column_view const& row_mask,
  std::span<cudf::size_type const> page_row_offsets,
  cudf::size_type max_page_size,
  cuda::stream_ref stream);

/**
 * @brief Builds a row mask by evaluating page statistics on compact common page segments and
 * expanding them to rows
 *
 * @param inputs Span of page statistics inputs
 * @param num_columns Number of columns
 * @param total_rows Total number of rows
 * @param stats_expression AST expression to evaluate on page statistics
 * @param stream CUDA stream
 * @param mr Device memory resource for the output device vector
 * @return Row mask column
 */
[[nodiscard]] std::unique_ptr<column> compute_row_mask_from_page_stats(
  std::span<page_statistics_input const> inputs,
  size_type num_columns,
  size_type total_rows,
  std::reference_wrapper<ast::expression const> stats_expression,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::io::parquet::experimental::detail
