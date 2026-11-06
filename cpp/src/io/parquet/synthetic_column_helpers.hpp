/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reader_impl_chunking.hpp"
#include "reader_impl_helpers.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <cstddef>
#include <memory>
#include <span>

namespace cudf::io::parquet::detail {

/**
 * @brief Synthesizes a source-index column.
 *
 * @param num_rows_per_source Number of rows per source
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for output and temporary device allocations
 * @return Synthesized source-index column
 */
[[nodiscard]] std::unique_ptr<cudf::column> synthesize_source_index_column(
  std::span<std::size_t const> num_rows_per_source,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

/**
 * @brief Synthesizes row-group indices from a sorted source-index column.
 *
 * @param source_indices Source-index column containing one row per row group
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for output and temporary device allocations
 * @return Synthesized row-group index column
 */
[[nodiscard]] std::unique_ptr<cudf::column> synthesize_row_group_index_column(
  cudf::column_view const& source_indices, cuda::stream_ref stream, cudf::memory_resources mr);

/**
 * @brief Synthesizes a row-index column for the selected row groups.
 *
 * @param row_groups Selected row groups that map global rows to source-local rows
 * @param read_info Row range of the output chunk relative to the first selected row group
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for output and temporary device allocations
 * @return Synthesized row-index column
 */
[[nodiscard]] std::unique_ptr<cudf::column> synthesize_row_index_column(
  std::span<row_group_info const> row_groups,
  row_range const& read_info,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

}  // namespace cudf::io::parquet::detail
