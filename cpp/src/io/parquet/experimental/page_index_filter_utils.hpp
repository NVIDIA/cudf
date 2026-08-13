/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/parquet/reader_impl_helpers.hpp"

#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <span>

namespace cudf::io::parquet::experimental::detail {

using metadata_base = parquet::detail::metadata;

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
  rmm::cuda_stream_view stream);

/**
 * @brief Computes page row offsets and the size (number of rows) of the largest page for a given
 * column schema index
 *
 * @param per_file_metadata Span of parquet footer metadata
 * @param row_group_indices Span of input row group indices
 * @param schema_idx Column's schema index
 * @return A pair of page row offsets and the size of the largest page in this
 * column
 */
[[nodiscard]] std::pair<std::vector<size_type>, size_type> compute_page_row_offsets(
  cudf::host_span<metadata_base const> per_file_metadata,
  std::span<std::vector<size_type> const> row_group_indices,
  size_type schema_idx);

/**
 * @brief Computes a device vector where each row contains the index of the page it belongs to
 *
 * @param page_row_offsets Span of page row offsets
 * @param total_rows Total number of rows
 * @param stream CUDA stream
 * @param mr Device memory resource for the output device vector
 * @return Device vector where each row contains the index of the page it belongs to
 */
[[nodiscard]] rmm::device_uvector<size_type> compute_page_indices_async(
  cudf::host_span<cudf::size_type const> page_row_offsets,
  cudf::size_type total_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes the offsets of the Fenwick tree levels (level 1 and higher) until the tree level
 * block size becomes larger than the maximum page (search range) size
 *
 * @param level0_size Size of the zeroth tree level (the row mask)
 * @param max_page_size Maximum page (search range) size
 * @return Fenwick tree level offsets
 */
[[nodiscard]] std::vector<size_type> compute_fenwick_tree_level_offsets(
  cudf::size_type level0_size, cudf::size_type max_page_size);

}  // namespace cudf::io::parquet::experimental::detail
