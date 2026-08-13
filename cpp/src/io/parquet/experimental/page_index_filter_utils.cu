/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "page_index_filter_utils.hpp"

#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/gather.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

std::pair<cudf::detail::host_vector<size_type>, cudf::detail::host_vector<size_type>>
compute_page_row_offsets_and_colchunk_page_offsets(
  std::span<metadata_base const> per_file_metadata,
  std::span<std::vector<size_type> const> row_group_indices,
  size_type schema_idx,
  rmm::cuda_stream_view stream)
{
  // Compute total number of row groups
  auto const total_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    std::size_t{0},
                    [](auto sum, auto const& rg_indices) { return sum + rg_indices.size(); });

  // Vector to store the cumulative number of rows in each page - - set initial capacity to two data
  // pages per row group
  auto page_row_offsets =
    cudf::detail::make_empty_host_vector<cudf::size_type>((2 * total_row_groups) + 1, stream);
  // Vector to store the cumulative number of pages in each column chunk
  auto col_chunk_page_offsets =
    cudf::detail::make_empty_host_vector<cudf::size_type>(total_row_groups + 1, stream);

  page_row_offsets.push_back(0);
  col_chunk_page_offsets.push_back(0);

  // For all data sources
  std::for_each(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{row_group_indices.size()},
    [&](auto src_idx) {
      // For all column chunks in this data source
      auto const& rg_indices = row_group_indices[src_idx];
      std::optional<size_type> colchunk_iter_offset{};
      std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
        auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
        colchunk_iter_offset =
          parquet::detail::find_colchunk_iter_offset(row_group, schema_idx, colchunk_iter_offset);
        auto const& colchunk_iter = row_group.columns.begin() + colchunk_iter_offset.value();

        CUDF_EXPECTS(colchunk_iter->offset_index.has_value(),
                     "Offset index not found for column chunk",
                     std::invalid_argument);

        auto const& offset_index       = colchunk_iter->offset_index.value();
        auto const row_group_num_pages = offset_index.page_locations.size();

        col_chunk_page_offsets.push_back(col_chunk_page_offsets.back() + row_group_num_pages);

        // For all pages in this column chunk, update page row offsets.
        std::for_each(
          cuda::counting_iterator<std::size_t>{0},
          cuda::counting_iterator{row_group_num_pages},
          [&](auto const page_idx) {
            int64_t const first_row_idx = offset_index.page_locations[page_idx].first_row_index;
            // For the last page, this is simply the total number of rows in the column chunk
            int64_t const last_row_idx =
              (page_idx < row_group_num_pages - 1)
                ? offset_index.page_locations[page_idx + 1].first_row_index
                : row_group.num_rows;

            // Update the page row offsets.
            page_row_offsets.push_back(page_row_offsets.back() + last_row_idx - first_row_idx);
          });
      });
    });

  return {std::move(page_row_offsets), std::move(col_chunk_page_offsets)};
}

std::pair<std::vector<size_type>, size_type> compute_page_row_offsets(
  cudf::host_span<metadata_base const> per_file_metadata,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::size_type schema_idx)
{
  // Compute total number of row groups
  auto const total_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    std::size_t{0},
                    [](auto sum, auto const& rg_indices) { return sum + rg_indices.size(); });

  std::vector<size_type> page_row_offsets;
  page_row_offsets.push_back(0);
  size_type max_page_size = 0;

  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{row_group_indices.size()},
                [&](auto const src_idx) {
                  // For all row groups in this source
                  auto const& rg_indices = row_group_indices[src_idx];
                  std::optional<size_type> colchunk_iter_offset{};
                  std::for_each(rg_indices.begin(), rg_indices.end(), [&](auto const& rg_idx) {
                    auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
                    colchunk_iter_offset  = parquet::detail::find_colchunk_iter_offset(
                      row_group, schema_idx, colchunk_iter_offset);
                    auto const& colchunk_iter =
                      row_group.columns.begin() + colchunk_iter_offset.value();
                    CUDF_EXPECTS(colchunk_iter->offset_index.has_value(),
                                 "Offset index not found for column chunk",
                                 std::invalid_argument);
                    auto const& offset_index       = colchunk_iter->offset_index.value();
                    auto const row_group_num_pages = offset_index.page_locations.size();
                    std::for_each(cuda::counting_iterator<std::size_t>{0},
                                  cuda::counting_iterator{row_group_num_pages},
                                  [&](auto const page_idx) {
                                    int64_t const first_row_idx =
                                      offset_index.page_locations[page_idx].first_row_index;
                                    int64_t const last_row_idx =
                                      (page_idx < row_group_num_pages - 1)
                                        ? offset_index.page_locations[page_idx + 1].first_row_index
                                        : row_group.num_rows;
                                    auto const page_size = last_row_idx - first_row_idx;
                                    max_page_size = std::max<size_type>(max_page_size, page_size);
                                    page_row_offsets.push_back(page_row_offsets.back() + page_size);
                                  });
                  });
                });

  return {std::move(page_row_offsets), max_page_size};
}

rmm::device_uvector<size_type> compute_page_indices_async(
  cudf::host_span<cudf::size_type const> page_row_offsets,
  cudf::size_type total_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto row_offsets = cudf::detail::make_device_uvector_async(
    page_row_offsets, stream, cudf::get_current_device_resource_ref());

  auto page_indices = rmm::device_uvector<cudf::size_type>(total_rows, stream, mr);
  cudf::detail::label_segments(
    row_offsets.begin(), row_offsets.end(), page_indices.begin(), page_indices.end(), stream);
  return page_indices;
}

std::vector<size_type> compute_fenwick_tree_level_offsets(cudf::size_type level0_size,
                                                          cudf::size_type max_page_size)
{
  std::vector<size_type> tree_level_offsets;
  tree_level_offsets.push_back(0);

  cudf::size_type current_level_size = cudf::util::div_rounding_up_safe(level0_size, 2);
  cudf::size_type current_level      = 1;

  while (current_level_size > 0) {
    auto const block_size = 1 << current_level;
    if (std::cmp_greater(block_size, max_page_size)) { break; }
    tree_level_offsets.push_back(tree_level_offsets.back() + current_level_size);
    current_level_size =
      current_level_size == 1 ? 0 : cudf::util::div_rounding_up_safe(current_level_size, 2);
    current_level++;
  }
  return tree_level_offsets;
}

}  // namespace cudf::io::parquet::experimental::detail
