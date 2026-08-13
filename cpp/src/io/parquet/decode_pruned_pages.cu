/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/std/algorithm>

namespace cudf::io::parquet::detail {

namespace {

auto constexpr block_size = 4 * cudf::detail::warp_size;

/**
 * @brief Initialize output entries for pruned string and list pages.
 *
 * String entries receive either the page's initial offset for small strings or a zero size for
 * large strings. List offsets receive the start value of their child nesting level. These targeted
 * writes avoid zero-initializing entire output buffers.
 */
CUDF_KERNEL void __launch_bounds__(block_size)
  fill_pruned_offsets_kernel(device_span<PageInfo> pages,
                             device_span<ColumnChunkDesc const> chunks,
                             device_span<bool const> page_mask,
                             device_span<size_t> initial_str_offsets,
                             size_t skip_rows,
                             size_t num_rows)
{
  namespace cg = cooperative_groups;

  auto const block    = cg::this_thread_block();
  auto const page_idx = cg::this_grid().block_rank();
  auto const t        = static_cast<size_type>(block.thread_rank());
  if (page_mask[page_idx]) { return; }

  auto const& page  = pages[page_idx];
  auto const& chunk = chunks[page.chunk_idx];

  if (chunk.column_data_base == nullptr) { return; }

  auto const is_list_col = chunk.max_level[level_type::REPETITION] != 0;

  // Write offsets for pruned non-list (flat) string columns.
  if (not is_list_col and is_string_col(chunk)) {
    auto data = static_cast<size_type*>(chunk.column_data_base[chunk.max_nesting_depth - 1]);
    if (data == nullptr) { return; }

    auto const page_begin = chunk.start_row + page.chunk_row;
    auto const page_end   = page_begin + page.num_rows;
    auto const read_end   = skip_rows + num_rows;
    auto const begin      = cuda::std::max(page_begin, skip_rows);
    auto const end        = cuda::std::min(page_end, read_end);
    if (begin >= end) { return; }

    // Large strings needs the first page string offset, including if the page was
    // pruned. Record it here.
    if (chunk.is_large_string_col and t == 0) {
      auto const chunks_per_rowgroup = initial_str_offsets.size();
      auto const input_col_idx       = page.chunk_idx % chunks_per_rowgroup;
      cuda::atomic_ref<size_t, cuda::std::thread_scope_device> initial_str_offset{
        initial_str_offsets[input_col_idx]};
      initial_str_offset.fetch_min(page.str_offset, cuda::std::memory_order_relaxed);
    }

    // Write zeros for large strings and the page's initial offset otherwise.
    auto const value =
      chunk.is_large_string_col ? size_type{0} : static_cast<size_type>(page.str_offset);
    for (auto row = begin + t; row < end; row += block.size()) {
      data[row - skip_rows] = value;
    }
    return;
  }

  // Write offsets to list locations at each depth.
  if (is_list_col and page.nesting != nullptr and page.nesting_decode != nullptr) {
    for (auto depth = 0; depth < chunk.max_nesting_depth - 1; depth++) {
      auto offsets       = static_cast<size_type*>(chunk.column_data_base[depth]);
      auto& nesting_info = page.nesting[depth];
      // Pruned list pages retain rows through the first list level but contribute no child values.
      // The preprocessing pass computes the output range and child start value at every list depth.
      if (nesting_info.type != type_id::LIST or offsets == nullptr) { continue; }
      // Emit an offset for the current nesting level equal to current length of the next nesting
      // level
      auto const output_begin = page.nesting_decode[depth].page_start_value;
      auto const offset       = page.nesting_decode[depth + 1].page_start_value;
      for (auto offset_idx = t; offset_idx < nesting_info.batch_size;
           offset_idx += static_cast<size_type>(block.size())) {
        offsets[output_begin + offset_idx] = offset;
      }
    }
  }
}

}  // namespace

void fill_pruned_offsets(cudf::device_span<PageInfo> pages,
                         cudf::device_span<ColumnChunkDesc const> chunks,
                         cudf::device_span<bool const> page_mask,
                         cudf::device_span<size_t> initial_str_offsets,
                         size_t skip_rows,
                         size_t num_rows,
                         rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() == page_mask.size(), "Page mask size does not match page count");
  fill_pruned_offsets_kernel<<<pages.size(), block_size, 0, stream.value()>>>(
    pages, chunks, page_mask, initial_str_offsets, skip_rows, num_rows);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::parquet::detail
