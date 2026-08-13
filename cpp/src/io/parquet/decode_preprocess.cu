/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delta_binary.cuh"
#include "io/utilities/column_buffer.hpp"
#include "page_decode.cuh"
#include "page_state_composed.cuh"
#include "reader_impl_chunking_utils.cuh"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/hashing/detail/default_hash.cuh>

#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/std/iterator>
#include <cuda/std/limits>

namespace cudf::io::parquet::detail {

namespace cg = cooperative_groups;

namespace {

// # of threads we're decoding with
constexpr int preprocess_block_size   = 512;
constexpr int level_decode_block_size = 128;

using unused_state_buf = page_state_buffers_s<0, 0, 0>;

/**
 * @brief Update output column sizes for every nesting level based on a batch
 * of incoming decoded definition and repetition level values.
 *
 * If bounds_set is true, computes skipped_values and skipped_leaf_values for the
 * page to indicate where we need to skip to based on min/max row.
 *
 * Operates at thread block level.
 *
 * @param s The local page info
 * @param target_value_count The target value count to process up to
 * @param rep Repetition level buffer
 * @param def Definition level buffer
 * @param bounds_set Boolean indicating whether min/max row bounds have been set
 * @param block The cooperative thread block
 */
template <typename level_t>
__device__ void update_page_sizes(auto* s,
                                  int target_value_count,
                                  level_t const* const rep,
                                  level_t const* const def,
                                  bool bounds_set,
                                  cg::thread_block const& block)
{
  // max nesting depth of the column
  int const max_depth          = s->setup.col.max_nesting_depth;
  int const t                  = block.thread_rank();
  constexpr int num_warps      = preprocess_block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  using block_reduce = cub::BlockReduce<int, preprocess_block_size>;
  using block_scan   = cub::BlockScan<int, preprocess_block_size>;
  __shared__ union {
    typename block_reduce::TempStorage reduce_storage;
    typename block_scan::TempStorage scan_storage;
  } temp_storage;

  // how many input level values we've processed in the page so far
  int value_count = s->progress.input_value_count;
  // how many rows we've processed in the page so far
  int row_count = s->progress.input_row_count;
  // how many leaf values we've processed in the page so far
  int leaf_count = s->progress.input_leaf_count;
  // whether or not we need to continue checking for the first row
  bool skipped_values_set = s->setup.page.skipped_values >= 0;

  while (value_count < target_value_count) {
    int const batch_size =
      cuda::std::min<int32_t>(max_batch_size, target_value_count - value_count);

    // start/end depth
    int start_depth, end_depth, d;
    get_nesting_bounds<level_t>(
      start_depth, end_depth, d, s, rep, def, value_count, value_count + batch_size, t);

    // is this thread within row bounds? in the non skip_rows/num_rows case this will always
    // be true.
    int in_row_bounds = 1;

    // if we are in the skip_rows/num_rows case, we need to check against these limits
    if (bounds_set) {
      // get absolute thread row index
      int const is_new_row = start_depth == 0;
      int thread_row_count, block_row_count;
      block_scan(temp_storage.scan_storage)
        .InclusiveSum(is_new_row, thread_row_count, block_row_count);
      block.sync();

      // get absolute thread leaf index
      int const is_new_leaf = (d >= s->nesting.nesting_info[max_depth - 1].max_def_level);
      int thread_leaf_count, block_leaf_count;
      block_scan(temp_storage.scan_storage)
        .InclusiveSum(is_new_leaf, thread_leaf_count, block_leaf_count);
      block.sync();

      // if this thread is in row bounds
      int const row_index = (thread_row_count + row_count) - 1;
      in_row_bounds       = (row_index >= s->progress.row_index_lower_bound) &&
                      (row_index < (s->setup.first_row + s->setup.num_rows));

      // if we have not set skipped values yet, see if we found the first in-bounds row
      if (!skipped_values_set) {
        int local_count, global_count;
        block_scan(temp_storage.scan_storage)
          .InclusiveSum(in_row_bounds, local_count, global_count);
        block.sync();

        // we found it
        if (global_count > 0) {
          // this is the thread that represents the first row.
          if (local_count == 1 && in_row_bounds) {
            s->setup.page.skipped_values = value_count + t;
            s->setup.page.skipped_leaf_values =
              leaf_count + (is_new_leaf ? thread_leaf_count - 1 : thread_leaf_count);
          }
          skipped_values_set = true;
        }
      }

      row_count += block_row_count;
      leaf_count += block_leaf_count;
    }

    // increment value counts across all nesting depths
    for (int s_idx = 0; s_idx < max_depth; s_idx++) {
      int const in_nesting_bounds = (s_idx >= start_depth && s_idx <= end_depth && in_row_bounds);
      int const count = block_reduce(temp_storage.reduce_storage).Sum(in_nesting_bounds);
      block.sync();
      if (!t) {
        PageNestingInfo* pni = &s->setup.page.nesting[s_idx];
        pni->batch_size += count;
      }
    }

    value_count += batch_size;
  }

  // update final outputs
  if (!t) {
    s->progress.input_value_count = value_count;

    // only used in the skip_rows/num_rows case
    s->progress.input_leaf_count = leaf_count;
    s->progress.input_row_count  = row_count;
  }

  block.sync();
}

/**
 * @brief Updates size information for a pruned page across all nesting levels
 *
 * @param[in,out] page The page to compute sizes for
 * @param[in] state The local page info
 * @param[in] has_repetition Whether the page has repetition
 * @param[in] is_base_pass Whether this is the base pass
 * @param[in] block The current thread block cooperative group
 */
__device__ void compute_page_sizes_for_pruned_pages(PageInfo* page,
                                                    auto* const state,
                                                    bool has_repetition,
                                                    bool is_base_pass,
                                                    cg::thread_block const& block)
{
  auto const max_depth = page->num_output_nesting_levels;
  // Return early if no repetition and max depth is 1
  if (not has_repetition and max_depth == 1) {
    if (!block.thread_rank()) {
      if (is_base_pass) { page->nesting[0].size = page->num_rows; }
      page->nesting[0].batch_size = state->setup.num_rows;
    }
    return;
  }

  // Use warp 0 to set nesting size information for all depths
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(block);
  if (warp.meta_group_rank() == 0) {
    auto list_depth = 0;
    // Find the depth of the first list
    if (has_repetition) {
      auto depth = 0;
      while (depth < max_depth) {
        auto const thread_depth = depth + warp.thread_rank();
        auto const is_list =
          thread_depth < max_depth and page->nesting[thread_depth].type == type_id::LIST;
        uint32_t const list_mask = warp.ballot(is_list);
        if (list_mask != 0) {
          auto const first_list_lane = cuda::std::countr_zero(list_mask);
          list_depth                 = warp.shfl(thread_depth, first_list_lane);
          break;
        }
        depth += warp.size();
      }
      // Zero out size information for all depths beyond the first list depth
      for (auto depth = list_depth + 1 + warp.thread_rank(); depth < max_depth;
           depth += warp.size()) {
        if (is_base_pass) { page->nesting[depth].size = 0; }
        page->nesting[depth].batch_size = 0;
      }
    }
    // Write size information for all depths up to the list depth
    for (auto depth = warp.thread_rank(); depth < list_depth; depth += warp.size()) {
      if (is_base_pass) { page->nesting[depth].size = page->num_rows; }
      page->nesting[depth].batch_size = state->setup.num_rows;
    }
    // Write size information at the list depth (zero if no list)
    if (warp.thread_rank() == 0) {
      if (is_base_pass) { page->nesting[list_depth].size = page->num_rows; }
      page->nesting[list_depth].batch_size = state->setup.num_rows;
    }
  }
}

/**
 * @brief Kernel for computing per-page column size information for all nesting levels.
 *
 * This function will write out the size field for each level of nesting.
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read. Pass as INT_MAX to guarantee reading all rows
 * @param is_base_pass Whether or not this is the base pass.  We first have to compute
 * the full size information of every page before we come through in a second (trim) pass
 * to determine what subset of rows in this page we should be reading
 * @param compute_string_sizes Whether or not we should be computing string sizes
 * (PageInfo::str_bytes) as part of the pass
 */
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(preprocess_block_size)
  compute_page_sizes_kernel(PageInfo* pages,
                            device_span<ColumnChunkDesc const> chunks,
                            device_span<bool const> page_mask,
                            size_t min_row,
                            size_t num_rows,
                            bool is_base_pass)
{
  __shared__ __align__(16) full_page_decode_state state_g;

  auto* const s      = &state_g;
  auto const block   = cg::this_thread_block();
  int const page_idx = cg::this_grid().block_rank();
  int const t        = block.thread_rank();
  PageInfo* pp       = &pages[page_idx];

  // whether or not we have repetition levels (lists)
  bool has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // setup page info
  if (!setup_local_page_info(
        s, pp, chunks, min_row, num_rows, all_types_filter{}, page_processing_stage::PREPROCESS)) {
    return;
  }

  // Return early if this page is pruned
  if (not page_mask.empty() and not page_mask[page_idx]) {
    return compute_page_sizes_for_pruned_pages(pp, s, has_repetition, is_base_pass, block);
  }

  // - if this is a flat hierarchy (no lists), we don't need
  // to do the expensive work of traversing the level data to determine sizes.  we can just compute
  // it directly.
  if (!has_repetition) {
    int depth = 0;
    while (depth < s->setup.page.num_output_nesting_levels) {
      auto const thread_depth = depth + t;
      if (thread_depth < s->setup.page.num_output_nesting_levels) {
        if (is_base_pass) { pp->nesting[thread_depth].size = pp->num_input_values; }
        pp->nesting[thread_depth].batch_size = pp->num_input_values;
      }
      depth += block.size();
    }
    return;
  }

  // in the trim pass, for anything with lists, we only need to fully process bounding pages (those
  // at the beginning or the end of the row bounds)
  if (!is_base_pass &&
      !is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition)) {
    int depth = 0;
    while (depth < s->setup.page.num_output_nesting_levels) {
      auto const thread_depth = depth + t;
      if (thread_depth < s->setup.page.num_output_nesting_levels) {
        // if we are not a bounding page (as checked above) then we are either
        // returning all rows/values from this page, or 0 of them
        pp->nesting[thread_depth].batch_size =
          (s->setup.num_rows == 0 &&
           !is_page_contained(s->setup.page, s->setup.col.start_row, min_row, num_rows))
            ? 0
            : pp->nesting[thread_depth].size;
      }
      depth += block.size();
    }
    return;
  }

  // zero sizes
  int depth = 0;
  while (depth < s->setup.page.num_output_nesting_levels) {
    auto const thread_depth = depth + t;
    if (thread_depth < s->setup.page.num_output_nesting_levels) {
      s->setup.page.nesting[thread_depth].batch_size = 0;
    }
    depth += blockDim.x;
  }

  auto* const rep          = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);
  bool const process_nulls = should_process_nulls(s);
  level_t* const def       = !process_nulls
                               ? nullptr
                               : reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);

  if (!t) {
    s->setup.page.skipped_values      = -1;
    s->setup.page.skipped_leaf_values = 0;
    s->progress.input_row_count       = 0;
    s->progress.input_value_count     = 0;

    // in the base pass, we're computing the number of rows, make sure we visit absolutely
    // everything
    if (is_base_pass) {
      s->setup.first_row                = 0;
      s->setup.num_rows                 = cuda::std::numeric_limits<int32_t>::max();
      s->progress.row_index_lower_bound = -1;
    }
  }

  block.sync();

  // update_page_sizes
  update_page_sizes<level_t>(s, s->setup.page.num_input_values, rep, def, !is_base_pass, block);

  // update output results:
  // - real number of rows for the whole page
  // - nesting sizes for the whole page
  // - skipped value information for trimmed pages
  // - string bytes
  if (is_base_pass) {
    // nesting level 0 is the root column, so the size is also the # of rows
    if (!t) { pp->num_rows = s->setup.page.nesting[0].batch_size; }

    // store off this batch size as the "full" size
    int depth = 0;
    while (depth < s->setup.page.num_output_nesting_levels) {
      auto const thread_depth = depth + t;
      if (thread_depth < s->setup.page.num_output_nesting_levels) {
        pp->nesting[thread_depth].size = pp->nesting[thread_depth].batch_size;
      }
      depth += block.size();
    }
  }

  if (!t) {
    pp->skipped_values      = s->setup.page.skipped_values;
    pp->skipped_leaf_values = s->setup.page.skipped_leaf_values;
  }
}

/**
 * @brief Kernel for preprocessing definition and repetition levels
 *
 * This kernel decodes definition and repetition levels for all pages in advance,
 * storing them in the pre-allocated level decode buffers. This allows other
 * kernels to skip RLE decoding and directly access the decoded levels.
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param page_mask Boolean vector indicating which pages need to be processed
 * @param min_row Minimum row index to read
 * @param num_rows Number of rows to read starting from min_row
 */
#pragma nv_diag_suppress static_var_with_dynamic_init
template <typename level_t, int level_decode_block_size>
CUDF_KERNEL void __launch_bounds__(level_decode_block_size)
  preprocess_levels_kernel(PageInfo* pages,
                           device_span<ColumnChunkDesc const> chunks,
                           cudf::device_span<bool const> page_mask,
                           size_t min_row,
                           size_t num_rows)
{
  __shared__ __align__(16) level_scan_state state_g;

  auto* const s      = &state_g;
  auto const block   = cg::this_thread_block();
  int const page_idx = cg::this_grid().block_rank();
  int const t        = block.thread_rank();
  PageInfo* pp       = &pages[page_idx];

  // Return early if this page is pruned
  if (not page_mask.empty() and not page_mask[page_idx]) { return; }

  // setup page info - use all_types_filter since we need to preprocess levels for all page types
  if (!setup_local_page_info(
        s, pp, chunks, min_row, num_rows, all_types_filter{}, page_processing_stage::PREPROCESS)) {
    return;
  }

  // whether or not we have repetition levels (lists)
  bool const has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // the level stream decoders. max_output_values is max to remove rolling buffer
  // logic from the decode step. The chunked-expand rle_stream does not need a
  // shared-memory ring buffer of run headers; it parses runs directly into
  // per-chunk tables, so we default-construct the decoders here.
  static constexpr int max_output_values = cuda::std::numeric_limits<int>::max();
  using decoder_stream_t = rle_stream_chunked<level_t, level_decode_block_size, max_output_values>;
  decoder_stream_t decoders[level_type::NUM_LEVEL_TYPES] = {};

  // Shared-memory staging scratch for the encoded level streams. Level streams
  // for a page are usually small (definition/repetition levels are dominated by
  // short RLE runs), and their serial run-header parse is latency-bound on
  // dependent global loads. Staging the bytes into shared memory once removes
  // that latency from fill_run_batch(). Streams larger than the per-stream
  // budget fall back to parsing from global with no behavior change.
  __shared__ __align__(16) uint8_t stage[decoder_stream_t::smem_stage_size];
  __shared__ cuda::barrier<cuda::thread_scope_block> copy_barrier;

  // Get the level decode buffers for this page
  auto* const def = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto* const rep = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  // Determine how many values need to be decoded
  size_t const num_to_decode =
    precompute_page_num_values_in_range(*pp, chunks[pp->chunk_idx], min_row, num_rows);
  if (num_to_decode == 0) { return; }

  // Initialize the stream decoders
  bool const process_nulls = should_process_nulls(s);
  if (has_repetition) {
    cg::invoke_one(block, [&]() { init(&copy_barrier, block.size()); });
    block.sync();
    decoders[level_type::REPETITION].init(block,
                                          s->setup.col.level_bits[level_type::REPETITION],
                                          s->stream.abs_lvl_start[level_type::REPETITION],
                                          s->stream.abs_lvl_end[level_type::REPETITION],
                                          rep,
                                          num_to_decode,
                                          stage,
                                          &copy_barrier,
                                          decoder_stream_t::smem_stage_size);
    copy_barrier.arrive_and_wait();
    decoders[level_type::REPETITION].decode_next(t, num_to_decode);
  }

  // Decode levels for this page up to the last row needed.
  // If skipping the first rows, we still need to decode their levels.
  // This is because we need to determine the number of non-null values we skipped.
  // Note that for lists we haven't computed skipped_leaf_values yet; this is used as input for
  // that.
  // Must sync as shared variables in decode_next() are shared between decoders!!
  block.sync();

  if (process_nulls) {
    cg::invoke_one(block, [&]() { init(&copy_barrier, block.size()); });
    block.sync();
    decoders[level_type::DEFINITION].init(block,
                                          s->setup.col.level_bits[level_type::DEFINITION],
                                          s->stream.abs_lvl_start[level_type::DEFINITION],
                                          s->stream.abs_lvl_end[level_type::DEFINITION],
                                          def,
                                          num_to_decode,
                                          stage,
                                          &copy_barrier,
                                          decoder_stream_t::smem_stage_size);
    copy_barrier.arrive_and_wait();
    decoders[level_type::DEFINITION].decode_next(t, num_to_decode);
  }
}

}  // anonymous namespace

/**
 * @copydoc cudf::io::parquet::gpu::compute_page_sizes
 */
void compute_page_sizes(cudf::detail::hostdevice_span<PageInfo> pages,
                        cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                        cudf::device_span<bool const> page_mask,
                        size_t min_row,
                        size_t num_rows,
                        bool compute_num_rows,
                        int level_type_size,
                        rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  if (pages.size() == 0) { return; }

  dim3 dim_block(preprocess_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  // computes:
  // PageNestingInfo::size for each level of nesting, for each page.
  // This computes the size for the entire page, not taking row bounds into account.
  // If uses_custom_row_bounds is set to true, we have to do a second pass later that "trims"
  // the starting and ending read values to account for these bounds.
  if (level_type_size == 1) {
    compute_page_sizes_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, page_mask, min_row, num_rows, compute_num_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    compute_page_sizes_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, page_mask, min_row, num_rows, compute_num_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

/**
 * @copydoc cudf::io::parquet::detail::preprocess_levels
 */
void preprocess_levels(cudf::detail::hostdevice_span<PageInfo> pages,
                       cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                       cudf::device_span<bool const> page_mask,
                       size_t min_row,
                       size_t num_rows,
                       int level_type_size,
                       rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  if (pages.size() == 0) { return; }

  dim3 dim_block(level_decode_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    preprocess_levels_kernel<uint8_t, level_decode_block_size>
      <<<dim_grid, dim_block, 0, stream.value()>>>(
        pages.device_ptr(), chunks, page_mask, min_row, num_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    preprocess_levels_kernel<uint16_t, level_decode_block_size>
      <<<dim_grid, dim_block, 0, stream.value()>>>(
        pages.device_ptr(), chunks, page_mask, min_row, num_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

}  // namespace cudf::io::parquet::detail
