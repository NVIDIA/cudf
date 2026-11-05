/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <reductions/jit/reduce.cuh>

namespace cudf::jit {

/**
 * @brief Executes one phase of a user-defined reduction over independent segments.
 *
 * Segment `s` covers input rows `[offsets[s], offsets[s + 1])`. Blocks visit logical work tiles
 * in a grid-stride loop, so the grid may contain fewer blocks than tiles:
 * - `REDUCE` assigns `blocks_per_segment` tiles to each segment and stores partial states at
 *   `results[s * blocks_per_segment + rank]`.
 * - `MERGE` assigns one tile per segment, merges its `blocks_per_segment` consecutive states
 *   from `partials`, and stores the combined state at `results[s]`.
 * - `FINALIZE` finalizes `partials[s]` using the tile's thread zero; it does not initialize or
 *   merge states, or use `results` and `scratch`.
 *
 * Every callback takes a leading segment ID. `update` additionally takes an input row index,
 * relative to the input views rather than the start of the segment. Empty segments and segments
 * with no selected rows are reduced from the identity state. Callback, null-handling, and error
 * contracts otherwise follow `reduce_kernel`.
 *
 * Finalization writes row zero of `outputs[s * Outputs::size + output]`. Each view must have
 * offset zero and independent single-row data and validity storage; the first mask word is
 * overwritten when present. `segment_errors[s]` receives the greatest state/finalizer error code.
 *
 * With `finalize_inline`, thread zero finalizes after `MERGE`, or after `REDUCE` when
 * `blocks_per_segment == 1`. Dependent phases must execute in stream order; this function
 * performs no grid-wide synchronization.
 *
 * @pre Launch one-dimensional blocks of exactly `BlockSize` threads, with all threads calling
 * this function. `BlockSize` must be a power of two between 32 and 1024.
 * @pre `blocks_per_segment` is positive and offsets describe nonnegative, nondecreasing ranges
 * within the inputs. `State` supports bytewise warp shuffles when `UseWarpReduce` is true.
 *
 * @tparam NullAware Whether callbacks receive nullable input/output values
 * @tparam Inputs Type list of input accessors in update-parameter order
 * @tparam Outputs Type list of output accessors in finalization-parameter order
 * @tparam BlockSize Number of threads per block
 * @tparam UseWarpReduce Use warp shuffles and shared warp leaders; otherwise use global scratch
 * @tparam State State containing `values`, `error`, and the field type list `types`
 * @tparam Filter Callable selecting rows before the update callback
 * @tparam Init Callable `errc(segment, state pointers...)` that initializes every state field
 * @tparam Update Callable `errc(segment, row, state pointers..., input values...)`
 * @tparam Merge Associative callable `errc(segment, destination pointers..., source values...)`
 * preserving the identity
 * @tparam Finalize Callable `errc(segment, output pointers..., state values...)`
 *
 * @param phase Reduction phase to execute
 * @param num_segments Nonnegative number of segments
 * @param offsets Device array of `num_segments + 1` boundaries; read only in `REDUCE`
 * @param inputs Device input views consumed by the input accessors in `REDUCE`
 * @param outputs Device array of `num_segments * Outputs::size` single-row views, in segment order
 * @param blocks_per_segment Number of partial states produced or consumed per segment
 * @param partials Source array of `num_segments * blocks_per_segment` states in `MERGE`, or
 * `num_segments` final states in `FINALIZE`; unused in `REDUCE`
 * @param results Writable array of `num_segments * blocks_per_segment` states in `REDUCE`, or
 * `num_segments` states in `MERGE`; unused in `FINALIZE`
 * @param scratch Writable array of `gridDim.x * BlockSize` states when `UseWarpReduce` is false;
 * reused by each physical block across tiles and unused in `FINALIZE`
 * @param segment_errors Device array of `num_segments` error slots, written only on finalization
 * @param finalize_inline Whether to finalize when a phase produces one reduced state per segment
 * @param init State initialization callback
 * @param filter Row filter callback, called only during input reduction
 * @param update Row update callback
 * @param merge State merge callback
 * @param finalize Output finalization callback
 * @see reduce_kernel
 */
template <bool NullAware,
          typename Inputs,
          typename Outputs,
          int BlockSize,
          bool UseWarpReduce,
          typename State,
          typename Init,
          typename Filter,
          typename Update,
          typename Merge,
          typename Finalize>
  requires(BlockSize >= detail::warp_size && BlockSize % detail::warp_size == 0 &&
           BlockSize <= 1024 && (BlockSize & (BlockSize - 1)) == 0)
__device__ void segmented_reduce_kernel(reduce::phase phase,
                                        size_type num_segments,
                                        size_type const* __restrict__ offsets,
                                        column_device_view_core const* __restrict__ inputs,
                                        mutable_column_device_view_core const* __restrict__ outputs,
                                        size_type blocks_per_segment,
                                        State const* __restrict__ partials,
                                        State* __restrict__ results,
                                        State* __restrict__ scratch,
                                        errc* __restrict__ segment_errors,
                                        bool finalize_inline,
                                        Init&& init,
                                        Filter&& filter,
                                        Update&& update,
                                        Merge&& merge,
                                        Finalize&& finalize)
{
  auto const tiles_per_segment = phase == reduce::phase::REDUCE ? blocks_per_segment : size_type{1};
  auto const num_tiles         = static_cast<thread_index_type>(num_segments) * tiles_per_segment;

  for (auto tile = static_cast<thread_index_type>(blockIdx.x); tile < num_tiles;
       tile += gridDim.x) {
    auto const segment  = tile / tiles_per_segment;
    auto const rank     = tile % tiles_per_segment;
    auto* segment_error = segment_errors + segment;
    auto do_merge       = [&](auto const& a, auto const& b) {
      return reduce::merge(a, b, merge, segment);
    };

    if (phase == reduce::phase::FINALIZE) {
      if (threadIdx.x == 0) {
        reduce::finalize<NullAware, Outputs>(
          partials[segment], outputs + segment * Outputs::size, segment_error, finalize, segment);
      }
      continue;
    }

    State local;

    local.error = [&]<int... I>(cuda::std::integer_sequence<int, I...>) {
      return init(segment, &cuda::std::get<I>(local.values)...);
    }(State::types::indexed);

    if (phase == reduce::phase::REDUCE) {
      auto const end    = static_cast<thread_index_type>(offsets[segment + 1]);
      auto const stride = static_cast<thread_index_type>(blocks_per_segment) * BlockSize;

      for (auto row = static_cast<thread_index_type>(offsets[segment]) +
                      static_cast<thread_index_type>(rank) * BlockSize + threadIdx.x;
           row < end;
           row += stride) {
        reduce::update_row<NullAware, Inputs>(local, inputs, row, filter, update, segment);
      }
    } else {
      auto const base = segment * blocks_per_segment;

      for (auto i = static_cast<thread_index_type>(threadIdx.x); i < blocks_per_segment;
           i += BlockSize) {
        local = do_merge(local, partials[base + i]);
      }
    }

    auto const result_index = phase == reduce::phase::REDUCE ? tile : segment;

    if constexpr (UseWarpReduce) {
      auto combined = reduce::block_combine<BlockSize>(local, do_merge);

      if (threadIdx.x == 0) { results[result_index] = combined; }

    } else {
      auto* block        = scratch + blockIdx.x * BlockSize;
      block[threadIdx.x] = local;

      __syncthreads();

      for (int offset = 1; offset < BlockSize; offset *= 2) {
        auto const rank = static_cast<int>(threadIdx.x);
        if (rank % (2 * offset) == 0 && rank + offset < BlockSize) {
          block[rank] = do_merge(block[rank], block[rank + offset]);
        }

        __syncthreads();
      }

      if (threadIdx.x == 0) { results[result_index] = block[0]; }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      if (finalize_inline && (phase == reduce::phase::MERGE || blocks_per_segment == 1)) {
        reduce::finalize<NullAware, Outputs>(results[result_index],
                                             outputs + segment * Outputs::size,
                                             segment_error,
                                             finalize,
                                             segment);
      }
    }

    // No warp may overwrite shared leaders or scratch while another is still reading it.
    __syncthreads();
  }
}

}  // namespace cudf::jit
