/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <reductions/jit/segmented_reduce.cuh>

namespace cudf::jit {

/**
 * @brief Reduces each segment within a fixed-size group of lanes in one warp.
 *
 * A warp processes `warp_size / GroupSize` segments concurrently. Lanes traverse their segment
 * with stride `GroupSize`, combine states with warp shuffles, and let the group's first lane
 * finalize the result. A warp-stride loop covers remaining segments. This completes reduction
 * and finalization in one launch without caller-provided intermediate buffers.
 *
 * All callbacks receive a leading segment ID; `update` additionally receives the input row
 * index, relative to the input views. Identity, null-handling, and error contracts follow
 * `reduce_kernel`. Empty segments and segments with no selected rows still initialize and
 * finalize an identity state. Groups beyond `num_segments` do not invoke callbacks or write
 * results, but their lanes participate in the warp shuffles.
 *
 * Finalization writes row zero of `outputs[segment * Outputs::size + output]` and overwrites
 * its first validity word when present. Views must have offset zero and independent single-row
 * storage. The group's first lane also writes the greatest callback error to
 * `segment_errors[segment]`.
 *
 * @pre Launch one-dimensional blocks of exactly `BlockSize` threads, a power of two between
 * 32 and 1024. All lanes of every participating warp must execute the call.
 * @pre `GroupSize` is a power of two between 1 and 32. Offsets describe nonnegative,
 * nondecreasing ranges within the inputs. `State` supports bytewise copying and warp shuffles.
 *
 * @tparam NullAware Whether callbacks receive nullable input/output values
 * @tparam Inputs Type list of input accessors in update-parameter order
 * @tparam Outputs Type list of output accessors in finalization-parameter order
 * @tparam BlockSize Number of threads per block
 * @tparam GroupSize Number of contiguous lanes cooperating on one segment
 * @tparam Filter Callable selecting rows before the update callback
 * @tparam State State containing `values`, `error`, and the field type list `types`
 * @tparam Init Callable `errc(segment, state pointers...)` that initializes every state field
 * @tparam Update Callable `errc(segment, row, state pointers..., input values...)`
 * @tparam Merge Associative callable `errc(segment, destination pointers..., source values...)`
 * preserving the identity
 * @tparam Finalize Callable `errc(segment, output pointers..., state values...)`
 *
 * @param num_segments Nonnegative number of segments
 * @param offsets Device array of `num_segments + 1` segment boundaries
 * @param inputs Device input views consumed by the input accessors
 * @param outputs Device array of `num_segments * Outputs::size` single-row views, in segment order
 * @param segment_errors Device array of `num_segments` final error codes
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
          int GroupSize,
          typename State,
          typename Init,
          typename Filter,
          typename Update,
          typename Merge,
          typename Finalize>
  requires(BlockSize >= detail::warp_size && BlockSize <= 1024 &&
           BlockSize % detail::warp_size == 0 && (BlockSize & (BlockSize - 1)) == 0 &&
           GroupSize > 0 && GroupSize <= detail::warp_size && (GroupSize & (GroupSize - 1)) == 0)
__device__ void small_groups_segmented_reduce_kernel(
  size_type num_segments,
  size_type const* __restrict__ offsets,
  column_device_view_core const* __restrict__ inputs,
  mutable_column_device_view_core const* __restrict__ outputs,
  errc* __restrict__ segment_errors,
  Init&& init,
  Filter&& filter,
  Update&& update,
  Merge&& merge,
  Finalize&& finalize)
{
  constexpr int groups_per_warp = detail::warp_size / GroupSize;
  auto const lane               = threadIdx.x % detail::warp_size;
  auto const rank               = lane % GroupSize;
  auto const warp =
    (static_cast<thread_index_type>(blockIdx.x) * BlockSize + threadIdx.x) / detail::warp_size;
  auto const stride =
    static_cast<thread_index_type>(gridDim.x) * (BlockSize / detail::warp_size) * groups_per_warp;

  for (auto first_segment = warp * groups_per_warp; first_segment < num_segments;
       first_segment += stride) {
    auto const segment = first_segment + lane / GroupSize;
    auto const active  = segment < num_segments;
    auto do_merge      = [&](auto const& a, auto const& b) {
      return reduce::merge(a, b, merge, segment);
    };

    State local;

    if (active) {
      local.error = [&]<int... I>(cuda::std::integer_sequence<int, I...>) {
        return init(segment, &cuda::std::get<I>(local.values)...);
      }(State::types::indexed);

      for (auto row = static_cast<thread_index_type>(offsets[segment]) + rank;
           row < offsets[segment + 1];
           row += GroupSize) {
        reduce::update_row<NullAware, Inputs>(local, inputs, row, filter, update, segment);
      }
    }

    // Every physical lane shuffles; only real segments invoke merge callbacks.
    auto combined = reduce::warp_combine<GroupSize>(local, do_merge, active);
    if (active && rank == 0) {
      reduce::finalize<NullAware, Outputs>(
        combined, outputs + segment * Outputs::size, segment_errors + segment, finalize, segment);
    }
  }
}

}  // namespace cudf::jit
