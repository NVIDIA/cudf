/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>

#include <cub/warp/warp_merge_sort.cuh>

namespace cudf {
namespace detail {

/**
 * @brief Warp-tier `cub::WarpMergeSort` kernel over a segment-size band
 *
 * Key type, builder, comparator and pad are parameters so the packed (null-aware) and raw
 * (no-null) modes and every `(W, IPT)` shape share one body. A virtual warp of `W` lanes owns one
 * segment from `d_seg_list` and returns before sorting when its size falls outside
 * `(band_lo, band_hi]`; all `W` lanes share `seg_size`, so a virtual warp returns together and
 * never desynchronizes the `__syncwarp` inside `WarpMergeSort`, letting several bands share one
 * segment list. Slots past the real size hold the pad key; only `[0, seg_size)` is written back.
 * A function template so the STABLE translation unit, which never instantiates the tiered path,
 * emits no unused `static` kernel under -Werror.
 */
template <typename KeyT,
          typename KeyBuilder,
          int W,
          int IPT,
          int BLOCK_THREADS,
          typename CompareOp,
          typename SegListIt>
CUDF_KERNEL __launch_bounds__(BLOCK_THREADS) void tiered_warp_band_kernel(
  size_type const* d_offsets,
  SegListIt d_seg_list,
  size_type num_class_segments,
  size_type band_lo,
  size_type band_hi,
  KeyBuilder build_key,
  size_type* d_out,
  CompareOp compare_op,
  KeyT pad_key)
{
  using WarpMergeSortT           = cub::WarpMergeSort<KeyT, IPT, W, size_type>;
  constexpr int VWARPS_PER_BLOCK = BLOCK_THREADS / W;
  __shared__ typename WarpMergeSortT::TempStorage temp_storage[VWARPS_PER_BLOCK];

  auto const global_tid = cudf::detail::grid_1d::global_thread_id<BLOCK_THREADS>();
  auto const vwarp_id   = static_cast<size_type>(global_tid / W);
  auto const lane       = static_cast<int>(threadIdx.x % W);
  auto const vwarp_slot = static_cast<int>(threadIdx.x / W);
  if (vwarp_id >= num_class_segments) { return; }

  auto const seg       = d_seg_list[vwarp_id];
  auto const seg_start = d_offsets[seg];
  auto const seg_size  = d_offsets[seg + 1] - seg_start;
  if (seg_size <= band_lo || seg_size > band_hi) { return; }
  // The register tile holds W*IPT elements; a band_hi above that would silently drop elements.
  cudf_assert(seg_size <= W * IPT && "band segment exceeds the register tile (band_hi > W*IPT)");

  KeyT keys[IPT];
  size_type vals[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = lane * IPT + i;
    if (local < seg_size) {
      auto const gidx = seg_start + local;
      keys[i]         = build_key(gidx);
      vals[i]         = gidx;
    } else {
      keys[i] = pad_key;
      vals[i] = size_type{0};
    }
  }

  WarpMergeSortT(temp_storage[vwarp_slot])
    .StableSort(keys, vals, compare_op, static_cast<int>(seg_size), pad_key);

#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    auto const local = lane * IPT + i;
    if (local < seg_size) { d_out[seg_start + local] = vals[i]; }
  }
}

}  // namespace detail
}  // namespace cudf
