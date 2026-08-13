/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels_common.cuh"

#include <cudf/detail/utilities/grid_1d.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cudf::detail {

/**
 * @brief Count matching build-side rows for each probe key.
 *
 * Each probing tile (@p cg_size threads) calls `ref.count()` for one probe key
 * and reduces the per-lane counts across the tile with a warp reduce. The result
 * is written to @p output by a single elected thread via `invoke_one`. If
 * @p IsOuter is true, keys with zero matches are recorded as 1 so every probe
 * row contributes at least one output row in the subsequent retrieve pass.
 *
 * This is the first phase of the two-phase partitioned join: count then retrieve.
 * The output array is consumed by `launch_partitioned_retrieve` to pre-allocate
 * the output index buffers.
 *
 * @tparam IsOuter  If true, zero-match keys produce a count of 1
 * @tparam Ref      cuco open-addressing reference type (carries hash, equality, storage)
 * @param keys    Packed probe keys: `.first` = hash, `.second` = probe row index
 * @param n       Number of probe keys
 * @param output  Per-key match count output (one entry per probe key)
 * @param ref     cuco hash-table reference for counting
 */
template <bool IsOuter, typename Ref>
CUDF_KERNEL void __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE)
  partitioned_count_kernel(probe_key_type const* __restrict__ keys,
                           thread_index_type n,
                           size_type* __restrict__ output,
                           Ref ref)
{
  auto constexpr cg_size = DEFAULT_JOIN_CG_SIZE;

  auto idx          = grid_1d::global_thread_id() / cg_size;
  auto const stride = grid_1d::grid_stride() / cg_size;

  while (idx < n) {
    auto const key = keys[idx];
    if constexpr (cg_size == 1) {
      auto const match_count = ref.count(key);
      if constexpr (IsOuter) {
        output[idx] = (match_count == 0) ? size_type{1} : match_count;
      } else {
        output[idx] = match_count;
      }
    } else {
      auto const tile =
        cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
      auto const temp_count = static_cast<size_type>(ref.count(tile, key));
      auto const match_count =
        cooperative_groups::reduce(tile, temp_count, cooperative_groups::plus<size_type>());
      cooperative_groups::invoke_one(tile, [&]() {
        if constexpr (IsOuter) {
          output[idx] = (match_count == 0) ? size_type{1} : match_count;
        } else {
          output[idx] = match_count;
        }
      });
    }
    idx += stride;
  }
}

template <bool IsOuter, typename Ref>
void launch_partitioned_count(probe_key_type const* keys,
                              thread_index_type n,
                              size_type* output,
                              Ref ref,
                              rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }

  auto const config =
    grid_1d{static_cast<thread_index_type>(n * DEFAULT_JOIN_CG_SIZE), DEFAULT_JOIN_BLOCK_SIZE};

  partitioned_count_kernel<IsOuter>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(keys, n, output, ref);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::detail
