/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels_common.cuh"

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cuco/pair.cuh>
#include <cuda/atomic>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <thrust/reduce.h>

namespace cudf::detail {

/**
 * @brief Count the number of set bits below a given position in a bitmask.
 */
__device__ __forceinline__ int count_lower_set_bits(unsigned int mask, int pos)
{
  return cuda::std::popcount(mask & ((1u << pos) - 1));
}

/**
 * @brief Retrieve matching build-side rows for each probe key.
 *
 * Each probing tile (@p cg_size threads) walks the hash table for one probe key,
 * collecting matches via warp ballot. Matches are staged in a per-flushing-tile (warp)
 * shared-memory buffer instead of being written directly to global memory. When the buffer
 * nears capacity, the flushing tile claims a contiguous range in the global output arrays
 * via a single atomic and flushes with coalesced writes, amortising atomic overhead across
 * many matches. If @p IsOuter is true, probe rows with no matches emit a
 * `(left_index, JoinNoMatch)` pair.
 *
 * @tparam IsOuter  If true, unmatched probe rows emit a null-padded output row
 * @tparam Ref      cuco open-addressing reference type (carries hash, equality, storage)
 * @param keys           Packed left keys: `.first` = hash, `.second` = left row index
 * @param n              Number of probe keys
 * @param left_offset    Added to each probe row index to produce an absolute left index
 * @param left_output    Output buffer for left (probe-side) row indices
 * @param right_output   Output buffer for right (build-side) row indices
 * @param output_counter Global atomic counter tracking total pairs written so far
 * @param ref            cuco hash-table reference for probing
 */
template <bool IsOuter, typename Ref>
CUDF_KERNEL void __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE)
  partitioned_retrieve_kernel(probe_key_type const* __restrict__ keys,
                              thread_index_type n,
                              size_type left_offset,
                              size_type* __restrict__ left_output,
                              size_type* __restrict__ right_output,
                              size_type* __restrict__ output_counter,
                              Ref ref)
{
  namespace cg = cooperative_groups;

  auto constexpr cg_size            = Ref::cg_size;
  auto constexpr bucket_size        = Ref::bucket_size;
  auto constexpr flushing_tile_size = 32;  // full warp for coalesced flushes
  static_assert(flushing_tile_size >= cg_size);
  static_assert(flushing_tile_size % cg_size == 0,
                "Every probing tile must sit inside a single flushing tile");
  static_assert(DEFAULT_JOIN_BLOCK_SIZE % flushing_tile_size == 0);

  auto constexpr num_flushing_tiles   = DEFAULT_JOIN_BLOCK_SIZE / flushing_tile_size;
  auto constexpr tiles_in_block       = DEFAULT_JOIN_BLOCK_SIZE / cg_size;
  auto constexpr max_matches_per_step = flushing_tile_size * bucket_size;
  // buffer_size leaves headroom so one full probing step can't overflow.
  auto constexpr buffer_size = max_matches_per_step + flushing_tile_size;

  using index_pair = cuco::pair<size_type, size_type>;
  __shared__ index_pair buffers[num_flushing_tiles][buffer_size];
  __shared__ cuda::std::int32_t counters[num_flushing_tiles];

  auto const block            = cg::this_thread_block();
  auto const flushing_tile    = cg::tiled_partition<flushing_tile_size>(block);
  auto const probing_tile     = cg::tiled_partition<cg_size>(block);
  auto const flushing_tile_id = flushing_tile.meta_group_rank();
  auto const empty_sentinel   = ref.empty_key_sentinel();
  auto const key_equal        = ref.key_eq();

  if (flushing_tile.thread_rank() == 0) { counters[flushing_tile_id] = 0; }
  flushing_tile.sync();

  auto atomic_counter = cuda::atomic_ref<size_type, cuda::thread_scope_device>{*output_counter};

  auto flush_buffers = [&](auto const& tile) {
    auto const count  = counters[flushing_tile_id];
    auto const offset = cg::invoke_one_broadcast(tile, [&]() {
      return atomic_counter.fetch_add(static_cast<size_type>(count), cuda::memory_order_relaxed);
    });
    auto const rank   = tile.thread_rank();
    for (int i = rank; i < count; i += tile.size()) {
      left_output[offset + i]  = buffers[flushing_tile_id][i].first;
      right_output[offset + i] = buffers[flushing_tile_id][i].second;
    }
  };

  auto const grid_stride_tiles = static_cast<thread_index_type>(gridDim.x) * tiles_in_block;
  auto idx =
    static_cast<thread_index_type>(blockIdx.x) * tiles_in_block + probing_tile.meta_group_rank();

  while (flushing_tile.any(idx < n)) {
    bool const active = idx < n;
    auto const active_flushing_tile =
      cg::binary_partition<flushing_tile_size>(flushing_tile, active);

    if (active) {
      auto const probe_key  = keys[idx];
      auto const left_index = probe_key.second + left_offset;

      auto probing_iter = ref.probing_scheme().template make_iterator<bucket_size>(
        probing_tile, probe_key, ref.storage_ref().extent());
      auto const init_probing_idx = *probing_iter;

      bool running                      = true;
      [[maybe_unused]] bool found_match = false;

      while (active_flushing_tile.any(running)) {
        if (running) {
          auto const bucket_slots = ref.storage_ref()[*probing_iter];

          bool equals[bucket_size];
          for (int i = 0; i < bucket_size; ++i) {
            equals[i] = false;
            if (running) {
              if (bucket_slots[i] == empty_sentinel) {
                running = false;
              } else if (key_equal(probe_key, bucket_slots[i])) {
                equals[i] = true;
              }
            }
          }

          probing_tile.sync();
          running = probing_tile.all(running);

          cuda::std::int32_t exists[bucket_size];
          cuda::std::int32_t num_matches[bucket_size];
          cuda::std::int32_t total_matches = 0;
          for (int i = 0; i < bucket_size; ++i) {
            exists[i]      = probing_tile.ballot(equals[i]);
            num_matches[i] = cuda::std::popcount(static_cast<unsigned>(exists[i]));
            total_matches += num_matches[i];
          }

          auto const lane_id = probing_tile.thread_rank();

          if (total_matches > 0) {
            if constexpr (IsOuter) { found_match = true; }

            cuda::std::int32_t output_idx = 0;
            if (lane_id == 0) {
              auto shared_ref = cuda::atomic_ref<cuda::std::int32_t, cuda::thread_scope_block>{
                counters[flushing_tile_id]};
              output_idx = shared_ref.fetch_add(total_matches, cuda::memory_order_relaxed);
            }
            output_idx = probing_tile.shfl(output_idx, 0);

            cuda::std::int32_t matches_offset = 0;
            for (int i = 0; i < bucket_size; ++i) {
              if (equals[i]) {
                auto const lane_offset = count_lower_set_bits(exists[i], lane_id);
                buffers[flushing_tile_id][output_idx + matches_offset + lane_offset] = {
                  left_index, bucket_slots[i].second};
              }
              matches_offset += num_matches[i];
            }
          }

          if constexpr (IsOuter) {
            if (!running && !found_match && lane_id == 0) {
              auto shared_ref = cuda::atomic_ref<cuda::std::int32_t, cuda::thread_scope_block>{
                counters[flushing_tile_id]};
              auto const output_idx = shared_ref.fetch_add(1, cuda::memory_order_relaxed);
              buffers[flushing_tile_id][output_idx] = {left_index, cudf::JoinNoMatch};
            }
          }
        }  // if running

        active_flushing_tile.sync();
        if (counters[flushing_tile_id] > (buffer_size - max_matches_per_step)) {
          flush_buffers(active_flushing_tile);
          active_flushing_tile.sync();
          if (active_flushing_tile.thread_rank() == 0) { counters[flushing_tile_id] = 0; }
          active_flushing_tile.sync();
        }

        ++probing_iter;
        if (*probing_iter == init_probing_idx) { running = false; }
      }  // while running
    }  // if active

    idx += grid_stride_tiles;
  }  // while idx < n

  flushing_tile.sync();
  if (counters[flushing_tile_id] > 0) { flush_buffers(flushing_tile); }
}

template <bool IsOuter, typename Ref>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
launch_partitioned_retrieve(probe_key_type const* keys,
                            thread_index_type n,
                            size_type const* match_counts,
                            Ref ref,
                            size_type left_offset,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  if (n == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  // Shared-memory buffered retrieve only needs the total output size, not
  // per-row offsets.  A reduce is cheaper than an exclusive_scan.
  auto const total_output =
    thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   match_counts,
                   match_counts + n,
                   size_type{0});

  if (total_output == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(total_output, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(total_output, stream, mr);

  // Global atomic counter claimed in bulk by each flushing-tile buffer flush.
  cudf::detail::device_scalar<size_type> output_counter(size_type{0}, stream);

  auto constexpr tiles_in_block = DEFAULT_JOIN_BLOCK_SIZE / Ref::cg_size;
  auto const num_blocks = static_cast<unsigned int>((n + tiles_in_block - 1) / tiles_in_block);

  partitioned_retrieve_kernel<IsOuter><<<num_blocks, DEFAULT_JOIN_BLOCK_SIZE, 0, stream.value()>>>(
    keys, n, left_offset, left_indices->data(), right_indices->data(), output_counter.data(), ref);
  CUDF_CUDA_TRY(cudaGetLastError());

  return std::pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace cudf::detail
