/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "hash_csr.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <cooperative_groups.h>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <cuda/stream>

namespace cudf::detail {

constexpr thread_index_type hash_csr_block_size = 256;
constexpr thread_index_type hash_csr_warps_per_block =
  hash_csr_block_size / cudf::detail::warp_size;
constexpr thread_index_type hash_csr_outputs_per_lane = 32;

template <typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_build_count_kernel(size_type num_rows,
                                             bitmask_type const* valid_rows,
                                             size_type* counts,
                                             size_type* representatives,
                                             hash_table_ref map,
                                             Equal equal,
                                             Hasher hasher)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index    = static_cast<size_type>(row);
    auto representative = size_type{CUDF_SIZE_TYPE_SENTINEL};
    if (valid_rows == nullptr || cudf::bit_is_set(valid_rows, index)) {
      representative = map.insert(cuda::std::pair{hasher(index), index}, equal);
    }
    // Initialize excluded rows too: the cached fill pass only reads representatives.
    if (representatives != nullptr) { representatives[index] = representative; }
    if (representative == CUDF_SIZE_TYPE_SENTINEL) { continue; }
    auto const peers = __match_any_sync(__activemask(), representative);
    auto const lane  = threadIdx.x % cudf::detail::warp_size;
    if (lane == __ffs(peers) - 1) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device>{counts[representative]}.fetch_add(
        __popc(peers), cuda::memory_order_relaxed);
    }
  }
}

__device__ inline void hash_csr_scatter_build_row(size_type index,
                                                  size_type representative,
                                                  size_type* offsets,
                                                  size_type* values)
{
  auto const peers  = __match_any_sync(__activemask(), representative);
  auto const lane   = threadIdx.x % cudf::detail::warp_size;
  auto const leader = __ffs(peers) - 1;
  size_type end{};
  if (lane == leader) {
    end = cuda::atomic_ref<size_type, cuda::thread_scope_device>{offsets[representative]}.fetch_sub(
      __popc(peers), cuda::memory_order_relaxed);
  }
  end                    = __shfl_sync(peers, end, leader);
  auto const rank        = __popc(peers & ((cuda::std::uint32_t{1} << lane) - 1));
  values[end - rank - 1] = index;
}

CUDF_KERNEL void hash_csr_build_fill_cached_kernel(size_type num_rows,
                                                   size_type const* representatives,
                                                   size_type* offsets,
                                                   size_type* values)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index          = static_cast<size_type>(row);
    auto const representative = representatives[index];
    if (representative == CUDF_SIZE_TYPE_SENTINEL) { continue; }
    hash_csr_scatter_build_row(index, representative, offsets, values);
  }
}

template <typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_build_fill_kernel(size_type num_rows,
                                            bitmask_type const* valid_rows,
                                            size_type* offsets,
                                            size_type* values,
                                            hash_table_ref map,
                                            Equal equal,
                                            Hasher hasher)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    if (valid_rows != nullptr && !cudf::bit_is_set(valid_rows, index)) { continue; }
    auto const representative = map.find<true>(cuda::std::pair{hasher(index), index}, equal);
    if (representative == CUDF_SIZE_TYPE_SENTINEL) { continue; }
    hash_csr_scatter_build_row(index, representative, offsets, values);
  }
}

template <bool IsOuter, typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_probe_count_kernel(size_type num_rows,
                                             bitmask_type const* valid_rows,
                                             size_type* probe_groups,
                                             size_type* match_counts,
                                             cuda::std::uint32_t* matched_groups,
                                             cuda::std::uint64_t* matched_build_rows,
                                             hash_table_ref map,
                                             csr_ref csr,
                                             Equal equal,
                                             Hasher hasher)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    auto group       = size_type{CUDF_SIZE_TYPE_SENTINEL};
    if (valid_rows == nullptr || cudf::bit_is_set(valid_rows, index)) {
      group = map.find(cuda::std::pair{hasher(index), index}, equal);
    }

    auto const found = group != CUDF_SIZE_TYPE_SENTINEL;
    auto const count = found ? csr.size(static_cast<size_type>(group)) : size_type{0};
    if (probe_groups != nullptr) {
      probe_groups[index] = found ? static_cast<size_type>(group) : CUDF_SIZE_TYPE_SENTINEL;
    }
    if (match_counts != nullptr) {
      match_counts[index] = IsOuter ? cuda::std::max(count, size_type{1}) : count;
    }

    // Only right and full joins consume the matched-row tally, and `matched_groups` is null for
    // every other kind, so this whole block compiles away outside outer joins rather than costing
    // a branch per probe row.
    if constexpr (IsOuter) {
      if (found && matched_groups != nullptr) {
        auto matched_group_ref =
          cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{matched_groups[group]};
        auto expected = cuda::std::uint32_t{0};
        if (matched_group_ref.compare_exchange_strong(
              expected, cuda::std::uint32_t{1}, cuda::memory_order_relaxed)) {
          cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device>{*matched_build_rows}
            .fetch_add(static_cast<cuda::std::uint64_t>(count), cuda::memory_order_relaxed);
        }
      }
    }
  }
}

template <typename Equal, typename Hasher>
void launch_hash_csr_build_count_kernel(size_type num_rows,
                                        bitmask_type const* valid_rows,
                                        size_type* counts,
                                        size_type* representatives,
                                        hash_table_ref map,
                                        Equal equal,
                                        Hasher hasher,
                                        cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_build_count_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, valid_rows, counts, representatives, map, equal, hasher);
  CUDF_CUDA_TRY(cudaGetLastError());
}

inline void launch_hash_csr_build_fill_cached_kernel(size_type num_rows,
                                                     size_type const* representatives,
                                                     size_type* offsets,
                                                     size_type* values,
                                                     cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_build_fill_cached_kernel<<<config.num_blocks,
                                      config.num_threads_per_block,
                                      0,
                                      stream.get()>>>(num_rows, representatives, offsets, values);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename Equal, typename Hasher>
void launch_hash_csr_build_fill_kernel(size_type num_rows,
                                       bitmask_type const* valid_rows,
                                       size_type* offsets,
                                       size_type* values,
                                       hash_table_ref map,
                                       Equal equal,
                                       Hasher hasher,
                                       cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_build_fill_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, valid_rows, offsets, values, map, equal, hasher);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <bool IsOuter, typename Equal, typename Hasher>
void launch_hash_csr_probe_count_kernel(size_type num_rows,
                                        bitmask_type const* valid_rows,
                                        size_type* probe_groups,
                                        size_type* match_counts,
                                        cuda::std::uint32_t* matched_groups,
                                        cuda::std::uint64_t* matched_build_rows,
                                        hash_table_ref map,
                                        csr_ref csr,
                                        Equal equal,
                                        Hasher hasher,
                                        cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_probe_count_kernel<IsOuter>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(num_rows,
                                                                           valid_rows,
                                                                           probe_groups,
                                                                           match_counts,
                                                                           matched_groups,
                                                                           matched_build_rows,
                                                                           map,
                                                                           csr,
                                                                           equal,
                                                                           hasher);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <bool IsOuter>
CUDF_KERNEL void hash_csr_retrieve_kernel(cuda::std::int64_t output_size,
                                          size_type num_probe_rows,
                                          cuda::std::int64_t outputs_per_warp,
                                          cuda::std::int64_t const* offsets,
                                          size_type const* probe_groups,
                                          csr_ref csr,
                                          size_type left_index_offset,
                                          size_type* left_indices,
                                          size_type* right_indices)
{
  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(
    cooperative_groups::this_thread_block());
  auto const lane_id       = static_cast<thread_index_type>(warp.thread_rank());
  auto const warp_in_block = static_cast<thread_index_type>(threadIdx.x) / cudf::detail::warp_size;
  auto const global_warp =
    static_cast<cuda::std::int64_t>(blockIdx.x) * hash_csr_warps_per_block + warp_in_block;
  auto const range_begin = outputs_per_warp * global_warp;
  if (range_begin >= output_size) { return; }
  auto const range_end = cuda::std::min(range_begin + outputs_per_warp, output_size);

  size_type endpoint_probe{};
  if (lane_id < 2) {
    auto const endpoint = lane_id == 0 ? range_begin : range_end - 1;
    endpoint_probe      = static_cast<size_type>(
      cuda::std::upper_bound(offsets, offsets + num_probe_rows + 1, endpoint) - offsets - 1);
  }
  auto const first_probe = warp.shfl(endpoint_probe, 0);
  auto const last_probe  = warp.shfl(endpoint_probe, 1);

#pragma unroll
  for (thread_index_type item = 0; item < hash_csr_outputs_per_lane; ++item) {
    auto const output_index = range_begin + lane_id + item * cudf::detail::warp_size;
    if (output_index < range_end) {
      auto const probe_row =
        first_probe == last_probe
          ? first_probe
          : static_cast<size_type>(cuda::std::upper_bound(offsets + first_probe,
                                                          offsets + last_probe + 2,
                                                          output_index) -
                                   offsets - 1);
      auto const group           = probe_groups[probe_row];
      left_indices[output_index] = probe_row + left_index_offset;
      if constexpr (IsOuter) {
        if (group == CUDF_SIZE_TYPE_SENTINEL) {
          right_indices[output_index] = JoinNoMatch;
          continue;
        }
      }
      auto const local_match      = static_cast<size_type>(output_index - offsets[probe_row]);
      right_indices[output_index] = csr.values[csr.begin(group) + local_match];
    }
  }
}

template <bool IsOuter>
void launch_hash_csr_retrieve_kernel(cuda::std::int64_t output_size,
                                     size_type num_probe_rows,
                                     cuda::std::int64_t const* offsets,
                                     size_type const* probe_groups,
                                     csr_ref csr,
                                     size_type left_index_offset,
                                     size_type* left_indices,
                                     size_type* right_indices,
                                     cuda::stream_ref stream)
{
  if (output_size == 0) { return; }
  auto const min_blocks = size_type{2} * cudf::detail::num_multiprocessors();
  constexpr auto outputs_per_block =
    hash_csr_warps_per_block * cudf::detail::warp_size * hash_csr_outputs_per_lane;
  auto const requested_blocks = cudf::util::div_rounding_up_safe(
    output_size, static_cast<cuda::std::int64_t>(outputs_per_block));
  auto const num_blocks = static_cast<cuda::std::uint32_t>(
    cuda::std::max<cuda::std::int64_t>(requested_blocks, min_blocks));
  auto const num_warps = static_cast<cuda::std::int64_t>(num_blocks) * hash_csr_warps_per_block;
  auto const outputs_per_warp = cudf::util::div_rounding_up_safe(output_size, num_warps);

  hash_csr_retrieve_kernel<IsOuter>
    <<<num_blocks, hash_csr_block_size, 0, stream.get()>>>(output_size,
                                                           num_probe_rows,
                                                           outputs_per_warp,
                                                           offsets,
                                                           probe_groups,
                                                           csr,
                                                           left_index_offset,
                                                           left_indices,
                                                           right_indices);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::detail
