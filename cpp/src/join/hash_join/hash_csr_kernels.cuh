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
#include <cuda/stream>

#include <cstdint>

namespace cudf::detail {

constexpr thread_index_type hash_csr_block_size = 256;
constexpr thread_index_type hash_csr_warps_per_block =
  hash_csr_block_size / cudf::detail::warp_size;
constexpr thread_index_type hash_csr_outputs_per_lane = 32;

template <typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_build_count_kernel(size_type num_rows,
                                             bitmask_type const* valid_rows,
                                             hash_csr_build_position_type* build_positions,
                                             size_type* slot_counts,
                                             hash_csr_map_view map,
                                             Equal equal,
                                             Hasher hasher)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    if (valid_rows != nullptr && !cudf::bit_is_set(valid_rows, index)) {
      build_positions[index] = {std::uint32_t{-1}, size_type{CUDF_SIZE_TYPE_SENTINEL}};
      continue;
    }

    auto const slot = map.insert(hash_csr_key_type{hasher(index), index}, equal);
    if (slot == map.capacity) {
      build_positions[index] = {std::uint32_t{-1}, size_type{CUDF_SIZE_TYPE_SENTINEL}};
      continue;
    }
    auto slot_count_ref = cuda::atomic_ref<size_type, cuda::thread_scope_device>{slot_counts[slot]};
    auto const rank     = slot_count_ref.fetch_add(size_type{1}, cuda::memory_order_relaxed);
    build_positions[index] = {slot, rank};
  }
}

CUDF_KERNEL void hash_csr_build_fill_kernel(size_type num_rows,
                                            hash_csr_build_position_type const* build_positions,
                                            size_type const* cumulative_ends,
                                            size_type* values)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index    = static_cast<size_type>(row);
    auto const position = build_positions[index];
    if (position.first == std::uint32_t{-1}) { continue; }
    auto const slot      = position.first;
    auto const rank      = position.second;
    auto const begin     = slot == 0 ? size_type{0} : cumulative_ends[slot - 1];
    values[begin + rank] = index;
  }
}

template <bool IsOuter, typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_probe_count_kernel(size_type num_rows,
                                             bitmask_type const* valid_rows,
                                             size_type* probe_slots,
                                             size_type* match_counts,
                                             std::uint32_t* matched_slots,
                                             unsigned long long* matched_build_rows,
                                             hash_csr_map_view map,
                                             hash_csr_view csr,
                                             Equal equal,
                                             Hasher hasher)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    auto slot        = map.capacity;
    if (valid_rows == nullptr || cudf::bit_is_set(valid_rows, index)) {
      slot = map.find(hash_csr_key_type{hasher(index), index}, equal);
    }

    auto const found = slot != map.capacity;
    auto const count = found ? csr.size(static_cast<size_type>(slot)) : size_type{0};
    if (probe_slots != nullptr) {
      probe_slots[index] = found ? static_cast<size_type>(slot) : CUDF_SIZE_TYPE_SENTINEL;
    }
    if (match_counts != nullptr) {
      match_counts[index] = IsOuter ? cuda::std::max(count, size_type{1}) : count;
    }

    if (found && matched_slots != nullptr) {
      auto matched_slot_ref =
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device>{matched_slots[slot]};
      auto expected = std::uint32_t{0};
      if (matched_slot_ref.compare_exchange_strong(
            expected, std::uint32_t{1}, cuda::memory_order_relaxed)) {
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>{*matched_build_rows}
          .fetch_add(static_cast<unsigned long long>(count), cuda::memory_order_relaxed);
      }
    }
  }
}

template <typename Equal, typename Hasher>
void launch_hash_csr_build_count(size_type num_rows,
                                 bitmask_type const* valid_rows,
                                 hash_csr_build_position_type* build_positions,
                                 size_type* slot_counts,
                                 hash_csr_map_view map,
                                 Equal equal,
                                 Hasher hasher,
                                 cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_build_count_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, valid_rows, build_positions, slot_counts, map, equal, hasher);
  CUDF_CUDA_TRY(cudaGetLastError());
}

[[maybe_unused]] static void launch_hash_csr_build_fill(
  size_type num_rows,
  hash_csr_build_position_type const* build_positions,
  size_type const* cumulative_ends,
  size_type* values,
  cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_build_fill_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, build_positions, cumulative_ends, values);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <bool IsOuter, typename Equal, typename Hasher>
void launch_hash_csr_probe_count(size_type num_rows,
                                 bitmask_type const* valid_rows,
                                 size_type* probe_slots,
                                 size_type* match_counts,
                                 std::uint32_t* matched_slots,
                                 unsigned long long* matched_build_rows,
                                 hash_csr_map_view map,
                                 hash_csr_view csr,
                                 Equal equal,
                                 Hasher hasher,
                                 cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_probe_count_kernel<IsOuter>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(num_rows,
                                                                           valid_rows,
                                                                           probe_slots,
                                                                           match_counts,
                                                                           matched_slots,
                                                                           matched_build_rows,
                                                                           map,
                                                                           csr,
                                                                           equal,
                                                                           hasher);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <bool IsOuter>
CUDF_KERNEL void hash_csr_retrieve_kernel(std::int64_t output_size,
                                          size_type num_probe_rows,
                                          std::int64_t outputs_per_warp,
                                          std::int64_t const* offsets,
                                          size_type const* probe_slots,
                                          hash_csr_view csr,
                                          size_type left_index_offset,
                                          size_type* left_indices,
                                          size_type* right_indices)
{
  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(
    cooperative_groups::this_thread_block());
  auto const lane_id       = static_cast<thread_index_type>(warp.thread_rank());
  auto const warp_in_block = static_cast<thread_index_type>(threadIdx.x) / cudf::detail::warp_size;
  auto const global_warp =
    static_cast<std::int64_t>(blockIdx.x) * hash_csr_warps_per_block + warp_in_block;
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
      auto const slot            = probe_slots[probe_row];
      left_indices[output_index] = probe_row + left_index_offset;
      if constexpr (IsOuter) {
        if (slot == CUDF_SIZE_TYPE_SENTINEL) {
          right_indices[output_index] = JoinNoMatch;
          continue;
        }
      }
      auto const local_match      = static_cast<size_type>(output_index - offsets[probe_row]);
      right_indices[output_index] = csr.values[csr.begin(slot) + local_match];
    }
  }
}

template <bool IsOuter>
void launch_hash_csr_retrieve(std::int64_t output_size,
                              size_type num_probe_rows,
                              std::int64_t const* offsets,
                              size_type const* probe_slots,
                              hash_csr_view csr,
                              size_type left_index_offset,
                              size_type* left_indices,
                              size_type* right_indices,
                              cuda::stream_ref stream)
{
  if (output_size == 0) { return; }
  auto const min_blocks = size_type{2} * cudf::detail::num_multiprocessors();
  constexpr auto outputs_per_block =
    hash_csr_warps_per_block * cudf::detail::warp_size * hash_csr_outputs_per_lane;
  auto const requested_blocks =
    cudf::util::div_rounding_up_safe(output_size, static_cast<std::int64_t>(outputs_per_block));
  auto const num_blocks =
    static_cast<std::uint32_t>(cuda::std::max<std::int64_t>(requested_blocks, min_blocks));
  auto const num_warps        = static_cast<std::int64_t>(num_blocks) * hash_csr_warps_per_block;
  auto const outputs_per_warp = cudf::util::div_rounding_up_safe(output_size, num_warps);

  hash_csr_retrieve_kernel<IsOuter>
    <<<num_blocks, hash_csr_block_size, 0, stream.get()>>>(output_size,
                                                           num_probe_rows,
                                                           outputs_per_warp,
                                                           offsets,
                                                           probe_slots,
                                                           csr,
                                                           left_index_offset,
                                                           left_indices,
                                                           right_indices);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::detail
