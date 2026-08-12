/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "hash_csr.cuh"

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/execution>
#include <cuda/iterator>
#include <cuda/memory_resource>
#include <cuda/std/functional>
#include <cuda/stream>

#include <algorithm>
#include <cstdint>
#include <optional>

namespace cudf::detail {

constexpr thread_index_type hash_csr_block_size = 256;
constexpr thread_index_type hash_csr_warps_per_block =
  hash_csr_block_size / cudf::detail::warp_size;
constexpr thread_index_type hash_csr_outputs_per_lane = 32;

template <typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_build_count_kernel(size_type num_rows,
                                             bitmask_type const* valid_rows,
                                             std::uint64_t* build_positions,
                                             size_type* slot_counts,
                                             hash_csr_map_view map,
                                             Equal equal,
                                             Hasher hasher)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    if (valid_rows != nullptr && !cudf::bit_is_set(valid_rows, index)) {
      build_positions[index] = hash_csr_empty_build_position;
      continue;
    }

    auto const slot = map.insert(hash_csr_key_type{hasher(index), index}, equal);
    if (slot == map.capacity) {
      build_positions[index] = hash_csr_empty_build_position;
      continue;
    }
    auto slot_count_ref = cuda::atomic_ref<size_type, cuda::thread_scope_device>{slot_counts[slot]};
    auto const rank     = slot_count_ref.fetch_add(size_type{1}, cuda::memory_order_relaxed);
    build_positions[index] = pack_hash_csr_build_position(slot, rank);
  }
}

CUDF_KERNEL void hash_csr_build_fill_kernel(size_type num_rows,
                                            std::uint64_t const* build_positions,
                                            size_type const* cumulative_ends,
                                            size_type* values)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index    = static_cast<size_type>(row);
    auto const position = build_positions[index];
    if (position == hash_csr_empty_build_position) { continue; }
    auto const slot      = unpack_hash_csr_build_slot(position);
    auto const rank      = unpack_hash_csr_build_rank(position);
    auto const begin     = slot == 0 ? size_type{0} : cumulative_ends[slot - 1];
    values[begin + rank] = index;
  }
}

template <bool is_outer, typename Equal, typename Hasher>
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
      probe_slots[index] = found ? static_cast<size_type>(slot) : hash_csr_empty_slot;
    }
    if (match_counts != nullptr) {
      match_counts[index] = is_outer ? cuda::std::max(count, size_type{1}) : count;
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
                                 std::uint64_t* build_positions,
                                 size_type* slot_counts,
                                 hash_csr_map_view map,
                                 Equal equal,
                                 Hasher hasher,
                                 rmm::cuda_stream_view stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_build_count_kernel<<<config.num_blocks,
                                config.num_threads_per_block,
                                0,
                                stream.value()>>>(
    num_rows, valid_rows, build_positions, slot_counts, map, equal, hasher);
  CUDF_CUDA_TRY(cudaGetLastError());
}

[[maybe_unused]] static void launch_hash_csr_build_fill(size_type num_rows,
                                                        std::uint64_t const* build_positions,
                                                        size_type const* cumulative_ends,
                                                        size_type* values,
                                                        rmm::cuda_stream_view stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_build_fill_kernel<<<config.num_blocks,
                               config.num_threads_per_block,
                               0,
                               stream.value()>>>(
    num_rows, build_positions, cumulative_ends, values);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <bool is_outer, typename Equal, typename Hasher>
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
                                 rmm::cuda_stream_view stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_probe_count_kernel<is_outer>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(num_rows,
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

template <typename InputIterator, typename OutputIterator, typename InitialValue>
void hash_csr_exclusive_scan(InputIterator input,
                             OutputIterator output,
                             size_type num_items,
                             InitialValue initial_value,
                             rmm::cuda_stream_view stream)
{
  auto env = cuda::std::execution::env{cuda::stream_ref{stream.value()},
                                       cudf::get_current_device_resource_ref()};
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveScan(
    input, output, cuda::std::plus<>{}, initial_value, num_items, env));
}

[[maybe_unused]] static void hash_csr_inclusive_sum(size_type* values,
                                                    size_type size,
                                                    rmm::cuda_stream_view stream)
{
  auto env = cuda::std::execution::env{cuda::stream_ref{stream.value()},
                                       cudf::get_current_device_resource_ref()};
  CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(values, values, size, env));
}

[[maybe_unused]] static std::int64_t hash_csr_scan_counts(size_type const* counts,
                                                          size_type num_rows,
                                                          std::int64_t* offsets,
                                                          rmm::cuda_stream_view stream)
{
  auto const mr = cudf::get_current_device_resource_ref();
  cudf::detail::device_scalar<std::int64_t> output_size(stream, mr);
  auto output = cudf::detail::make_sizes_to_offsets_iterator(
    offsets, offsets + num_rows + 1, output_size.data());
  hash_csr_exclusive_scan(counts, output, num_rows + 1, std::int64_t{0}, stream);
  return output_size.value(stream);
}

struct hash_csr_count_to_int64 {
  __device__ std::int64_t operator()(size_type value) const
  {
    return static_cast<std::int64_t>(value);
  }
};

[[maybe_unused]] static std::int64_t hash_csr_reduce_counts(size_type const* counts,
                                                            size_type num_rows,
                                                            rmm::cuda_stream_view stream)
{
  if (num_rows == 0) { return 0; }
  auto const mr = cudf::get_current_device_resource_ref();
  auto input    = cuda::transform_iterator{counts, hash_csr_count_to_int64{}};
  cudf::detail::device_scalar<std::int64_t> result(stream, mr);
  auto env = cuda::std::execution::env{cuda::stream_ref{stream.value()}, mr};
  CUDF_CUDA_TRY(cub::DeviceReduce::Sum(input, result.data(), num_rows, env));
  return result.value(stream);
}

static __device__ size_type hash_csr_find_probe_row_in_range(std::int64_t const* offsets,
                                                             size_type first_probe,
                                                             size_type last_probe,
                                                             std::int64_t output_index)
{
  auto first = static_cast<std::int64_t>(first_probe);
  auto last  = static_cast<std::int64_t>(last_probe) + 2;
  while (first < last) {
    auto const middle = first + (last - first) / 2;
    if (offsets[middle] <= output_index) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  return static_cast<size_type>(first - 1);
}

template <bool is_outer>
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
  auto const lane_id       = static_cast<thread_index_type>(threadIdx.x) % cudf::detail::warp_size;
  auto const warp_in_block = static_cast<thread_index_type>(threadIdx.x) / cudf::detail::warp_size;
  auto const global_warp =
    static_cast<std::int64_t>(blockIdx.x) * hash_csr_warps_per_block + warp_in_block;
  auto const range_begin = outputs_per_warp * global_warp;
  if (range_begin >= output_size) { return; }
  auto const range_end = cuda::std::min(range_begin + outputs_per_warp, output_size);

  size_type endpoint_probe{};
  if (lane_id < 2) {
    auto const endpoint = lane_id == 0 ? range_begin : range_end - 1;
    endpoint_probe = hash_csr_find_probe_row_in_range(offsets, 0, num_probe_rows - 1, endpoint);
  }
  auto const first_probe = __shfl_sync(0xffffffff, endpoint_probe, 0);
  auto const last_probe  = __shfl_sync(0xffffffff, endpoint_probe, 1);

#pragma unroll
  for (thread_index_type item = 0; item < hash_csr_outputs_per_lane; ++item) {
    auto const output_index = range_begin + lane_id + item * cudf::detail::warp_size;
    if (output_index < range_end) {
      auto const probe_row =
        first_probe == last_probe
          ? first_probe
          : hash_csr_find_probe_row_in_range(offsets, first_probe, last_probe, output_index);
      auto const slot            = probe_slots[probe_row];
      left_indices[output_index] = probe_row + left_index_offset;
      if constexpr (is_outer) {
        if (slot == hash_csr_empty_slot) {
          right_indices[output_index] = JoinNoMatch;
          continue;
        }
      }
      auto const local_match      = static_cast<size_type>(output_index - offsets[probe_row]);
      right_indices[output_index] = csr.values[csr.begin(slot) + local_match];
    }
  }
}

template <bool is_outer>
void launch_hash_csr_retrieve(std::int64_t output_size,
                              size_type num_probe_rows,
                              std::int64_t const* offsets,
                              size_type const* probe_slots,
                              hash_csr_view csr,
                              size_type left_index_offset,
                              size_type* left_indices,
                              size_type* right_indices,
                              rmm::cuda_stream_view stream)
{
  if (output_size == 0) { return; }
  auto const min_blocks = size_type{2} * cudf::detail::num_multiprocessors();
  constexpr auto outputs_per_block =
    hash_csr_warps_per_block * cudf::detail::warp_size * hash_csr_outputs_per_lane;
  auto const requested_blocks = (output_size + outputs_per_block - 1) / outputs_per_block;
  auto const num_blocks =
    static_cast<std::uint32_t>(cuda::std::max<std::int64_t>(requested_blocks, min_blocks));
  auto const num_warps        = static_cast<std::int64_t>(num_blocks) * hash_csr_warps_per_block;
  auto const outputs_per_warp = (output_size + num_warps - 1) / num_warps;

  hash_csr_retrieve_kernel<is_outer>
    <<<num_blocks, hash_csr_block_size, 0, stream.value()>>>(output_size,
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
