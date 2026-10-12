/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "dispatch.cuh"
#include "hash_csr.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <cuda/stream>

namespace cudf::detail {

// `dispatch_join_comparator` has exactly these three concrete comparator forms. Keep their
// aliases here so the count launchers can be explicitly instantiated in hash_csr_kernels.cu.
// This prevents every HashCSR caller from emitting an identical copy of the corresponding
// device kernel.
using hash_csr_dynamic_nulls = cudf::nullate::DYNAMIC;
using hash_csr_row_hasher =
  decltype(std::declval<cudf::detail::row::hash::row_hasher const&>().device_hasher(
    std::declval<hash_csr_dynamic_nulls>()));
using hash_csr_primitive_hasher =
  cudf::detail::row::primitive::row_hasher<cudf::hashing::detail::default_hash>;
using hash_csr_primitive_equal = primitive_pair_equal;
using hash_csr_non_nested_equal =
  pair_equal<decltype(std::declval<cudf::detail::row::equality::two_table_comparator const&>()
                        .template equal_to<false>(std::declval<hash_csr_dynamic_nulls>(),
                                                  std::declval<null_equality>()))>;
using hash_csr_nested_equal =
  pair_equal<decltype(std::declval<cudf::detail::row::equality::two_table_comparator const&>()
                        .template equal_to<true>(std::declval<hash_csr_dynamic_nulls>(),
                                                 std::declval<null_equality>()))>;

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

extern template void
launch_hash_csr_build_count_kernel<hash_csr_primitive_equal, hash_csr_primitive_hasher>(
  size_type,
  bitmask_type const*,
  size_type*,
  size_type*,
  hash_table_ref,
  hash_csr_primitive_equal,
  hash_csr_primitive_hasher,
  cuda::stream_ref);
extern template void
launch_hash_csr_build_count_kernel<hash_csr_non_nested_equal, hash_csr_row_hasher>(
  size_type,
  bitmask_type const*,
  size_type*,
  size_type*,
  hash_table_ref,
  hash_csr_non_nested_equal,
  hash_csr_row_hasher,
  cuda::stream_ref);
extern template void launch_hash_csr_build_count_kernel<hash_csr_nested_equal, hash_csr_row_hasher>(
  size_type,
  bitmask_type const*,
  size_type*,
  size_type*,
  hash_table_ref,
  hash_csr_nested_equal,
  hash_csr_row_hasher,
  cuda::stream_ref);

#define CUDF_EXTERN_HASH_CSR_PROBE_COUNT(OUTER, EQUAL, HASHER)                   \
  extern template void launch_hash_csr_probe_count_kernel<OUTER, EQUAL, HASHER>( \
    size_type,                                                                   \
    bitmask_type const*,                                                         \
    size_type*,                                                                  \
    size_type*,                                                                  \
    cuda::std::uint32_t*,                                                        \
    cuda::std::uint64_t*,                                                        \
    hash_table_ref,                                                              \
    csr_ref,                                                                     \
    EQUAL,                                                                       \
    HASHER,                                                                      \
    cuda::stream_ref)

CUDF_EXTERN_HASH_CSR_PROBE_COUNT(false, hash_csr_primitive_equal, hash_csr_primitive_hasher);
CUDF_EXTERN_HASH_CSR_PROBE_COUNT(true, hash_csr_primitive_equal, hash_csr_primitive_hasher);
CUDF_EXTERN_HASH_CSR_PROBE_COUNT(false, hash_csr_non_nested_equal, hash_csr_row_hasher);
CUDF_EXTERN_HASH_CSR_PROBE_COUNT(true, hash_csr_non_nested_equal, hash_csr_row_hasher);
CUDF_EXTERN_HASH_CSR_PROBE_COUNT(false, hash_csr_nested_equal, hash_csr_row_hasher);
CUDF_EXTERN_HASH_CSR_PROBE_COUNT(true, hash_csr_nested_equal, hash_csr_row_hasher);

#undef CUDF_EXTERN_HASH_CSR_PROBE_COUNT

void launch_hash_csr_inner_retrieve_kernel(cuda::std::int64_t output_size,
                                           size_type num_probe_rows,
                                           cuda::std::int64_t const* offsets,
                                           size_type const* probe_groups,
                                           csr_ref csr,
                                           size_type left_index_offset,
                                           size_type* left_indices,
                                           size_type* right_indices,
                                           cuda::stream_ref stream);

void launch_hash_csr_outer_retrieve_kernel(cuda::std::int64_t output_size,
                                           size_type num_probe_rows,
                                           cuda::std::int64_t const* offsets,
                                           size_type const* probe_groups,
                                           csr_ref csr,
                                           size_type left_index_offset,
                                           size_type* left_indices,
                                           size_type* right_indices,
                                           cuda::stream_ref stream);

}  // namespace cudf::detail
