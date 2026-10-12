/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hash_csr_kernels.cuh"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/join/join.hpp>

#include <cooperative_groups.h>
#include <cuda/std/algorithm>

namespace cudf::detail {
namespace {

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

}  // namespace

void launch_hash_csr_inner_retrieve_kernel(cuda::std::int64_t output_size,
                                           size_type num_probe_rows,
                                           cuda::std::int64_t const* offsets,
                                           size_type const* probe_groups,
                                           csr_ref csr,
                                           size_type left_index_offset,
                                           size_type* left_indices,
                                           size_type* right_indices,
                                           cuda::stream_ref stream)
{
  launch_hash_csr_retrieve_kernel<false>(output_size,
                                         num_probe_rows,
                                         offsets,
                                         probe_groups,
                                         csr,
                                         left_index_offset,
                                         left_indices,
                                         right_indices,
                                         stream);
}

void launch_hash_csr_outer_retrieve_kernel(cuda::std::int64_t output_size,
                                           size_type num_probe_rows,
                                           cuda::std::int64_t const* offsets,
                                           size_type const* probe_groups,
                                           csr_ref csr,
                                           size_type left_index_offset,
                                           size_type* left_indices,
                                           size_type* right_indices,
                                           cuda::stream_ref stream)
{
  launch_hash_csr_retrieve_kernel<true>(output_size,
                                        num_probe_rows,
                                        offsets,
                                        probe_groups,
                                        csr,
                                        left_index_offset,
                                        left_indices,
                                        right_indices,
                                        stream);
}

template void
launch_hash_csr_build_count_kernel<hash_csr_primitive_equal, hash_csr_primitive_hasher>(
  size_type,
  bitmask_type const*,
  size_type*,
  size_type*,
  hash_table_ref,
  hash_csr_primitive_equal,
  hash_csr_primitive_hasher,
  cuda::stream_ref);
template void launch_hash_csr_build_count_kernel<hash_csr_non_nested_equal, hash_csr_row_hasher>(
  size_type,
  bitmask_type const*,
  size_type*,
  size_type*,
  hash_table_ref,
  hash_csr_non_nested_equal,
  hash_csr_row_hasher,
  cuda::stream_ref);
template void launch_hash_csr_build_count_kernel<hash_csr_nested_equal, hash_csr_row_hasher>(
  size_type,
  bitmask_type const*,
  size_type*,
  size_type*,
  hash_table_ref,
  hash_csr_nested_equal,
  hash_csr_row_hasher,
  cuda::stream_ref);

#define CUDF_INSTANTIATE_HASH_CSR_PROBE_COUNT(OUTER, EQUAL, HASHER)                            \
  template void launch_hash_csr_probe_count_kernel<OUTER, EQUAL, HASHER>(size_type,            \
                                                                         bitmask_type const*,  \
                                                                         size_type*,           \
                                                                         size_type*,           \
                                                                         cuda::std::uint32_t*, \
                                                                         cuda::std::uint64_t*, \
                                                                         hash_table_ref,       \
                                                                         csr_ref,              \
                                                                         EQUAL,                \
                                                                         HASHER,               \
                                                                         cuda::stream_ref)

CUDF_INSTANTIATE_HASH_CSR_PROBE_COUNT(false, hash_csr_primitive_equal, hash_csr_primitive_hasher);
CUDF_INSTANTIATE_HASH_CSR_PROBE_COUNT(true, hash_csr_primitive_equal, hash_csr_primitive_hasher);
CUDF_INSTANTIATE_HASH_CSR_PROBE_COUNT(false, hash_csr_non_nested_equal, hash_csr_row_hasher);
CUDF_INSTANTIATE_HASH_CSR_PROBE_COUNT(true, hash_csr_non_nested_equal, hash_csr_row_hasher);
CUDF_INSTANTIATE_HASH_CSR_PROBE_COUNT(false, hash_csr_nested_equal, hash_csr_row_hasher);
CUDF_INSTANTIATE_HASH_CSR_PROBE_COUNT(true, hash_csr_nested_equal, hash_csr_row_hasher);

#undef CUDF_INSTANTIATE_HASH_CSR_PROBE_COUNT

}  // namespace cudf::detail
