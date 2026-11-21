// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cudf/strings/detail/gather.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/cstring>
#include <cuda/std/memory>
namespace cudf::detail {
struct partition_string_column {
  size_type const* input_offsets;
  char const* input_chars;
  size_type* output_offsets;
  char* output_chars;
};
namespace partition_strings_plan {
// All addresses loaded by load_uint4 stay within this string, including its
// unaligned loads. Destination stores are aligned and the ends use byte copies.
template <int Lanes>
__device__ void copy_vector(
  partition_string_column col, size_type source, size_type target, size_type length, int lane)
{
  int64_t const first = (int64_t{target} + 4 + 15) / 16 * 16;
  int64_t const end   = (int64_t{target} + length - 4) / 16 * 16;
  for (auto byte = first + lane * 16; byte < end; byte += Lanes * 16) {
    auto const value = strings::detail::load_uint4(col.input_chars + source + byte - target);
    cuda::std::memcpy(cuda::std::assume_aligned<16>(col.output_chars + byte), &value, 16);
  }
  for (size_type byte = lane; int64_t{target} + byte < first; byte += Lanes) {
    col.output_chars[int64_t{target} + byte] = col.input_chars[int64_t{source} + byte];
  }
  for (auto byte = end + lane; byte < int64_t{target} + length; byte += Lanes) {
    col.output_chars[byte] = col.input_chars[int64_t{source} + byte - target];
  }
}

/** All warp lanes participate, including groups beyond the last row. */
template <bool RebalanceMedium = false>
__device__ inline void copy_group(partition_string_column col,
                                  cuda::std::size_t source_row,
                                  size_type destination,
                                  size_type rows)
{
  constexpr int lanes = 4;
  auto const lane     = threadIdx.x % lanes;
  size_type begin = 0, length = 0, target = 0;
  if (source_row < static_cast<cuda::std::size_t>(rows)) {
    begin  = col.input_offsets[source_row];
    length = col.input_offsets[source_row + 1] - begin;
    target = col.output_offsets[destination];
  }
  // This device-side split balances individual strings. It does not change the
  // CPU-selected partition method or require a size/skew sampling pass.
  auto long_rows = __ballot_sync(0xffff'ffffu, lane == 0 && length > 128);
  if constexpr (RebalanceMedium) {
    auto medium_rows = __ballot_sync(0xffff'ffffu, lane == 0 && length > 8 && length <= 128);
    if (length <= 8) {
      for (size_type byte = lane; byte < length; byte += lanes) {
        col.output_chars[int64_t{target} + byte] = col.input_chars[int64_t{begin} + byte];
      }
    }
    // Two 16-lane groups copy medium strings; every lane also participates
    // when the last queue entry occupies only one half warp.
    auto const warp_lane = threadIdx.x % 32;
    while (medium_rows) {
      auto const first = __ffs(medium_rows) - 1;
      medium_rows &= medium_rows - 1;
      bool const has_second = medium_rows != 0;
      auto const second     = has_second ? __ffs(medium_rows) - 1 : first;
      if (has_second) { medium_rows &= medium_rows - 1; }
      auto const owner  = warp_lane < 16 ? first : second;
      auto const source = __shfl_sync(0xffff'ffffu, begin, owner);
      auto const output = __shfl_sync(0xffff'ffffu, target, owner);
      auto count        = __shfl_sync(0xffff'ffffu, length, owner);
      if (warp_lane >= 16 && !has_second) { count = 0; }
      for (size_type byte = warp_lane % 16; byte < count; byte += 16) {
        col.output_chars[int64_t{output} + byte] = col.input_chars[int64_t{source} + byte];
      }
    }
  } else {
    if (length <= 128) {
      for (size_type byte = lane; byte < length; byte += lanes) {
        col.output_chars[int64_t{target} + byte] = col.input_chars[int64_t{begin} + byte];
      }
    }
  }
  while (long_rows) {
    auto const leader = __ffs(long_rows) - 1;
    auto const source = __shfl_sync(0xffff'ffffu, begin, leader);
    auto const output = __shfl_sync(0xffff'ffffu, target, leader);
    auto const count  = __shfl_sync(0xffff'ffffu, length, leader);
    copy_vector<32>(col, source, output, count, threadIdx.x % 32);
    long_rows &= long_rows - 1;
  }
}

template <bool RebalanceMedium = false>
CUDF_KERNEL void copy_characters(partition_string_column const* columns,
                                 size_type const* scatter_map,
                                 size_type rows)
{
  auto const col = columns[blockIdx.y];
  for (auto base = cuda::std::size_t{blockIdx.x} * 64; base < static_cast<cuda::std::size_t>(rows);
       base += cuda::std::size_t{gridDim.x} * 64) {
    auto const source = base + threadIdx.x / 4;
    auto const target = source < static_cast<cuda::std::size_t>(rows) ? scatter_map[source] : 0;
    copy_group<RebalanceMedium>(col, source, target, rows);
  }
}

}  // namespace partition_strings_plan
}  // namespace cudf::detail
