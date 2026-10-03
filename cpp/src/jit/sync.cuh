/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/types.hpp>

#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/bit>

namespace cudf {

__device__ inline bool warp_elect(unsigned int mask)
{
#if __CUDA_ARCH__ >= 900
  // use elect.sync
  return cuda::ptx::elect_sync(mask);
#else
  // fallback: manually elect a leader (e.g., the first active thread)
  int leader = mask == 0 ? 0 : cuda::std::countr_zero(mask);
  int lane   = (threadIdx.x & (cudf::detail::warp_size - 1));
  return (lane == leader);
#endif
}

namespace jit {

template <typename Out>
__device__ void warp_compact_validity(unsigned int active_mask,
                                      mutable_column_device_view_core const* outcols,
                                      size_type row,
                                      bool is_valid)
{
  auto null_word = __ballot_sync(active_mask, is_valid);
  // use warp-elect to make sure we only issue one memory transaction per warp
  if (warp_elect(active_mask)) {
    Out::set_null_mask_word(outcols, row / cudf::detail::warp_size, null_word);
  }
}

/**
 * @brief Sums `value` across the warp and adds the total to `result` with a single atomic
 *
 * Must be called by every lane of the warp.
 */
__device__ inline void warp_add(size_type value, size_type* result)
{
  for (int offset = cudf::detail::warp_size / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffff'ffffu, value, offset);
  }

  auto lane = threadIdx.x % cudf::detail::warp_size;

  if (lane == 0 && value != 0) {
    cuda::atomic_ref ref(*result);
    ref.fetch_add(value, cuda::std::memory_order_relaxed);
  }
}

}  // namespace jit
}  // namespace cudf
