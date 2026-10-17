/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/types.hpp>

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

__device__ inline void warp_compact_validity(unsigned int active_mask,
                                             bitmask_type* null_mask,
                                             size_type row,
                                             bool is_valid)
{
  auto null_word = __ballot_sync(active_mask, is_valid);
  // use warp-elect to make sure we only issue one memory transaction per warp
  if (null_mask != nullptr && warp_elect(active_mask)) {
    null_mask[row / cudf::detail::warp_size] = null_word;
  }
}

template <typename Out, typename Column>
__device__ void warp_compact_validity(unsigned int active_mask,
                                      Column const* outcols,
                                      size_type row,
                                      bool is_valid)
{
  warp_compact_validity(active_mask, Out::column(outcols).null_mask(), row, is_valid);
}

}  // namespace jit
}  // namespace cudf
