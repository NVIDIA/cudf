/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/types.hpp>

#include <cuda/ptx>
#include <cuda/std/bit>

namespace cudf::detail::jit {

__device__ inline bool warp_elect(unsigned int mask)
{
#if __CUDA_ARCH__ >= 900
  return cuda::ptx::elect_sync(mask);
#else
  auto const leader = mask == 0 ? 0 : cuda::std::countr_zero(mask);
  auto const lane   = threadIdx.x & (cudf::detail::warp_size - 1);
  return lane == leader;
#endif
}

template <typename Out>
__device__ void warp_compact_validity(unsigned int active_mask,
                                      mutable_column_device_view_core const* outcols,
                                      size_type row,
                                      bool is_valid)
{
  auto const null_word = __ballot_sync(active_mask, is_valid);
  if (warp_elect(active_mask)) {
    Out::set_null_mask_word(outcols, row / cudf::detail::warp_size, null_word);
  }
}

}  // namespace cudf::detail::jit
