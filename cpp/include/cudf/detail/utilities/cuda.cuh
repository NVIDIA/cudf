/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#ifndef __CUDACC_RTC__

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/cub.cuh>
#include <cuda/std/type_traits>

#endif

namespace cudf {
namespace detail {

/**
 * @brief Size of a warp in a CUDA kernel.
 */
static constexpr int32_t warp_size{32};

#ifndef __CUDACC_RTC__

/**
 * @brief Performs a sum reduction of values from the same lane across all
 * warps in a thread block and returns the result on thread 0 of the block.
 *
 * All threads in a block must call this function, but only values from the
 * threads indicated by `leader_lane` will contribute to the result. Similarly,
 * the returned result is only defined on `threadIdx.x==0`.
 *
 * @tparam block_size The number of threads in the thread block (must be less
 * than or equal to 1024)
 * @tparam leader_lane The id of the lane in the warp whose value contributes to
 * the reduction
 * @tparam T Arithmetic type
 * @param lane_value The value from the lane that contributes to the reduction
 * @return The sum reduction of the values from each lane. Only valid on
 * `threadIdx.x == 0`. The returned value on all other threads is undefined.
 */
template <int32_t block_size, int32_t leader_lane = 0, typename T>
__device__ T single_lane_block_sum_reduce(T lane_value)
{
  static_assert(block_size <= 1024, "Invalid block size.");
  static_assert(cuda::std::is_arithmetic_v<T>, "Invalid non-arithmetic type.");
  constexpr auto warps_per_block{block_size / warp_size};
  auto const lane_id{threadIdx.x % warp_size};
  auto const warp_id{threadIdx.x / warp_size};
  __shared__ T lane_values[warp_size];

  // Load each lane's value into a shared memory array
  if (lane_id == leader_lane) { lane_values[warp_id] = lane_value; }
  __syncthreads();

  // Use a single warp to do the reduction, result is only defined on
  // threadId.x == 0
  T result{0};
  if (warp_id == 0) {
    __shared__ typename cub::WarpReduce<T>::TempStorage temp;
    lane_value = (lane_id < warps_per_block) ? lane_values[lane_id] : T{0};
    result     = cub::WarpReduce<T>(temp).Sum(lane_value);
  }
  // Shared memory has block scope, so sync here to ensure no data
  // races between successive calls to this function in the same
  // kernel.
  __syncthreads();
  return result;
}

template <class F>
CUDF_KERNEL void single_thread_kernel(F f)
{
  f();
}

/**
 * @brief single thread cuda kernel
 *
 * @tparam Functor Device functor type
 * @param functor device functor object or device lambda function
 * @param stream CUDA stream used for the kernel launch
 */
template <class Functor>
void device_single_thread(Functor functor, rmm::cuda_stream_view stream)
{
  single_thread_kernel<<<1, 1, 0, stream.value()>>>(functor);
}

#endif

}  // namespace detail
}  // namespace cudf
