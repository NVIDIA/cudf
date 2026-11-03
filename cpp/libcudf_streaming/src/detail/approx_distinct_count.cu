/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_streaming/detail/approx_distinct_count.hpp>

#include <cuda_runtime_api.h>

#include <rapidsmpf/error.hpp>

namespace cudf_streaming::detail {
namespace {

__global__ void set_value_kernel(std::uint64_t* data, std::uint64_t value)
{
  if (threadIdx.x == 0) { data[0] = value; }
}

__global__ void add_values_kernel(std::uint64_t const* left, std::uint64_t* right)
{
  if (threadIdx.x == 0) { right[0] += left[0]; }
}

}  // namespace

void set_value(std::uint64_t* data, std::uint64_t value, cuda::stream_ref stream)
{
  set_value_kernel<<<1, 1, 0, stream.get()>>>(data, value);
  RAPIDSMPF_CUDA_TRY(cudaPeekAtLastError());
}

void add_values(std::uint64_t const* left, std::uint64_t* right, cuda::stream_ref stream)
{
  add_values_kernel<<<1, 1, 0, stream.get()>>>(left, right);
  RAPIDSMPF_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cudf_streaming::detail
