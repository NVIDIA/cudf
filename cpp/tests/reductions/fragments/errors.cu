/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda/std/cstdint>
#include <cuda/std/optional>
#include <cuda/std/span>

extern "C" __device__ int cudf_reduce_init(void* p, cuda::std::int64_t* sum)
{
  if (*static_cast<int*>(p) == 1) { return 1; }
  *sum = 0;
  return 0;
}

extern "C" __device__ int cudf_reduce_update(void* p,
                                             cuda::std::int32_t row,
                                             cuda::std::int64_t* sum,
                                             cuda::std::int32_t value)
{
  if (*static_cast<int*>(p) == 2 && row == 7) { return 2; }
  *sum += value;
  return 0;
}

extern "C" __device__ int cudf_reduce_merge(void* p,
                                            cuda::std::int64_t* sum,
                                            cuda::std::int64_t other)
{
  if (*static_cast<int*>(p) == 3) { return 1; }
  *sum += other;
  return 0;
}

extern "C" __device__ int cudf_reduce_finalize(void* p,
                                               cuda::std::int32_t row,
                                               cuda::std::span<cuda::std::int32_t const>,
                                               cuda::std::optional<cuda::std::int64_t>* out,
                                               cuda::std::int64_t sum)
{
  if (*static_cast<int*>(p) == 4 && row == 1) { return 2; }
  *out = sum;
  return 0;
}
