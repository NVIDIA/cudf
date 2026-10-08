/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda/std/cstdint>
#include <cuda/std/optional>
#include <cuda/std/span>

extern "C" __device__ int cudf_reduce_init(cuda::std::int64_t* sum, cuda::std::int32_t* count)
{
  *sum   = 0;
  *count = 0;
  return 0;
}

extern "C" __device__ int cudf_reduce_update(cuda::std::int32_t,
                                             cuda::std::int64_t* sum,
                                             cuda::std::int32_t* count,
                                             cuda::std::int32_t value)
{
  *sum += value;
  ++*count;
  return 0;
}

extern "C" __device__ int cudf_reduce_merge(cuda::std::int64_t* sum,
                                            cuda::std::int32_t* count,
                                            cuda::std::int64_t other_sum,
                                            cuda::std::int32_t other_count)
{
  *sum += other_sum;
  *count += other_count;
  return 0;
}

extern "C" __device__ int cudf_reduce_finalize(cuda::std::int32_t index,
                                               cuda::std::span<cuda::std::int32_t const>,
                                               cuda::std::optional<cuda::std::int64_t>* sum_out,
                                               cuda::std::optional<double>* mean_out,
                                               cuda::std::int64_t sum,
                                               cuda::std::int32_t count)
{
  if (sum_out != nullptr) { *sum_out = sum + index; }
  if (mean_out != nullptr && count != 0) { *mean_out = static_cast<double>(sum) / count; }
  return 0;
}
