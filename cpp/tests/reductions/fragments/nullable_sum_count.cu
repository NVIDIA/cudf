/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define cudf_reduce_update cudf_reduce_update_nonnullable
#include "sum_count.cu"
#undef cudf_reduce_update

extern "C" __device__ int cudf_reduce_update(cuda::std::int32_t row,
                                             cuda::std::int64_t* sum,
                                             cuda::std::int32_t* count,
                                             cuda::std::optional<cuda::std::int32_t> value)
{
  return cudf_reduce_update_nonnullable(row, sum, count, value.value_or(100));
}
