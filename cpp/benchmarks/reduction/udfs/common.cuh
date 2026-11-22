/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/optional>

namespace tpc_reduction {
// Inline arithmetic helpers shared by independently compiled query fragments.
using cuda::std::int32_t;
using optional_double = cuda::std::optional<double>;
using optional_int32  = cuda::std::optional<cuda::std::int32_t>;
using optional_int64  = cuda::std::optional<cuda::std::int64_t>;

__device__ inline void add_value(optional_double* sum, optional_int64* count, optional_double value)
{
  if (value) {
    **sum += *value;
    ++**count;
  }
}

__device__ inline cudf::errc init_sum_count(optional_double* sum, optional_int64* count)
{
  *sum   = 0;
  *count = 0;
  return cudf::errc::SUCCESS;
}

__device__ inline cudf::errc merge_sum_count(optional_double* sum,
                                             optional_int64* count,
                                             optional_double other_sum,
                                             optional_int64 other_count)
{
  **sum += *other_sum;
  **count += *other_count;
  return cudf::errc::SUCCESS;
}

__device__ inline cudf::errc finalize_sum(optional_double* output,
                                          optional_double sum,
                                          optional_int64 count)
{
  *output = *count == 0 ? cuda::std::nullopt : sum;
  return cudf::errc::SUCCESS;
}

__device__ inline cudf::errc finalize_mean(optional_double* output,
                                           optional_double sum,
                                           optional_int64 count)
{
  *output = cuda::std::nullopt;
  if (*count != 0) { *output = *sum / *count; }
  return cudf::errc::SUCCESS;
}

}  // namespace tpc_reduction
