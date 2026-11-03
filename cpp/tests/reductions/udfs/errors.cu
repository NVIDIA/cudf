/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

extern "C" {

__device__ errc error_filter(size_type, bool* selected, optional_int32)
{
  *selected = true;
  return errc::ARITHMETIC_OVERFLOW;
}

__device__ errc error_update(size_type, optional_int64*, optional_int64*, optional_int32)
{
  return errc::DIVISION_BY_ZERO;
}

__device__ errc error_init(optional_int64* sum, optional_int64* count)
{
  *sum   = cuda::std::nullopt;
  *count = 0;
  return errc::ARITHMETIC_OVERFLOW;
}

__device__ errc error_merge(optional_int64*, optional_int64*, optional_int64, optional_int64)
{
  return errc::DIVISION_BY_ZERO;
}

__device__ errc error_finalize(
  optional_int64*, optional_int64*, cuda::std::optional<double>*, optional_int64, optional_int64)
{
  return errc::DIVISION_BY_ZERO;
}

}  // extern "C"
