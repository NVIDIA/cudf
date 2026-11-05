/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

extern "C" {

__device__ errc sum_update(size_type,
                           optional_int64* sum,
                           optional_int64* count,
                           optional_int32 value)
{
  if (value) {
    *sum   = sum->value_or(0) + *value;
    *count = count->value() + 1;
  }
  return errc::SUCCESS;
}

__device__ errc sum_segment_update(
  size_type, size_type row, optional_int64* sum, optional_int64* count, optional_int32 value)
{
  return sum_update(row, sum, count, value);
}

// Value-only callbacks implement COALESCE(SUM(value), 0), skipping null input rows.
__device__ errc raw_init(int64_t* sum)
{
  *sum = 0;
  return errc::SUCCESS;
}

__device__ errc raw_update(size_type, int64_t* sum, int32_t value)
{
  *sum += value;
  return errc::SUCCESS;
}

__device__ errc raw_merge(int64_t* sum, int64_t other)
{
  *sum += other;
  return errc::SUCCESS;
}

__device__ errc raw_finalize(int64_t* output, int64_t sum)
{
  *output = sum;
  return errc::SUCCESS;
}

}  // extern "C"
