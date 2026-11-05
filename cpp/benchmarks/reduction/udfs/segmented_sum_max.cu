/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "common.cuh"

using cudf::size_type;
using tpc_reduction::optional_int64;

extern "C" {
__device__ cudf::errc segmented_sum_init(size_type, optional_int64* sum)
{
  *sum = cuda::std::nullopt;
  return cudf::errc::SUCCESS;
}

__device__ cudf::errc segmented_sum_update(size_type,
                                           size_type,
                                           optional_int64* sum,
                                           optional_int64 value)
{
  if (value) { *sum = sum->value_or(0) + *value; }
  return cudf::errc::SUCCESS;
}

__device__ cudf::errc segmented_sum_merge(size_type, optional_int64* sum, optional_int64 other)
{
  if (other) { *sum = sum->value_or(0) + *other; }
  return cudf::errc::SUCCESS;
}

__device__ cudf::errc segmented_sum_finalize(size_type, optional_int64* output, optional_int64 sum)
{
  *output = sum;
  return cudf::errc::SUCCESS;
}

__device__ cudf::errc segmented_sum_max_init(size_type segment,
                                             optional_int64* sum,
                                             optional_int64* maximum)
{
  *maximum = cuda::std::nullopt;
  return segmented_sum_init(segment, sum);
}

__device__ cudf::errc segmented_sum_max_update(size_type segment,
                                               size_type row,
                                               optional_int64* sum,
                                               optional_int64* maximum,
                                               optional_int64 value)
{
  if (value && (!*maximum || **maximum < *value)) { *maximum = value; }
  return segmented_sum_update(segment, row, sum, value);
}

__device__ cudf::errc segmented_sum_max_merge(size_type segment,
                                              optional_int64* sum,
                                              optional_int64* maximum,
                                              optional_int64 other_sum,
                                              optional_int64 other_maximum)
{
  if (other_maximum && (!*maximum || **maximum < *other_maximum)) { *maximum = other_maximum; }
  return segmented_sum_merge(segment, sum, other_sum);
}

__device__ cudf::errc segmented_sum_max_finalize(size_type,
                                                 optional_int64* sum_output,
                                                 optional_int64* max_output,
                                                 optional_int64 sum,
                                                 optional_int64 maximum)
{
  *sum_output = sum;
  *max_output = maximum;
  return cudf::errc::SUCCESS;
}
}
