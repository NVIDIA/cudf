/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

extern "C" {

__device__ errc discount_update(size_type,
                                optional_int64* sum,
                                optional_int64* count,
                                optional_int32 amount,
                                optional_int32 discount,
                                optional_int32 rate)
{
  if (amount && discount && rate) {
    *sum   = sum->value_or(0) + (int64_t{*amount} - *discount) * *rate;
    *count = count->value() + 1;
  }
  return errc::SUCCESS;
}

__device__ errc user_data_update(
  size_type row, optional_int64* sum, optional_int64* count, optional_int32 value, void* user_data)
{
  auto const* weights = static_cast<int64_t const*>(user_data);
  if (value) {
    *sum   = sum->value_or(0) + *value * weights[0] + weights[row + 1];
    *count = count->value() + 1;
  }
  return errc::SUCCESS;
}

__device__ errc fused_finalize(optional_int64* out_sum,
                               optional_int64* out_count,
                               cuda::std::optional<double>* out_average,
                               cuda::std::optional<bool>* out_qualifies,
                               optional_int64 sum,
                               optional_int64 count)
{
  *out_qualifies = cuda::std::nullopt;
  if (sum) { *out_qualifies = *sum >= 100; }
  return tpch_q6_sales_finalize(out_sum, out_count, out_average, sum, count);
}

}  // extern "C"

extern "C" {
__device__ errc empty_init() { return errc::SUCCESS; }
__device__ errc empty_update(size_type, int32_t) { return errc::SUCCESS; }
__device__ errc empty_merge() { return errc::SUCCESS; }
__device__ errc empty_finalize() { return errc::SUCCESS; }
__device__ errc empty_finalize_error() { return errc::DIVISION_BY_ZERO; }
}
