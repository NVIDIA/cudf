/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/udf_includes.hpp>

#include <cstdint>

namespace cudf::benchmark_udfs {

template <typename T>
__device__ errc minmax(bool* out, T value, T min, T max)
{
  *out = value >= min && value <= max;
  return errc::SUCCESS;
}

extern "C" {

__device__ errc join_predicate_greater(bool* output,
                                       int32_t left_col0,
                                       int32_t left_col1,
                                       int32_t left_col2,
                                       int32_t right_col0,
                                       int32_t right_col1,
                                       int32_t right_col2)
{
  *output = left_col1 > right_col1;
  return errc::SUCCESS;
}

__device__ errc calculate_tpch_q9_price(
  double* amount, double discount, double extended_price, double supply_cost, int8_t quantity)
{
  *amount = extended_price * (1 - discount) - supply_cost * quantity;

  return errc::SUCCESS;
}

#define DEFINE_MINMAX(type)                                                \
  __device__ errc minmax_##type(bool* out, type value, type min, type max) \
  {                                                                        \
    return minmax(out, value, min, max);                                   \
  }

DEFINE_MINMAX(int32_t)
DEFINE_MINMAX(int64_t)
DEFINE_MINMAX(float)
DEFINE_MINMAX(double)

#undef DEFINE_MINMAX

__device__ errc add_f32(float* out, float a, float b)
{
  *out = a + b;
  return errc::SUCCESS;
}

__device__ errc mul_f32(float* out, float a, float b)
{
  *out = a * b;
  return errc::SUCCESS;
}

}  // extern "C"
}  // namespace cudf::benchmark_udfs
