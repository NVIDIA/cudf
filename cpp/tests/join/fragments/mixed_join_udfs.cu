/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/udf_includes.hpp>

#include <cstdint>

extern "C" __device__ cudf::errc greater_equal_double(bool* output,
                                                      double left_col0,
                                                      double right_col0)
{
  *output = left_col0 >= right_col0;

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc greater_both_int32_t(
  bool* output, int32_t left_col0, int32_t left_col1, int32_t right_col0, int32_t right_col1)
{
  *output = (left_col0 > right_col0) && (left_col1 > right_col1);

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc equal_int32_t(bool* output, int32_t left_col0, int32_t right_col0)
{
  *output = left_col0 == right_col0;

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc xor_popcount(bool* output, int32_t left_val, int32_t right_val)
{
  *output = __popc(left_val ^ right_val) <= 1;

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc checked_greater_equal(bool* output, int a, int b)
{
  if (a < 0) { return cudf::errc::ARITHMETIC_OVERFLOW; }
  if (b == 0) { return cudf::errc::DIVISION_BY_ZERO; }
  *output = a >= b;
  return cudf::errc::SUCCESS;
}

template <typename T>
__device__ cudf::errc greater(bool* output, T left, T right)
{
  *output = left > right;
  return cudf::errc::SUCCESS;
}

template <typename T>
__device__ cudf::errc equal_first(bool* output, T left0, T, T right0, T)
{
  *output = left0 == right0;
  return cudf::errc::SUCCESS;
}

template <typename T>
__device__ cudf::errc greater_second(bool* output, T, T left1, T, T right1)
{
  *output = left1 > right1;
  return cudf::errc::SUCCESS;
}

#define DEFINE_JOIN_COMPARISONS(type)                                                  \
  extern "C" __device__ cudf::errc greater_##type(bool* output, type left, type right) \
  {                                                                                    \
    return greater(output, left, right);                                               \
  }                                                                                    \
  extern "C" __device__ cudf::errc equal_first_##type(                                 \
    bool* output, type left0, type left1, type right0, type right1)                    \
  {                                                                                    \
    return equal_first(output, left0, left1, right0, right1);                          \
  }                                                                                    \
  extern "C" __device__ cudf::errc greater_second_##type(                              \
    bool* output, type left0, type left1, type right0, type right1)                    \
  {                                                                                    \
    return greater_second(output, left0, left1, right0, right1);                       \
  }

DEFINE_JOIN_COMPARISONS(int8_t)
DEFINE_JOIN_COMPARISONS(int16_t)
DEFINE_JOIN_COMPARISONS(int32_t)
DEFINE_JOIN_COMPARISONS(int64_t)
DEFINE_JOIN_COMPARISONS(uint8_t)
DEFINE_JOIN_COMPARISONS(uint16_t)
DEFINE_JOIN_COMPARISONS(uint32_t)
DEFINE_JOIN_COMPARISONS(uint64_t)

#undef DEFINE_JOIN_COMPARISONS
