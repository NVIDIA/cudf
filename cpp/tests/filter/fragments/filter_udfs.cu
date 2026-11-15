/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/udf_includes.hpp>

namespace cudf::test_udfs {

using numeric::decimal128;
using numeric::decimal32;
using numeric::decimal64;

template <typename T>
__device__ errc is_equal(bool* out, T a, T b)
{
  *out = (a == b);
  return errc::SUCCESS;
}

extern "C" {

#define DEFINE_EQUALITY_UDF(type) \
  __device__ errc is_equal_##type(bool* out, type a, type b) { return is_equal(out, a, b); }

DEFINE_EQUALITY_UDF(int8_t)
DEFINE_EQUALITY_UDF(int16_t)
DEFINE_EQUALITY_UDF(int32_t)
DEFINE_EQUALITY_UDF(int64_t)
DEFINE_EQUALITY_UDF(uint8_t)
DEFINE_EQUALITY_UDF(uint16_t)
DEFINE_EQUALITY_UDF(uint32_t)
DEFINE_EQUALITY_UDF(uint64_t)
DEFINE_EQUALITY_UDF(float)
DEFINE_EQUALITY_UDF(double)
DEFINE_EQUALITY_UDF(timestamp_D)
DEFINE_EQUALITY_UDF(timestamp_s)
DEFINE_EQUALITY_UDF(timestamp_ms)
DEFINE_EQUALITY_UDF(timestamp_us)
DEFINE_EQUALITY_UDF(timestamp_ns)
DEFINE_EQUALITY_UDF(duration_D)
DEFINE_EQUALITY_UDF(duration_s)
DEFINE_EQUALITY_UDF(duration_ms)
DEFINE_EQUALITY_UDF(duration_us)
DEFINE_EQUALITY_UDF(duration_ns)
DEFINE_EQUALITY_UDF(string_view)
DEFINE_EQUALITY_UDF(decimal32)
DEFINE_EQUALITY_UDF(decimal64)
DEFINE_EQUALITY_UDF(decimal128)

#undef DEFINE_EQUALITY_UDF

__device__ errc is_even(bool* out, int32_t a)
{
  *out = (a % 2 == 0);
  return errc::SUCCESS;
}

__device__ errc nullable_is_even(cuda::std::optional<bool>* out, cuda::std::optional<int32_t> a)
{
  *out = a.has_value() && (*a % 2 == 0);
  return errc::SUCCESS;
}

__device__ errc is_divisible(bool* out, int32_t a, int32_t b)
{
  *out = ((a % b) == 0);
  return errc::SUCCESS;
}

__device__ errc filter_climate(bool* out,
                               [[maybe_unused]] string_view country,
                               string_view tz,
                               float tmp,
                               float hum,
                               float min_tmp,
                               float max_tmp,
                               float min_hum,
                               float max_hum,
                               string_view tz1,
                               string_view tz2)
{
  *out = (tmp >= min_tmp && tmp <= max_tmp) && (hum >= min_hum && hum <= max_hum) &&
         (tz == tz1 || tz == tz2);

  return errc::SUCCESS;
}

__device__ errc less_than_ten(bool* out, int32_t value)
{
  *out = value < 10;
  return errc::SUCCESS;
}

}  // extern "C"
}  // namespace cudf::test_udfs
