/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/udf_includes.hpp>

template <typename T>
__device__ cudf::errc is_equal(bool* out, T a, T b)
{
  *out = (a == b);
  return cudf::errc::SUCCESS;
}

#define DEFINE_EQUALITY_UDF(name, type)                                       \
  extern "C" __device__ cudf::errc is_equal_##name(bool* out, type a, type b) \
  {                                                                           \
    return is_equal(out, a, b);                                               \
  }

DEFINE_EQUALITY_UDF(int8_t, int8_t)
DEFINE_EQUALITY_UDF(int16_t, int16_t)
DEFINE_EQUALITY_UDF(int32_t, int32_t)
DEFINE_EQUALITY_UDF(int64_t, int64_t)
DEFINE_EQUALITY_UDF(uint8_t, uint8_t)
DEFINE_EQUALITY_UDF(uint16_t, uint16_t)
DEFINE_EQUALITY_UDF(uint32_t, uint32_t)
DEFINE_EQUALITY_UDF(uint64_t, uint64_t)
DEFINE_EQUALITY_UDF(float, float)
DEFINE_EQUALITY_UDF(double, double)
DEFINE_EQUALITY_UDF(timestamp_D, cudf::timestamp_D)
DEFINE_EQUALITY_UDF(timestamp_s, cudf::timestamp_s)
DEFINE_EQUALITY_UDF(timestamp_ms, cudf::timestamp_ms)
DEFINE_EQUALITY_UDF(timestamp_us, cudf::timestamp_us)
DEFINE_EQUALITY_UDF(timestamp_ns, cudf::timestamp_ns)
DEFINE_EQUALITY_UDF(duration_D, cudf::duration_D)
DEFINE_EQUALITY_UDF(duration_s, cudf::duration_s)
DEFINE_EQUALITY_UDF(duration_ms, cudf::duration_ms)
DEFINE_EQUALITY_UDF(duration_us, cudf::duration_us)
DEFINE_EQUALITY_UDF(duration_ns, cudf::duration_ns)
DEFINE_EQUALITY_UDF(string_view, cudf::string_view)
DEFINE_EQUALITY_UDF(decimal32, numeric::decimal32)
DEFINE_EQUALITY_UDF(decimal64, numeric::decimal64)
DEFINE_EQUALITY_UDF(decimal128, numeric::decimal128)

#undef DEFINE_EQUALITY_UDF

extern "C" __device__ cudf::errc is_even(bool* out, int32_t a)
{
  *out = (a % 2 == 0);
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc nullable_is_even(cuda::std::optional<bool>* out,
                                                  cuda::std::optional<int32_t> a)
{
  *out = a.has_value() && (*a % 2 == 0);
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc is_divisible(bool* out, int32_t a, int32_t b)
{
  *out = ((a % b) == 0);
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc filter_climate(bool* out,
                                                [[maybe_unused]] cudf::string_view country,
                                                cudf::string_view tz,
                                                float tmp,
                                                float hum,
                                                float min_tmp,
                                                float max_tmp,
                                                float min_hum,
                                                float max_hum,
                                                cudf::string_view tz1,
                                                cudf::string_view tz2)
{
  *out = (tmp >= min_tmp && tmp <= max_tmp) && (hum >= min_hum && hum <= max_hum) &&
         (tz == tz1 || tz == tz2);

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc less_than_ten(bool* out, int32_t value)
{
  *out = value < 10;
  return cudf::errc::SUCCESS;
}
