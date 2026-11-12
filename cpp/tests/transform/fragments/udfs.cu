/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/operators/operators.cuh>
#include <cudf/udf_includes.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/chrono>
#include <cuda/std/cmath>

#include <cstdint>

extern "C" __device__ cudf::errc add_day(int64_t* output, int64_t input)
{
  using dur = cuda::std::chrono::duration<int32_t, cuda::std::ratio<86400>>;
  *output   = (cudf::timestamp_us{cudf::duration_us{input}} + dur{1}).time_since_epoch().count();

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc decode_string(cudf::string_view* output, cudf::string_view input)
{
  *output = input;

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc compare(bool* out,
                                         cudf::string_view a,
                                         cudf::string_view b,
                                         cudf::string_view c)
{
  *out = (a == b) || (b == c);

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc string_contains(bool* out,
                                                 cudf::string_view a,
                                                 cudf::string_view b)
{
  *out = a.find(b) != cudf::string_view::npos;

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc string_find(int32_t* out, cudf::string_view a, cudf::string_view b)
{
  *out = b.find(a);

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc compare_string_and_integer(
  bool* out, cudf::string_view a, cudf::string_view b, int32_t c, int32_t d)
{
  *out = (a == b) && (c == d);

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc max_string(cudf::string_view* out,
                                            cudf::string_view a,
                                            cudf::string_view b,
                                            cudf::string_view c,
                                            cudf::string_view d)
{
  *out = cuda::std::max(cuda::std::max(cuda::std::max(a, b), c), d);

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc nullable_string_identity(
  cuda::std::optional<cudf::string_view>* output, cuda::std::optional<cudf::string_view> input)
{
  *output = input;

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc concat(cuda::std::span<char>* out,
                                        cudf::string_view a,
                                        cudf::string_view b,
                                        cudf::string_view c)
{
  auto iter = out->data();
  memcpy(iter, a.data(), a.size_bytes());
  iter += a.size_bytes();
  memcpy(iter, b.data(), b.size_bytes());
  iter += b.size_bytes();
  memcpy(iter, c.data(), c.size_bytes());

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc concat_with_metadata(int32_t* ab_size,
                                                      cuda::std::span<char>* out,
                                                      bool* is_odd,
                                                      cudf::string_view a,
                                                      cudf::string_view b,
                                                      cudf::string_view c)
{
  auto iter  = out->data();
  auto begin = iter;
  memcpy(iter, a.data(), a.size_bytes());
  iter += a.size_bytes();
  memcpy(iter, b.data(), b.size_bytes());
  iter += b.size_bytes();
  memcpy(iter, c.data(), c.size_bytes());
  iter += c.size_bytes();
  auto out_size = static_cast<int32_t>(iter - begin);
  *is_odd       = (out_size % 2) == 1;
  *ab_size      = static_cast<int32_t>(a.size_bytes() + b.size_bytes());

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc format_name(cudf::size_type row,
                                             cudf::string_view* out,
                                             void* user_data,
                                             cudf::string_view first_name,
                                             cudf::string_view last_name,
                                             int32_t size)
{
  char* it = static_cast<char*>(user_data) + static_cast<size_t>(row) * static_cast<size_t>(size);
  char const* const begin = it;

  memcpy(it, first_name.data(), first_name.size_bytes());
  it += first_name.size_bytes();

  memcpy(it, " ", 1);
  it += 1;

  memcpy(it, last_name.data(), last_name.size_bytes());
  it += last_name.size_bytes();

  *out = cudf::string_view{begin, static_cast<cudf::size_type>(it - begin)};

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc matching_string(cudf::string_view* out,
                                                 cudf::string_view a,
                                                 cudf::string_view b)
{
  *out = (a == b ? a : cudf::string_view{});

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc capture_group(cudf::string_view* area_code,
                                               cudf::string_view* prefix,
                                               cudf::string_view* line_number,
                                               cudf::string_view input)
{
  auto seek = [](char const*& iter, char const* end, char c) {
    while (iter != end && *iter != c) {
      iter++;
    }
  };
  auto* iter            = input.data();
  auto* end             = iter + input.size_bytes();
  auto* area_code_begin = iter;
  seek(iter, end, ' ');
  auto* area_code_end = iter;
  iter++;
  auto* prefix_begin = iter;
  seek(iter, end, ' ');
  auto* prefix_end = iter;
  iter++;
  auto* line_number_begin = iter;
  seek(iter, end, ' ');
  auto* line_number_end = iter;
  *area_code            = cudf::string_view{area_code_begin,
                                 static_cast<cudf::size_type>(area_code_end - area_code_begin)};
  *prefix =
    cudf::string_view{prefix_begin, static_cast<cudf::size_type>(prefix_end - prefix_begin)};
  *line_number = cudf::string_view{
    line_number_begin, static_cast<cudf::size_type>(line_number_end - line_number_begin)};

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc is_null(cuda::std::optional<bool>* output,
                                         cuda::std::optional<float> input)
{
  *output = !input.has_value();

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc null_lerp(cuda::std::optional<float>* output,
                                           cuda::std::optional<float> low,
                                           cuda::std::optional<float> high,
                                           cuda::std::optional<float> t)
{
  auto lerp = [](auto l, auto h, auto t) { return l - t * l + t * h; };
  *output   = low.has_value() && high.has_value() && t.has_value() ? lerp(*low, *high, *t) : 0.0F;

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc divide_checked(int* c, int a, int b)
{
  auto e = cudf::detail::ops::div_overflow(a, b);
  if (!e.has_value()) { return e.error(); }
  *c = e.value();
  return cudf::errc::SUCCESS;
}

template <typename Decimal>
__device__ cudf::errc decimal_arithmetic(Decimal* out, Decimal a, Decimal b, Decimal c)
{
  *out = (a + b) * c;
  return cudf::errc::SUCCESS;
}

#define DEFINE_DECIMAL_ARITHMETIC(bits)                                                   \
  extern "C" __device__ cudf::errc decimal_arithmetic_##bits(numeric::decimal##bits* out, \
                                                             numeric::decimal##bits a,    \
                                                             numeric::decimal##bits b,    \
                                                             numeric::decimal##bits c)    \
  {                                                                                       \
    return decimal_arithmetic(out, a, b, c);                                              \
  }

DEFINE_DECIMAL_ARITHMETIC(32)
DEFINE_DECIMAL_ARITHMETIC(64)
DEFINE_DECIMAL_ARITHMETIC(128)

#undef DEFINE_DECIMAL_ARITHMETIC

extern "C" __device__ cudf::errc invsqrt(float* out, float a)
{
  *out = 1.0F / sqrtf(a);
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc distance(float* out, float x1, float y1, float x2, float y2)
{
  *out = cuda::std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc to_upper(uint8_t* output, uint8_t input)
{
  *output = input > 96 && input < 123 ? input - 32 : input;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc sum_of_squares(float* out, float a, float b)
{
  *out = a * a + b * b;
  return cudf::errc::SUCCESS;
}

/**
 * @brief Calculates the integer Lehmer mean of two integers using checked arithmetic.
 * The Lehmer mean is defined as `(a^2 + b^2) / (a + b)`.
 * @param out The output integer.
 * @param a The first integer.
 * @param b The second integer.
 * @return Success, or an error code if checked arithmetic fails.
 *
 */
extern "C" __device__ cudf::errc lehmer_mean(int32_t* out, int32_t a, int32_t b)
{
  auto a2 = cudf::detail::ops::mul_overflow(a, a);
  if (!a2) return a2.error();

  auto b2 = cudf::detail::ops::mul_overflow(b, b);
  if (!b2) return b2.error();

  auto a_b_sum = cudf::detail::ops::add_overflow(a, b);
  if (!a_b_sum) return a_b_sum.error();

  auto a2_b2_sum = cudf::detail::ops::add_overflow(a2.value(), b2.value());
  if (!a2_b2_sum) return a2_b2_sum.error();

  auto result = cudf::detail::ops::div_overflow(a2_b2_sum.value(), a_b_sum.value());
  if (!result) return result.error();
  *out = result.value();
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc bankers_round(numeric::decimal128* out, numeric::decimal128 x)
{
  using Rep = numeric::decimal128::rep;

  auto const scale = x.scale();
  auto const value = x.value();

  if (scale >= numeric::scale_type{0}) {
    *out = x;
    return cudf::errc::SUCCESS;
  }

  Rep factor = 1;
  for (int32_t i = 0; i < -static_cast<int32_t>(scale); ++i) {
    factor *= 10;
  }

  Rep q = value / factor;
  Rep r = value % factor;

  if (r < 0) r = -r;

  auto const half = factor / 2;

  bool round_up = false;

  if (r > half) {
    round_up = true;
  } else if (r == half) {
    // tie: round to even
    round_up = (q % 2) != 0;
  }

  if (round_up) { q += value >= 0 ? Rep{1} : Rep{-1}; }

  *out = numeric::decimal128{q, numeric::scale_type{0}};
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc fourth_power(float* C, float a)
{
  *C = a * a * a * a;

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc mulsub(int* output, int input)
{
  *output = input * input - input;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc ascii_upper(signed char* output, signed char input)
{
  if (input > 96 && input < 123) {
    *output = input - 32;
  } else {
    *output = input;
  }

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc decode_float(float* output, float input)
{
  *output = input;

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc addmul(float* out, float a, float b, float c)
{
  *out = (a + b) * c;

  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lerp(float* out, float a, float b, float t)
{
  *out = a - t * a + t * b;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc add_f32(float* out, float a, float b)
{
  *out = a * a * a + b;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc add_i32(int64_t* out, int32_t a, int32_t b)
{
  auto const value = static_cast<int64_t>(a);
  *out             = value * value * value + b;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc add_i32_i64(int64_t* out, int32_t a, int64_t b)
{
  auto const value = static_cast<int64_t>(a);
  *out             = value * value * value + b * b;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc division_by_zero(int32_t*, int32_t, int32_t)
{
  return cudf::errc::DIVISION_BY_ZERO;
}
