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

namespace cudf::test_udfs {

template <typename Decimal>
__device__ errc decimal_arithmetic(Decimal* out, Decimal a, Decimal b, Decimal c)
{
  *out = (a + b) * c;
  return errc::SUCCESS;
}

extern "C" {

__device__ errc add_day(int64_t* output, int64_t input)
{
  using dur = cuda::std::chrono::duration<int32_t, cuda::std::ratio<86400>>;
  *output   = (timestamp_us{duration_us{input}} + dur{1}).time_since_epoch().count();

  return errc::SUCCESS;
}

__device__ errc decode_string(string_view* output, string_view input)
{
  *output = input;

  return errc::SUCCESS;
}

__device__ errc compare(bool* out, string_view a, string_view b, string_view c)
{
  *out = (a == b) || (b == c);

  return errc::SUCCESS;
}

__device__ errc string_contains(bool* out, string_view a, string_view b)
{
  *out = a.find(b) != string_view::npos;

  return errc::SUCCESS;
}

__device__ errc string_find(int32_t* out, string_view a, string_view b)
{
  *out = b.find(a);

  return errc::SUCCESS;
}

__device__ errc copy_string(mutable_string_view* output, string_view input)
{
  for (size_type i = 0; i < input.size_bytes(); ++i) {
    output->data()[i] = input.data()[i];
  }
  return errc::SUCCESS;
}

__device__ errc
compare_string_and_integer(bool* out, string_view a, string_view b, int32_t c, int32_t d)
{
  *out = (a == b) && (c == d);

  return errc::SUCCESS;
}

__device__ errc
max_string(string_view* out, string_view a, string_view b, string_view c, string_view d)
{
  *out = cuda::std::max(cuda::std::max(cuda::std::max(a, b), c), d);

  return errc::SUCCESS;
}

__device__ errc nullable_string_identity(cuda::std::optional<string_view>* output,
                                         cuda::std::optional<string_view> input)
{
  *output = input;

  return errc::SUCCESS;
}

__device__ errc concat(mutable_string_view* out, string_view a, string_view b, string_view c)
{
  auto iter = out->data();
  memcpy(iter, a.data(), a.size_bytes());
  iter += a.size_bytes();
  memcpy(iter, b.data(), b.size_bytes());
  iter += b.size_bytes();
  memcpy(iter, c.data(), c.size_bytes());

  return errc::SUCCESS;
}

__device__ errc concat_with_metadata(int32_t* ab_size,
                                     mutable_string_view* out,
                                     bool* is_odd,
                                     string_view a,
                                     string_view b,
                                     string_view c)
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

  return errc::SUCCESS;
}

__device__ errc format_name(size_type row,
                            string_view* out,
                            void* user_data,
                            string_view first_name,
                            string_view last_name,
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

  *out = string_view{begin, static_cast<size_type>(it - begin)};

  return errc::SUCCESS;
}

__device__ errc matching_string(string_view* out, string_view a, string_view b)
{
  *out = (a == b ? a : string_view{});

  return errc::SUCCESS;
}

__device__ errc capture_group(string_view* area_code,
                              string_view* prefix,
                              string_view* line_number,
                              string_view input)
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
  *area_code =
    string_view{area_code_begin, static_cast<size_type>(area_code_end - area_code_begin)};
  *prefix = string_view{prefix_begin, static_cast<size_type>(prefix_end - prefix_begin)};
  *line_number =
    string_view{line_number_begin, static_cast<size_type>(line_number_end - line_number_begin)};

  return errc::SUCCESS;
}

__device__ errc is_null(cuda::std::optional<bool>* output, cuda::std::optional<float> input)
{
  *output = !input.has_value();

  return errc::SUCCESS;
}

__device__ errc null_lerp(cuda::std::optional<float>* output,
                          cuda::std::optional<float> low,
                          cuda::std::optional<float> high,
                          cuda::std::optional<float> t)
{
  auto lerp = [](auto l, auto h, auto t) { return l - t * l + t * h; };
  *output   = low.has_value() && high.has_value() && t.has_value() ? lerp(*low, *high, *t) : 0.0F;

  return errc::SUCCESS;
}

__device__ errc divide_checked(int* c, int a, int b)
{
  auto e = detail::ops::div_overflow(a, b);
  if (!e.has_value()) { return e.error(); }
  *c = e.value();
  return errc::SUCCESS;
}

#define DEFINE_DECIMAL_ARITHMETIC(bits)                                  \
  __device__ errc decimal_arithmetic_##bits(numeric::decimal##bits* out, \
                                            numeric::decimal##bits a,    \
                                            numeric::decimal##bits b,    \
                                            numeric::decimal##bits c)    \
  {                                                                      \
    return decimal_arithmetic(out, a, b, c);                             \
  }

DEFINE_DECIMAL_ARITHMETIC(32)
DEFINE_DECIMAL_ARITHMETIC(64)
DEFINE_DECIMAL_ARITHMETIC(128)

#undef DEFINE_DECIMAL_ARITHMETIC

__device__ errc invsqrt(float* out, float a)
{
  *out = 1.0F / sqrtf(a);
  return errc::SUCCESS;
}

__device__ errc distance(float* out, float x1, float y1, float x2, float y2)
{
  *out = cuda::std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
  return errc::SUCCESS;
}

__device__ errc to_upper(uint8_t* output, uint8_t input)
{
  *output = input > 96 && input < 123 ? input - 32 : input;
  return errc::SUCCESS;
}

__device__ errc sum_of_squares(float* out, float a, float b)
{
  *out = a * a + b * b;
  return errc::SUCCESS;
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
__device__ errc lehmer_mean(int32_t* out, int32_t a, int32_t b)
{
  auto a2 = detail::ops::mul_overflow(a, a);
  if (!a2) return a2.error();

  auto b2 = detail::ops::mul_overflow(b, b);
  if (!b2) return b2.error();

  auto a_b_sum = detail::ops::add_overflow(a, b);
  if (!a_b_sum) return a_b_sum.error();

  auto a2_b2_sum = detail::ops::add_overflow(a2.value(), b2.value());
  if (!a2_b2_sum) return a2_b2_sum.error();

  auto result = detail::ops::div_overflow(a2_b2_sum.value(), a_b_sum.value());
  if (!result) return result.error();
  *out = result.value();
  return errc::SUCCESS;
}

__device__ errc bankers_round(numeric::decimal128* out, numeric::decimal128 x)
{
  using Rep = numeric::decimal128::rep;

  auto const scale = x.scale();
  auto const value = x.value();

  if (scale >= numeric::scale_type{0}) {
    *out = x;
    return errc::SUCCESS;
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
  return errc::SUCCESS;
}

__device__ errc fourth_power(float* C, float a)
{
  *C = a * a * a * a;

  return errc::SUCCESS;
}

__device__ errc mulsub(int* output, int input)
{
  *output = input * input - input;
  return errc::SUCCESS;
}

__device__ errc ascii_upper(signed char* output, signed char input)
{
  if (input > 96 && input < 123) {
    *output = input - 32;
  } else {
    *output = input;
  }

  return errc::SUCCESS;
}

__device__ errc decode_float(float* output, float input)
{
  *output = input;

  return errc::SUCCESS;
}

__device__ errc addmul(float* out, float a, float b, float c)
{
  *out = (a + b) * c;

  return errc::SUCCESS;
}

__device__ errc lerp(float* out, float a, float b, float t)
{
  *out = a - t * a + t * b;
  return errc::SUCCESS;
}

__device__ errc add_f32(float* out, float a, float b)
{
  *out = a * a * a + b;
  return errc::SUCCESS;
}

__device__ errc add_i32(int64_t* out, int32_t a, int32_t b)
{
  auto const value = static_cast<int64_t>(a);
  *out             = value * value * value + b;
  return errc::SUCCESS;
}

__device__ errc add_i32_i64(int64_t* out, int32_t a, int64_t b)
{
  auto const value = static_cast<int64_t>(a);
  *out             = value * value * value + b * b;
  return errc::SUCCESS;
}

__device__ errc division_by_zero(int32_t*, int32_t, int32_t) { return errc::DIVISION_BY_ZERO; }

}  // extern "C"

__device__ unsigned long long issued_ids = 0;

extern "C" {

__device__ errc lto_add_one(int32_t* out, int32_t a)
{
  *out = a + 1;
  return errc::SUCCESS;
}

__device__ errc lto_twice(int32_t* out, int32_t a)
{
  *out = a * 2;
  return errc::SUCCESS;
}

// Fails a row with a negative argument.
__device__ errc lto_checked_halve(int32_t* out, int32_t a)
{
  if (a < 0) { return errc::ARITHMETIC_OVERFLOW; }
  *out = a / 2;
  return errc::SUCCESS;
}

// Divides 100 by its argument, failing a zero with DIVISION_BY_ZERO and a negative argument with
// the lower code ARITHMETIC_OVERFLOW.
__device__ errc lto_checked_hundred_over(int32_t* out, int32_t a)
{
  if (a == 0) { return errc::DIVISION_BY_ZERO; }
  if (a < 0) { return errc::ARITHMETIC_OVERFLOW; }
  *out = 100 / a;
  return errc::SUCCESS;
}

// Fails a row with the error code it is given; zero succeeds.
__device__ errc lto_fail_with(int32_t* out, int32_t code)
{
  *out = code;
  return static_cast<errc>(code);
}

// Named like a parameter and a local of the generated function, which must not hide them.
__device__ errc in_0(int32_t* out, int32_t a)
{
  *out = a + 1;
  return errc::SUCCESS;
}

__device__ errc tmp_0(int32_t* out, int32_t a)
{
  *out = a * 2;
  return errc::SUCCESS;
}

__device__ errc lto_xor_popcount(int32_t* out, int32_t a, int32_t b)
{
  *out = __popc(a ^ b);
  return errc::SUCCESS;
}

// Returns a new number on every call and ignores its argument, so it is not pure.
__device__ errc lto_next_id(int64_t* out, int32_t)
{
  *out = static_cast<int64_t>(atomicAdd(&issued_ids, 1ULL));
  return errc::SUCCESS;
}

// The Monday on or before a day; 1970-01-01 was a Thursday.
__device__ errc lto_week_start_ns(timestamp_ns* out, timestamp_ns in)
{
  constexpr long long ticks_per_day = 86'400'000'000'000LL;
  auto const ticks                  = in.time_since_epoch().count();
  auto days                         = ticks / ticks_per_day;
  if (ticks % ticks_per_day != 0 && ticks < 0) { --days; }
  auto const shifted = days + 3;
  auto weeks         = shifted / 7;
  if (shifted % 7 != 0 && shifted < 0) { --weeks; }
  *out = timestamp_ns{duration_ns{(weeks * 7 - 3) * ticks_per_day}};
  return errc::SUCCESS;
}

__device__ errc lto_decimal_add(numeric::decimal64* out,
                                numeric::decimal32 a,
                                numeric::decimal64 b,
                                int32_t output_scale)
{
  auto const widened_a = numeric::decimal64{numeric::scaled_integer<int64_t>{a.value(), a.scale()}};
  auto const sum       = (widened_a + b).rescaled(numeric::scale_type{output_scale});
  // The caller uses only the representation, so this scale deliberately differs from output_scale.
  *out = numeric::decimal64{numeric::scaled_integer<int64_t>{sum.value(), numeric::scale_type{0}}};
  return errc::SUCCESS;
}

__device__ errc lto_string_tail(string_view* out, string_view input)
{
  *out = input.empty() ? input : input.substr(1, input.length() - 1);
  return errc::SUCCESS;
}

__device__ errc lto_nullable_string_tail(cuda::std::optional<string_view>* out,
                                         cuda::std::optional<string_view> input)
{
  auto const value = input.value_or(string_view{});
  *out             = value.empty() ? value : value.substr(1, value.length() - 1);
  return errc::SUCCESS;
}

__device__ errc lto_constant(int32_t* out)
{
  *out = 42;
  return errc::SUCCESS;
}

__device__ errc lto_nullable_constant(cuda::std::optional<int32_t>* out)
{
  *out = cuda::std::nullopt;
  return errc::SUCCESS;
}

}  // extern "C"

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

template <typename T>
__device__ errc greater(bool* output, T left, T right)
{
  *output = left > right;
  return errc::SUCCESS;
}

template <typename T>
__device__ errc equal_first(bool* output, T left0, T, T right0, T)
{
  *output = left0 == right0;
  return errc::SUCCESS;
}

template <typename T>
__device__ errc greater_second(bool* output, T, T left1, T, T right1)
{
  *output = left1 > right1;
  return errc::SUCCESS;
}

extern "C" {

__device__ errc greater_equal_double(bool* output, double left_col0, double right_col0)
{
  *output = left_col0 >= right_col0;

  return errc::SUCCESS;
}

__device__ errc greater_both_int32_t(
  bool* output, int32_t left_col0, int32_t left_col1, int32_t right_col0, int32_t right_col1)
{
  *output = (left_col0 > right_col0) && (left_col1 > right_col1);

  return errc::SUCCESS;
}

__device__ errc equal_int32_t(bool* output, int32_t left_col0, int32_t right_col0)
{
  *output = left_col0 == right_col0;

  return errc::SUCCESS;
}

__device__ errc xor_popcount(bool* output, int32_t left_val, int32_t right_val)
{
  *output = __popc(left_val ^ right_val) <= 1;

  return errc::SUCCESS;
}

__device__ errc checked_greater_equal(bool* output, int a, int b)
{
  if (a < 0) { return errc::ARITHMETIC_OVERFLOW; }
  if (b == 0) { return errc::DIVISION_BY_ZERO; }
  *output = a >= b;
  return errc::SUCCESS;
}

#define DEFINE_JOIN_COMPARISONS(type)                                 \
  __device__ errc greater_##type(bool* output, type left, type right) \
  {                                                                   \
    return greater(output, left, right);                              \
  }                                                                   \
  __device__ errc equal_first_##type(                                 \
    bool* output, type left0, type left1, type right0, type right1)   \
  {                                                                   \
    return equal_first(output, left0, left1, right0, right1);         \
  }                                                                   \
  __device__ errc greater_second_##type(                              \
    bool* output, type left0, type left1, type right0, type right1)   \
  {                                                                   \
    return greater_second(output, left0, left1, right0, right1);      \
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

}  // extern "C"
}  // namespace cudf::test_udfs
