/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/operators/operators.cuh>
#include <cudf/errc.hpp>
#include <cudf/fixed_point/fixed_point.hpp>

#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/expected>

template <typename Decimal>
__device__ Decimal bankers_round(Decimal x)
{
  using Rep = typename Decimal::rep;

  auto const scale = x.scale();
  auto const value = x.value();

  if (scale >= numeric::scale_type{0}) { return x; }

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

  return Decimal{q, numeric::scale_type{0}};
}

__device__ float distance(float x1, float y1, float x2, float y2)
{
  return cuda::std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

__device__ float invsqrt(float a) { return 1.0F / sqrtf(a); }

/**
 * @brief Calculates the integer Lehmer mean of two integers using checked arithmetic.
 * The Lehmer mean is defined as `(a^2 + b^2) / (a + b)`.
 * @param a The first integer.
 * @param b The second integer.
 * @return The Lehmer mean of the two integers, or an error code if an overflow occurs during the
 * calculation.
 *
 */
__device__ cuda::std::expected<int32_t, cudf::errc> lehmer_mean(int32_t a, int32_t b)
{
  auto a2 = cudf::detail::ops::mul_overflow(a, a);
  if (!a2) return cuda::std::unexpected(a2.error());

  auto b2 = cudf::detail::ops::mul_overflow(b, b);
  if (!b2) return cuda::std::unexpected(b2.error());

  auto a_b_sum = cudf::detail::ops::add_overflow(a, b);
  if (!a_b_sum) return cuda::std::unexpected(a_b_sum.error());

  auto a2_b2_sum = cudf::detail::ops::add_overflow(a2.value(), b2.value());
  if (!a2_b2_sum) return cuda::std::unexpected(a2_b2_sum.error());

  return cudf::detail::ops::div_overflow(a2_b2_sum.value(), a_b_sum.value());
}

__device__ float sum_of_squares(float a, float b) { return a * a + b * b; }

__device__ uint8_t to_upper(uint8_t input)
{
  if (input > 96 && input < 123) {
    return input - 32;
  } else {
    return input;
  }
}

extern "C" __device__ int bankers_round(numeric::decimal128* out, numeric::decimal128 in)
{
  *out = bankers_round(in);
  return 0;
}

extern "C" __device__ int distance(float* out, float x1, float y1, float x2, float y2)
{
  *out = distance(x1, y1, x2, y2);
  return 0;
}

extern "C" __device__ int invsqrt(float* out, float a)
{
  *out = invsqrt(a);
  return 0;
}

extern "C" __device__ int lehmer_mean(int32_t* out, int32_t a, int32_t b)
{
  auto result = lehmer_mean(a, b);
  if (!result) return static_cast<int>(result.error());
  *out = result.value();
  return 0;
}

extern "C" __device__ int sum_of_squares(float* out, float a, float b)
{
  *out = sum_of_squares(a, b);
  return 0;
}

extern "C" __device__ int to_upper(uint8_t* output, uint8_t input)
{
  *output = to_upper(input);
  return 0;
}
