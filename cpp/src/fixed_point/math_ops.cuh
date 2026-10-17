/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/round.hpp>

#include <cuda/cmath>
#include <cuda/numeric>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/optional>

namespace cudf::detail {

/**
 * @brief Unsigned type used to hold the magnitudes of `Rep` operands during division.
 *
 * It is wider than `Rep` where possible, so that rescaling the dividend overflows less often.
 */
template <typename Rep>
struct division_magnitude {
  using type = __uint128_t;
};

template <>
struct division_magnitude<int32_t> {
  using type = cuda::std::uint64_t;
};

template <typename Rep>
using division_magnitude_t = typename division_magnitude<Rep>::type;

/**
 * @brief Computes `Rad` to the non-negative power `exponent`, returning false instead if it
 * overflows `U`.
 */
template <typename U, numeric::Radix Rad>
__device__ inline bool checked_power(U& result, int32_t exponent)
{
  // The largest exponent for which `Rad^exponent` fits in `U`
  constexpr int32_t max_exponent = Rad == numeric::Radix::BASE_2
                                     ? cuda::std::numeric_limits<U>::digits - 1
                                     : cuda::std::numeric_limits<U>::digits10;
  if (exponent > max_exponent) { return false; }
  result = numeric::detail::ipow<U, Rad>(exponent);
  return true;
}

/**
 * @brief Divides two fixed-point numbers, keeping the scale of the dividend.
 *
 * Similar to Java's `BigDecimal.divide(divisor, roundingMode)`. The quotient is computed exactly
 * and then rounded to the scale of `lhs` using `method`.
 *
 * Returns no value if `rhs` is zero, if the rounded quotient does not fit in `Rep`, or, for
 * 128-bit `Rep`, if `lhs` rescaled to the scale of the quotient does not fit in 128 bits.
 *
 * @param lhs The dividend
 * @param rhs The divisor
 * @param method The rounding method to use
 * @return The quotient with the scale of `lhs`, or no value
 */
template <typename Rep, numeric::Radix Rad>
__device__ inline cuda::std::optional<numeric::fixed_point<Rep, Rad>> divide_and_round(
  numeric::fixed_point<Rep, Rad> const& lhs,
  numeric::fixed_point<Rep, Rad> const& rhs,
  rounding_method method)
{
  using U = division_magnitude_t<Rep>;

  auto const make_result = [&](Rep value) {
    return numeric::fixed_point<Rep, Rad>{numeric::scaled_integer<Rep>{value, lhs.scale()}};
  };
  auto const magnitude = [](Rep value) { return static_cast<U>(cuda::uabs(value)); };

  if (rhs.value() == 0) { return cuda::std::nullopt; }
  if (lhs.value() == 0) { return make_result(0); }

  bool const negative = (lhs.value() < 0) != (rhs.value() < 0);
  U dividend          = magnitude(lhs.value());
  U divisor           = magnitude(rhs.value());

  // The quotient at the scale of `lhs` is `lhs.value() * radix^(-rhs.scale()) / rhs.value()`.
  auto const exponent = -static_cast<int32_t>(rhs.scale());
  U factor{};
  if (exponent >= 0) {
    if (!checked_power<U, Rad>(factor, exponent) ||
        cuda::mul_overflow(dividend, dividend, factor)) {
      return cuda::std::nullopt;
    }
  } else if (!checked_power<U, Rad>(factor, -exponent) ||
             cuda::mul_overflow(divisor, divisor, factor)) {
    // The divisor is more than twice the largest possible dividend, so the quotient rounds to 0.
    return make_result(0);
  }

  U quotient            = dividend / divisor;
  U const remainder     = dividend % divisor;
  U const rest          = divisor - remainder;
  bool const round_away = method == rounding_method::HALF_EVEN
                            ? (remainder > rest || (remainder == rest && quotient % 2 != 0))
                            : remainder >= rest;
  if (round_away) { ++quotient; }

  auto const result =
    negative ? cuda::sub_overflow<Rep>(Rep{0}, quotient) : cuda::overflow_cast<Rep>(quotient);
  if (result.overflow) { return cuda::std::nullopt; }

  return make_result(result.value);
}

}  // namespace cudf::detail
