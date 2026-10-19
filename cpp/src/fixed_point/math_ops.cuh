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
 * @brief Divides two decimal values and rounds the quotient to the output scale.
 *
 * Similar to Java's `BigDecimal.divide(divisor, scale, roundingMode)`. The quotient is
 * `lhs * 10^exponent / rhs`, computed exactly and then rounded using `method`, where
 * `exponent = lhs scale - rhs scale - output scale`.
 *
 * Returns no value if `rhs` is zero, if the rounded quotient has more digits than `OutRep` can
 * always hold, or if `lhs * 10^exponent` does not fit in the magnitude type of `CompRep`.
 *
 * @tparam OutRep The representation type of the output
 * @tparam CompRep The representation type both operands have been widened to
 * @param lhs The dividend's representation
 * @param rhs The divisor's representation
 * @param exponent The power of 10 to rescale the quotient by
 * @param method The rounding method to use
 * @return The quotient's representation at the output scale, or no value
 */
template <typename OutRep, typename CompRep>
__device__ inline cuda::std::optional<OutRep> divide_and_round(CompRep lhs,
                                                               CompRep rhs,
                                                               int32_t exponent,
                                                               rounding_method method)
{
  using U = division_magnitude_t<CompRep>;

  auto const magnitude = [](CompRep value) { return static_cast<U>(cuda::uabs(value)); };

  if (rhs == 0) { return cuda::std::nullopt; }
  if (lhs == 0) { return OutRep{0}; }

  bool const negative = (lhs < 0) != (rhs < 0);
  U dividend          = magnitude(lhs);
  U divisor           = magnitude(rhs);

  U factor{};
  if (exponent >= 0) {
    if (!checked_power<U, numeric::Radix::BASE_10>(factor, exponent) ||
        cuda::mul_overflow(dividend, dividend, factor)) {
      return cuda::std::nullopt;
    }
  } else if (!checked_power<U, numeric::Radix::BASE_10>(factor, -exponent) ||
             cuda::mul_overflow(divisor, divisor, factor)) {
    // The divisor is more than twice the largest possible dividend, so the quotient rounds to 0.
    return OutRep{0};
  }

  U quotient            = dividend / divisor;
  U const remainder     = dividend % divisor;
  U const rest          = divisor - remainder;
  bool const round_away = method == rounding_method::HALF_EVEN
                            ? (remainder > rest || (remainder == rest && quotient % 2 != 0))
                            : remainder >= rest;
  if (round_away) { ++quotient; }

  // The largest magnitude with no more digits than `OutRep` can always hold
  constexpr U max_magnitude =
    numeric::detail::ipow<U, numeric::Radix::BASE_10>(cuda::std::numeric_limits<OutRep>::digits10) -
    1;
  if (quotient > max_magnitude) { return cuda::std::nullopt; }

  auto const value = static_cast<OutRep>(quotient);
  return negative ? -value : value;
}

}  // namespace cudf::detail
