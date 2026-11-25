/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/fixed_point/math_ops.hpp>
#include <cudf/round.hpp>

#include <cuda/cmath>
#include <cuda/numeric>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
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
 * @brief Computes 10 to the non-negative power `exponent`, returning false instead if it
 * overflows `U`.
 */
template <typename U>
__device__ inline bool checked_power(U& result, int32_t exponent)
{
  if (exponent > cuda::std::numeric_limits<U>::digits10) { return false; }
  result = numeric::detail::ipow<U, numeric::Radix::BASE_10>(exponent);
  return true;
}

/**
 * @brief Rounds the magnitude of a quotient using its remainder, and applies its sign.
 *
 * Returns no value if the rounded magnitude has more digits than `OutRep` can always hold.
 *
 * @tparam OutRep The representation type of the output
 * @tparam U The unsigned type holding the magnitudes
 * @param quotient The magnitude of the truncated quotient
 * @param remainder The remainder of the division, less than `divisor`
 * @param divisor The magnitude of the divisor
 * @param negative Whether the quotient is negative
 * @param method The rounding method to use
 * @return The rounded quotient, or no value
 */
template <typename OutRep, typename U>
__device__ inline cuda::std::optional<OutRep> round_quotient(
  U quotient, U remainder, U divisor, bool negative, rounding_method method)
{
  U const rest          = divisor - remainder;
  bool const round_away = method == rounding_method::HALF_EVEN
                            ? (remainder > rest || (remainder == rest && quotient % 2 != 0))
                            : remainder >= rest;

  // The largest magnitude with no more digits than `OutRep` can always hold
  constexpr U max_magnitude =
    numeric::detail::ipow<U, numeric::Radix::BASE_10>(cuda::std::numeric_limits<OutRep>::digits10) -
    1;
  // Checked before rounding, so that the increment can't wrap
  auto const increment = static_cast<U>(round_away);
  if (quotient > max_magnitude - increment) { return cuda::std::nullopt; }

  auto const value = static_cast<OutRep>(quotient + increment);
  return negative ? -value : value;
}

/**
 * @brief Divides `remainder * 2^64 + low` by `divisor`, replacing `remainder` with the remainder.
 *
 * Requires the top bit of `divisor` to be set and `remainder < divisor`, so that the quotient fits
 * in 64 bits.
 *
 * @param remainder The high 128 bits of the dividend, replaced with the remainder
 * @param low The low 64 bits of the dividend
 * @param divisor The divisor
 * @return The quotient
 */
__device__ inline cuda::std::uint64_t divide_3by2(__uint128_t& remainder,
                                                  cuda::std::uint64_t low,
                                                  __uint128_t divisor)
{
  using U         = __uint128_t;
  auto const d_hi = static_cast<cuda::std::uint64_t>(divisor >> 64);
  auto const d_lo = static_cast<cuda::std::uint64_t>(divisor);
  auto const r_hi = static_cast<cuda::std::uint64_t>(remainder >> 64);
  // Estimated from the high 64 bits of `divisor`, this is at most 2 too large since `d_hi >= 2^63`
  U estimate =
    r_hi == d_hi ? U{cuda::std::numeric_limits<cuda::std::uint64_t>::max()} : remainder / d_hi;
  U estimate_rest = remainder - estimate * d_hi;
  while ((estimate_rest >> 64) == 0 && estimate * d_lo > ((estimate_rest << 64) | low)) {
    --estimate;
    estimate_rest += d_hi;
  }
  // The true remainder is less than `divisor`, so computing it modulo 2^128 is exact
  remainder = ((remainder << 64) | low) - estimate * divisor;
  return static_cast<cuda::std::uint64_t>(estimate);
}

/**
 * @brief Divides `dividend * 10^exponent` by `divisor` with a 256-bit intermediate, and rounds
 * the quotient.
 *
 * Used when `dividend * 10^exponent` overflows the magnitude type of the operands. Kept out of
 * line so that it doesn't add to the register use of the common path.
 *
 * @tparam OutRep The representation type of the output
 * @param dividend The magnitude of the dividend, at most 2^127
 * @param divisor The magnitude of the divisor, nonzero and at most 2^127
 * @param exponent The non-negative power of 10 to rescale the dividend by
 * @param negative Whether the quotient is negative
 * @param method The rounding method to use
 * @return The rounded quotient, or no value if it has more digits than `OutRep` can always hold
 */
template <typename OutRep>
__device__ __noinline__ cuda::std::optional<OutRep> divide_and_round_wide(__uint128_t dividend,
                                                                          __uint128_t divisor,
                                                                          int32_t exponent,
                                                                          bool negative,
                                                                          rounding_method method)
{
  using U = __uint128_t;
  // The largest power of 10 that fits in `U`
  constexpr int32_t max_step = cuda::std::numeric_limits<U>::digits10;

  // The 256-bit product, as its high and low 128 bits
  U hi = 0;
  U lo = dividend;
  while (exponent > 0) {
    auto const step   = cuda::std::min(exponent, max_step);
    auto const factor = numeric::detail::ipow<U, numeric::Radix::BASE_10>(step);
    auto const carry  = cuda::mul_hi(lo, factor);
    lo *= factor;
    // A product of 2^256 or more gives a quotient of at least 2^129, which never fits
    if (cuda::mul_overflow(hi, hi, factor) || cuda::add_overflow(hi, hi, carry)) {
      return cuda::std::nullopt;
    }
    exponent -= step;
  }

  // The quotient is at least 2^128, which never fits
  if (hi >= divisor) { return cuda::std::nullopt; }

  // Long division in 64-bit digits, after shifting so that the top bit of the divisor is set.
  // Since `hi < divisor`, shifting `hi` by the same amount can't overflow.
  int const shift    = cuda::std::countl_zero(divisor);
  U const normalized = divisor << shift;
  if (shift > 0) {
    hi = (hi << shift) | (lo >> (cuda::std::numeric_limits<U>::digits - shift));
    lo <<= shift;
  }
  U remainder      = hi;
  auto const q_hi  = divide_3by2(remainder, static_cast<cuda::std::uint64_t>(lo >> 64), normalized);
  auto const q_lo  = divide_3by2(remainder, static_cast<cuda::std::uint64_t>(lo), normalized);
  U const quotient = (U{q_hi} << 64) | q_lo;
  return round_quotient<OutRep>(quotient, remainder >> shift, divisor, negative, method);
}

/**
 * @brief Divides two decimal values and rounds the quotient to the output scale.
 *
 * Similar to Java's `BigDecimal.divide(divisor, scale, roundingMode)`. The quotient is
 * `lhs * 10^exponent / rhs`, computed exactly and then rounded using `method`, where
 * `exponent = lhs scale - rhs scale - output scale`.
 *
 * Returns no value if `rhs` is zero, or if the rounded quotient has more digits than `OutRep` can
 * always hold. With `decimal_intermediate::STANDARD`, also returns no value if
 * `lhs * 10^exponent` does not fit in the magnitude type of `CompRep`.
 *
 * @tparam OutRep The representation type of the output
 * @tparam RhsRep The representation type of the divisor, before widening
 * @tparam CompRep The representation type both operands have been widened to
 * @param lhs The dividend's representation
 * @param rhs The divisor's representation
 * @param exponent The power of 10 to rescale the quotient by
 * @param method The rounding method to use
 * @param intermediate The width of the intermediate value used to rescale the dividend
 * @return The quotient's representation at the output scale, or no value
 */
template <typename OutRep, typename RhsRep, typename CompRep>
__device__ inline cuda::std::optional<OutRep> divide_and_round(CompRep lhs,
                                                               CompRep rhs,
                                                               int32_t exponent,
                                                               rounding_method method,
                                                               decimal_intermediate intermediate)
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
    if (!checked_power<U>(factor, exponent) || cuda::mul_overflow(dividend, dividend, factor)) {
      // An overflowing dividend is at least 2^64 (or 2^128), so unless the divisor or the output
      // is 128 bits wide, the quotient has more digits than the output can hold.
      if constexpr (sizeof(OutRep) == sizeof(__int128_t) || sizeof(RhsRep) == sizeof(__int128_t)) {
        if (intermediate == decimal_intermediate::WIDE) {
          return divide_and_round_wide<OutRep>(
            magnitude(lhs), magnitude(rhs), exponent, negative, method);
        }
      }
      return cuda::std::nullopt;
    }
  } else if (!checked_power<U>(factor, -exponent) || cuda::mul_overflow(divisor, divisor, factor)) {
    // The divisor is more than twice the largest possible dividend, so the quotient rounds to 0.
    return OutRep{0};
  }

  return round_quotient<OutRep>(dividend / divisor, dividend % divisor, divisor, negative, method);
}

}  // namespace cudf::detail
