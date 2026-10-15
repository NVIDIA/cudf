/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/round.hpp>

#include <cuda/std/type_traits>

#include <cassert>

namespace cudf::detail {

/**
 * @brief Divides two fixed-point numbers, keeping the scale of the dividend.
 *
 * Similar to Java's `BigDecimal.divide(divisor, roundingMode)`.
 *
 * @param lhs The dividend
 * @param rhs The divisor
 * @param method The rounding method to use
 * @return The quotient, with the scale of `lhs`
 */
template <typename Rep, numeric::Radix Rad>
__device__ inline numeric::fixed_point<Rep, Rad> divide_decimal(
  numeric::fixed_point<Rep, Rad> const& lhs,
  numeric::fixed_point<Rep, Rad> const& rhs,
  rounding_method method)
{
#if defined(__CUDACC_DEBUG__)
  assert(rhs.value() != 0 && "division by zero");
  assert(!numeric::division_overflow<Rep>(lhs.value(), rhs.value()) && "fixed_point overflow");
#endif

  // Dividing at scale `lhs.scale() - rhs.scale()` and rescaling to `lhs.scale()` is the same as
  // multiplying the dividend by 10^(-rhs.scale()) before dividing.
  auto const scale_factor = numeric::detail::ipow<Rep, Rad>(-static_cast<int>(rhs.scale()));

  if (!numeric::multiplication_overflow<Rep>(lhs.value(), scale_factor)) {
    Rep scaled_dividend = lhs.value() * scale_factor;
    Rep quotient        = scaled_dividend / rhs.value();
    Rep remainder       = scaled_dividend % rhs.value();

    if (method == rounding_method::HALF_UP) {
      auto abs_remainder = (remainder < 0) ? -remainder : remainder;
      auto abs_divisor   = (rhs.value() < 0) ? -rhs.value() : rhs.value();
      if (abs_remainder * 2 >= abs_divisor) { quotient += (quotient >= 0) ? 1 : -1; }
    } else if (method == rounding_method::HALF_EVEN) {
      auto abs_rem_2   = ((remainder < 0) ? -remainder : remainder) * 2;
      auto abs_divisor = (rhs.value() < 0) ? -rhs.value() : rhs.value();
      if (abs_rem_2 > abs_divisor || (abs_rem_2 == abs_divisor && (quotient % 2) != 0)) {
        quotient += (quotient >= 0) ? 1 : -1;
      }
    }

    return numeric::fixed_point<Rep, Rad>{numeric::scaled_integer<Rep>{quotient, lhs.scale()}};
  }

  if constexpr (cuda::std::is_same_v<Rep, int32_t>) {
    using WiderRep      = int64_t;
    WiderRep wide_scale = static_cast<WiderRep>(scale_factor);
    bool overflow_in_int64 =
      numeric::multiplication_overflow<WiderRep>(static_cast<WiderRep>(lhs.value()), wide_scale);

    if (!overflow_in_int64) {
      WiderRep scaled_dividend = static_cast<WiderRep>(lhs.value()) * wide_scale;
      WiderRep wide_divisor    = static_cast<WiderRep>(rhs.value());
      WiderRep quotient        = scaled_dividend / wide_divisor;
      WiderRep remainder       = scaled_dividend % wide_divisor;

      if (method == rounding_method::HALF_UP) {
        auto abs_remainder = (remainder < 0) ? -remainder : remainder;
        auto abs_divisor   = (wide_divisor < 0) ? -wide_divisor : wide_divisor;
        if (abs_remainder * 2 >= abs_divisor) { quotient += (quotient >= 0) ? 1 : -1; }
      } else if (method == rounding_method::HALF_EVEN) {
        auto abs_rem_2   = ((remainder < 0) ? -remainder : remainder) * 2;
        auto abs_divisor = (wide_divisor < 0) ? -wide_divisor : wide_divisor;
        if (abs_rem_2 > abs_divisor || (abs_rem_2 == abs_divisor && (quotient % 2) != 0)) {
          quotient += (quotient >= 0) ? 1 : -1;
        }
      }

      return numeric::fixed_point<Rep, Rad>{
        numeric::scaled_integer<Rep>{static_cast<Rep>(quotient), lhs.scale()}};
    }
  }

  using WidestRep           = __int128_t;
  WidestRep wide_scale      = static_cast<WidestRep>(scale_factor);
  WidestRep scaled_dividend = static_cast<WidestRep>(lhs.value()) * wide_scale;
  WidestRep wide_divisor    = static_cast<WidestRep>(rhs.value());
  WidestRep quotient        = scaled_dividend / wide_divisor;
  WidestRep remainder       = scaled_dividend % wide_divisor;

  if (method == rounding_method::HALF_UP) {
    WidestRep abs_rem = (remainder < 0) ? -remainder : remainder;
    WidestRep abs_div = (wide_divisor < 0) ? -wide_divisor : wide_divisor;
    if (abs_rem * 2 >= abs_div) { quotient += (quotient >= 0) ? 1 : -1; }
  } else if (method == rounding_method::HALF_EVEN) {
    WidestRep abs_rem   = (remainder < 0) ? -remainder : remainder;
    WidestRep abs_div   = (wide_divisor < 0) ? -wide_divisor : wide_divisor;
    WidestRep abs_rem_2 = abs_rem * 2;
    if (abs_rem_2 > abs_div || (abs_rem_2 == abs_div && (quotient % 2) != 0)) {
      quotient += (quotient >= 0) ? 1 : -1;
    }
  }

  return numeric::fixed_point<Rep, Rad>{
    numeric::scaled_integer<Rep>{static_cast<Rep>(quotient), lhs.scale()}};
}

}  // namespace cudf::detail
