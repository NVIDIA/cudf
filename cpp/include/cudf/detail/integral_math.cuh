/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

namespace cudf::detail {

/**
 * @brief Integral exponentiation via binary exponentiation.
 *
 * Returns `base` raised to the power `exp`. Negative exponents yield zero
 * (integer truncation toward zero). The result type is always `Base`.
 *
 * @tparam Base The type of the base operand.
 * @tparam Exp The type of the exponent operand.
 * @param base Base operand
 * @param exp Exponent operand
 * @return Result of `base` to the power of `exp` of type `Base`
 */
template <cuda::std::integral Base, cuda::std::integral Exp>
__device__ inline constexpr auto integral_pow(Base base, Exp exp) -> Base
{
  if constexpr (cuda::std::is_signed_v<Exp>) {
    if (exp < 0) {
      // Integer exponentiation with negative exponent is not possible.
      return 0;
    }
  }
  if (exp == 0) { return 1; }
  if (base == 0) { return 0; }
  Base extra = 1;
  while (exp > 1) {
    // The exponent is odd, so multiply by one factor of x.
    if (exp & 1) { extra *= base; }
    // The exponent is even, so square x and divide the exponent y by 2.
    exp /= 2;
    base *= base;
  }
  return base * extra;
}

/**
 * @brief Integral floor division.
 *
 * For signed types this is Euclidean-style: the quotient is rounded toward
 * negative infinity (matching Python's `//` semantics). For unsigned types
 * this is plain truncating division.
 *
 * @tparam LHS Left-hand side operand type
 * @tparam RHS Right-hand side operand type
 * @tparam Common Common type of left-hand side and right-hand side operands
 * @param lhs Left-hand side operand
 * @param rhs Right-hand side operand
 * @return Result of integer floor division of `lhs` by `rhs` of type `Common`
 */
template <typename LHS,
          typename RHS,
          cuda::std::integral Common = cuda::std::common_type_t<LHS, RHS>>
__device__ inline constexpr auto integral_floor_div(LHS lhs, RHS rhs) -> Common
{
  if constexpr (cuda::std::is_signed_v<Common>) {
    auto const numerator         = static_cast<Common>(lhs);
    auto const denominator       = static_cast<Common>(rhs);
    auto const quotient          = numerator / denominator;
    auto const nonzero_remainder = (numerator % denominator) != 0;
    auto const mixed_sign        = (numerator ^ denominator) < 0;
    return quotient - mixed_sign * nonzero_remainder;
  } else {
    return static_cast<Common>(lhs) / static_cast<Common>(rhs);
  }
}

}  // namespace cudf::detail
