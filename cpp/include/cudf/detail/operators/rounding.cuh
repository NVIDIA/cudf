/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/detail/operators/math.cuh>
#include <cudf/fixed_point/temporary.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/cmath>
#include <cuda/std/concepts>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

namespace detail {

template <bool HalfEven, cuda::std::floating_point T>
__device__ T round_nearest(T a)
{
  if constexpr (HalfEven) {
    return ops::rint(a);
  } else {
    return cuda::std::round(a);
  }
}

template <cuda::std::integral T>
__device__ T rounding_abs(T a)
{
  if constexpr (cuda::std::signed_integral<T>) {
    return numeric::detail::abs(a);
  } else {
    return a;
  }
}

template <bool HalfEven, concepts::arithmetic_not_bool T>
__device__ T round_to_multiple(T a, T factor)
{
  if constexpr (cuda::std::floating_point<T>) {
    return round_nearest<HalfEven>(a / factor) * factor;
  } else {
    auto const quotient = a / factor;
    auto const down     = quotient * factor;
    auto const diff     = rounding_abs(a - down);
    auto const sign     = [&] {
      if constexpr (cuda::std::signed_integral<T>) {
        return a < 0 ? -1 : 1;
      } else {
        return 1;
      }
    }();
    if constexpr (HalfEven) {
      auto const adjustment =
        (diff > factor / 2 || (diff == factor / 2 && rounding_abs(quotient) % 2 == 1)) ? factor : 0;
      return down + sign * adjustment;
    } else {
      return down + sign * (diff >= factor / 2 ? factor : 0);
    }
  }
}

template <bool HalfEven, cuda::std::floating_point T>
__device__ T round_fractional(T a, T factor)
{
  // Split before scaling so the integral part does not lose precision or overflow.
  T integral_part;
  auto const fractional_part = cuda::std::modf(a, &integral_part);
  return integral_part + round_nearest<HalfEven>(fractional_part * factor) / factor;
}

}  // namespace detail

/**
 * @brief Rounds to the nearest integer, with ties away from zero.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T round_half_up(T a)
{
  return detail::round_nearest<false>(a);
}

/**
 * @brief Rounds to the nearest integer, with ties to even.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T round_half_even(T a)
{
  return detail::round_nearest<true>(a);
}

/**
 * @brief Rounds to a multiple of a positive power of ten, with ties away from zero.
 *
 * @tparam T Value or decimal representation type.
 * @param a Input value or representation.
 * @param factor Power of ten, at least 10, defining the rounding increment.
 */
template <concepts::arithmetic_not_bool T>
__device__ T round_half_up_to_multiple(T a, T factor)
{
  return detail::round_to_multiple<false>(a, factor);
}

/**
 * @brief Rounds to a multiple of a positive power of ten, with ties to even.
 *
 * @tparam T Value or decimal representation type.
 * @param a Input value or representation.
 * @param factor Power of ten, at least 10, defining the rounding increment.
 */
template <concepts::arithmetic_not_bool T>
__device__ T round_half_even_to_multiple(T a, T factor)
{
  return detail::round_to_multiple<true>(a, factor);
}

/**
 * @brief Rounds fractional decimal places, with ties away from zero.
 *
 * @tparam T Value type.
 * @param a Input value.
 * @param factor Ten raised to the positive number of decimal places to retain.
 */
template <cuda::std::floating_point T>
__device__ T round_half_up_fractional(T a, T factor)
{
  return detail::round_fractional<false>(a, factor);
}

/**
 * @brief Rounds fractional decimal places, with ties to even.
 *
 * @tparam T Value type.
 * @param a Input value.
 * @param factor Ten raised to the positive number of decimal places to retain.
 */
template <cuda::std::floating_point T>
__device__ T round_half_even_fractional(T a, T factor)
{
  return detail::round_fractional<true>(a, factor);
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
