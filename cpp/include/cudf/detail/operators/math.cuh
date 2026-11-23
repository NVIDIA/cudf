/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/integral_math.cuh>
#include <cudf/detail/operators/concepts.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/cmath>
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Tests whether a floating-point value is NaN.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ bool is_nan(T a)
{
  return cuda::std::isnan(a);
}

/**
 * @brief Tests whether a floating-point value is not NaN.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ bool is_not_nan(T a)
{
  return !is_nan(a);
}

/**
 * @brief Tests whether a floating-point value is finite.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ bool is_finite(T a)
{
  return cuda::std::isfinite(a);
}

/**
 * @brief Canonicalizes NaNs and replaces either signed zero with positive zero.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T normalize_nans_and_zeros(T a)
{
  if (is_nan(a)) { return cuda::std::numeric_limits<T>::quiet_NaN(); }
  return a == T{0.0} ? T{0.0} : a;
}

// Separate names keep AST dispatch from accepting binary forms of the unary ceil/floor operators.
/**
 * @brief Rounds an integral representation up to a multiple of a positive factor.
 *
 * @tparam T Representation type.
 * @param a Input representation.
 * @param factor Positive rounding factor, such as ten to the negated decimal scale.
 */
template <cuda::std::integral T>
__device__ T ceil_to_multiple(T a, T factor)
{
  auto const truncated = (a / factor) * factor;
  return truncated + (a > 0 && truncated != a ? factor : 0);
}

/**
 * @brief Rounds an integral representation down to a multiple of a positive factor.
 *
 * @tparam T Representation type.
 * @param a Input representation.
 * @param factor Positive rounding factor, such as ten to the negated decimal scale.
 */
template <cuda::std::integral T>
__device__ T floor_to_multiple(T a, T factor)
{
  auto const truncated = (a / factor) * factor;
  return truncated - (a < 0 && truncated != a ? factor : 0);
}

/**
 * @brief Computes cube root
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T cbrt(T a)
{
  return cuda::std::cbrt(a);
}

/**
 * @brief Computes ceiling.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T ceil(T a)
{
  return cuda::std::ceil(a);
}

namespace detail {

/**
 * @brief Rounds a decimal value to an integral value.
 *
 * @tparam R Rep type of the decimal.
 * @tparam ceil If true `ceil`s the value, otherwise `floor`s the value.
 */
template <typename R, bool ceil>
__device__ numeric::decimal<R> decimal_round(numeric::decimal<R> a)
{
  if (a.scale() >= 0) {
    return a;
  } else {
    auto factor =
      numeric::detail::ipow<R, numeric::Radix::BASE_10>(-static_cast<int32_t>(a.scale()));
    auto rep =
      ceil ? ops::ceil_to_multiple(a.value(), factor) : ops::floor_to_multiple(a.value(), factor);
    return numeric::decimal<R>{numeric::scaled_integer<R>{rep, a.scale()}};
  }
}

}  // namespace detail

template <typename R>
__device__ numeric::decimal<R> ceil(numeric::decimal<R> a)
{
  return detail::decimal_round<R, true>(a);
}

/**
 * @brief Computes natural exponential.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T exp(T a)
{
  return cuda::std::exp(a);
}

/**
 * @brief Computes floor of a value.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T floor(T a)
{
  return cuda::std::floor(a);
}

template <typename R>
__device__ numeric::decimal<R> floor(numeric::decimal<R> a)
{
  return detail::decimal_round<R, false>(a);
}

/**
 * @brief Computes natural logarithm.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T log(T a)
{
  return cuda::std::log(a);
}

/**
 * @brief Computes exponentiation.
 *
 * @tparam A Base value type.
 * @tparam B Exponent value type.
 * @param a Base value.
 * @param b Exponent value.
 */
template <cuda::std::floating_point A, cuda::std::floating_point B>
__device__ auto pow(A a, B b) -> decltype(cuda::std::pow(a, b))
{
  return cuda::std::pow(a, b);
}

template <concepts::integral_not_bool A, concepts::integral_not_bool B>
__device__ auto pow(A a, B b) -> decltype(cudf::detail::integral_pow(a, b))
{
  return cudf::detail::integral_pow(a, b);
}

/**
 * @brief Rounds to integral value.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T rint(T a)
{
  return cuda::std::rint(a);
}

/**
 * @brief Computes square root.
 *
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T sqrt(T a)
{
  return cuda::std::sqrt(a);
}

/**
 * @brief Computes integral exponentiation, retaining the base type.
 *
 * @tparam A Base type.
 * @tparam B Exponent type.
 * @param a Base value.
 * @param b Exponent value.
 */
template <cuda::std::integral A, cuda::std::integral B>
__device__ A int_pow(A a, B b)
{
  return cudf::detail::integral_pow(a, b);
}

/**
 * @brief Computes the logarithm of a value in the supplied base, returning double.
 *
 * @tparam A Value type.
 * @tparam B Base type.
 * @param a Input value.
 * @param b Logarithm base.
 */
template <typename A, typename B>
__device__ double log_base(A a, B b)
  requires(cuda::std::is_convertible_v<A, double> && cuda::std::is_convertible_v<B, double>)
{
  return cuda::std::log(static_cast<double>(a)) / cuda::std::log(static_cast<double>(b));
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
