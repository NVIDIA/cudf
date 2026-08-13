/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/integral_math.cuh>
#include <cudf/detail/operators/concepts.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/cmath>
#include <cuda/std/concepts>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Computes cube root
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <floating_point T>
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
template <floating_point T>
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
    auto div = a.value() / factor;
    auto rem = a.value() % factor;
    if (rem == 0) {
      return a;
    } else {
      auto val = ceil ? (a.value() > 0 ? (div + 1) : div) : (a.value() > 0 ? div : (div - 1));
      return numeric::decimal<R>{numeric::scaled_integer<R>{val * factor, a.scale()}};
    }
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
template <floating_point T>
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
template <floating_point T>
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
template <floating_point T>
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
template <floating_point A, floating_point B>
__device__ auto pow(A a, B b) -> decltype(cuda::std::pow(a, b))
{
  return cuda::std::pow(a, b);
}

template <integer A, integer B>
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
template <floating_point T>
__device__ T rint(T a)
{
  return cuda::std::rint(a);
}

/**
 * @brief Computes square root.
 *
 * @param a Input value.
 */
template <floating_point T>
__device__ T sqrt(T a)
{
  return cuda::std::sqrt(a);
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
