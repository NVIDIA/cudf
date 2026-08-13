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
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Computes absolute value.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <typename T>
__device__ T abs(T a)
  requires(cuda::std::is_signed_v<T>)
{
  return cuda::std::abs(a);
}

template <unsigned_integer T>
__device__ T abs(T a)
{
  return a;
}

template <typename R>
__device__ numeric::decimal<R> abs(numeric::decimal<R> a)
{
  auto rep = a.value() < 0 ? -a.value() : a.value();
  return numeric::decimal<R>{numeric::scaled_integer<R>{rep, a.scale()}};
}

/**
 * @brief Computes sum of two values.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ auto add(A a, B b) -> decltype(a + b)
{
  return a + b;
}

/**
 * @brief Computes quotient of two values.
 *
 * @tparam A Dividend type.
 * @tparam B Divisor type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename A, typename B>
__device__ auto div(A a, B b) -> decltype(a / b)
{
  return a / b;
}

/**
 * @brief Computes floor division of two values.
 *
 * @tparam A Dividend type.
 * @tparam B Divisor type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <integer A, integer B>
__device__ auto floor_div(A a, B b) -> decltype(cudf::detail::integral_floor_div(a, b))
{
  return cudf::detail::integral_floor_div(a, b);
}

template <floating_point A, floating_point B>
__device__ auto floor_div(A a, B b) -> decltype(cuda::std::floor(a / b))
{
  return cuda::std::floor(a / b);
}

/**
 * @brief Computes remainder of two values.
 *
 * @tparam A Dividend type.
 * @tparam B Divisor type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename A, typename B>
__device__ auto mod(A a, B b) -> decltype(a % b)
{
  return a % b;
}

template <floating_point A, floating_point B>
__device__ auto mod(A a, B b) -> decltype(cuda::std::fmod(a, b))
{
  return cuda::std::fmod(a, b);
}

/**
 * @brief Computes Python-style modulus.
 *
 * @tparam A Dividend type.
 * @tparam B Divisor type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename A, typename B>
__device__ auto pymod(A a, B b) -> decltype((a % b + b) % b)
{
  return (a % b + b) % b;
}

template <floating_point A, floating_point B>
__device__ auto pymod(A a, B b) -> decltype(cuda::std::fmod(cuda::std::fmod(a, b) + b, b))
{
  return cuda::std::fmod(cuda::std::fmod(a, b) + b, b);
}

/**
 * @brief Computes product of two values.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ auto mul(A a, B b) -> decltype(a * b)
{
  return a * b;
}

/**
 * @brief Computes unary negation.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <typename T>
__device__ auto neg(T a) -> decltype(-a)
{
  return -a;
}

template <typename R>
__device__ numeric::decimal<R> neg(numeric::decimal<R> a)
{
  auto rep = -a.value();
  return numeric::decimal<R>{numeric::scaled_integer<R>{rep, a.scale()}};
}

/**
 * @brief Computes subtraction of two values.
 *
 * @tparam A Minuend type.
 * @tparam B Subtrahend type.
 * @param a Minuend.
 * @param b Subtrahend.
 */
template <typename A, typename B>
__device__ auto sub(A a, B b) -> decltype(a - b)
{
  return a - b;
}

/**
 * @brief Computes true division and returns a double.
 *
 * @tparam A Dividend type.
 * @tparam B Divisor type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename A, typename B>
__device__ auto true_div(A a, B b) -> decltype(static_cast<double>(a) / static_cast<double>(b))
{
  return static_cast<double>(a) / static_cast<double>(b);
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
