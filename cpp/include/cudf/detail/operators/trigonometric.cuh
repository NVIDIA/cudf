/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/cmath>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Computes inverse cosine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T arccos(T a)
{
  return cuda::std::acos(a);
}

/**
 * @brief Computes inverse hyperbolic cosine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T arccosh(T a)
{
  return cuda::std::acosh(a);
}

/**
 * @brief Computes inverse sine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T arcsin(T a)
{
  return cuda::std::asin(a);
}

/**
 * @brief Computes inverse hyperbolic sine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T arcsinh(T a)
{
  return cuda::std::asinh(a);
}

/**
 * @brief Computes inverse tangent.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T arctan(T a)
{
  return cuda::std::atan(a);
}

/**
 * @brief Computes inverse hyperbolic tangent.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T arctanh(T a)
{
  return cuda::std::atanh(a);
}

/**
 * @brief Computes cosine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T cos(T a)
{
  return cuda::std::cos(a);
}

/**
 * @brief Computes hyperbolic cosine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T cosh(T a)
{
  return cuda::std::cosh(a);
}

/**
 * @brief Computes sine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T sin(T a)
{
  return cuda::std::sin(a);
}

/**
 * @brief Computes hyperbolic sine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T sinh(T a)
{
  return cuda::std::sinh(a);
}

/**
 * @brief Computes tangent.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T tan(T a)
{
  return cuda::std::tan(a);
}

/**
 * @brief Computes hyperbolic tangent.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <cuda::std::floating_point T>
__device__ T tanh(T a)
{
  return cuda::std::tanh(a);
}

/**
 * @brief Computes the quadrant-aware inverse tangent of a divided by b, returning double.
 *
 * @tparam A Numerator type.
 * @tparam B Denominator type.
 * @param a Numerator.
 * @param b Denominator.
 */
template <typename A, typename B>
__device__ double arctan2(A a, B b)
  requires(cuda::std::is_convertible_v<A, double> && cuda::std::is_convertible_v<B, double>)
{
  return cuda::std::atan2(static_cast<double>(a), static_cast<double>(b));
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
