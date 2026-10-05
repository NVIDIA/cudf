/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/cmath>
#include <cuda/std/concepts>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Computes inverse cosine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T arccos(T a)
{ return cuda::std::acos(a); }

/**
 * @brief Computes inverse hyperbolic cosine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T arccosh(T a)
{ return cuda::std::acosh(a); }

/**
 * @brief Computes inverse sine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T arcsin(T a)
{ return cuda::std::asin(a); }

/**
 * @brief Computes inverse hyperbolic sine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T arcsinh(T a)
{ return cuda::std::asinh(a); }

/**
 * @brief Computes inverse tangent.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T arctan(T a)
{ return cuda::std::atan(a); }

/**
 * @brief Computes inverse hyperbolic tangent.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T arctanh(T a)
{ return cuda::std::atanh(a); }

/**
 * @brief Computes cosine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T cos(T a)
{ return cuda::std::cos(a); }

/**
 * @brief Computes hyperbolic cosine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T cosh(T a)
{ return cuda::std::cosh(a); }

/**
 * @brief Computes sine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T sin(T a)
{ return cuda::std::sin(a); }

/**
 * @brief Computes hyperbolic sine.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T sinh(T a)
{ return cuda::std::sinh(a); }

/**
 * @brief Computes tangent.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T tan(T a)
{ return cuda::std::tan(a); }

/**
 * @brief Computes hyperbolic tangent.
 *
 * @tparam T Value type
 * @param a Input value.
 */
template <floating_point T>
__device__ T tanh(T a)
{ return cuda::std::tanh(a); }

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
