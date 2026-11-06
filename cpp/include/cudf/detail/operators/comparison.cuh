/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/cmath>
#include <cuda/std/concepts>
#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Tests `a == b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a == b; })
{
  return a == b;
}

/**
 * @brief Tests `a != b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool not_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a != b; })
{
  return a != b;
}

/**
 * @brief Tests `a > b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool greater(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a > b; })
{
  return a > b;
}

/**
 * @brief Tests `a >= b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool greater_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a >= b; })
{
  return a >= b;
}

/**
 * @brief Tests `a < b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool less(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a < b; })
{
  return a < b;
}

/**
 * @brief Tests `a <= b`.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool less_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a <= b; })
{
  return a <= b;
}

/**
 * @brief Tests equality between two values for null-aware equality semantics.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool null_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a == b; })
{
  return a == b;
}

template <typename A, typename B>
__device__ bool null_equal(cuda::std::optional<A> a, cuda::std::optional<B> b)
  requires(!nullable<A> && !nullable<B> && requires { a == b; })
{
  return a == b;
}

// The functions below implement `nan_semantics::ETL` comparisons: NaNs are equal to each other
// and greater than every other value. Non-floating-point operands compare as usual.

/**
 * @brief Returns true if `value` is a floating-point NaN
 */
template <typename T>
__device__ inline bool is_nan_value(T value)
{
  if constexpr (cuda::std::floating_point<T>) {
    return cuda::std::isnan(value);
  } else {
    return false;
  }
}

template <typename A, typename B>
__device__ bool etl_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a == b; })
{
  auto const a_nan = is_nan_value(a);
  auto const b_nan = is_nan_value(b);
  return (a_nan || b_nan) ? (a_nan && b_nan) : (a == b);
}

template <typename A, typename B>
__device__ bool etl_not_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a != b; })
{
  return !etl_equal(a, b);
}

template <typename A, typename B>
__device__ bool etl_less(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a < b; })
{
  auto const a_nan = is_nan_value(a);
  auto const b_nan = is_nan_value(b);
  // With a NaN operand, a < b only when b is the NaN and a is not
  return (a_nan || b_nan) ? !a_nan : (a < b);
}

template <typename A, typename B>
__device__ bool etl_greater(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a > b; })
{
  return etl_less(b, a);
}

template <typename A, typename B>
__device__ bool etl_less_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a <= b; })
{
  return !etl_less(b, a);
}

template <typename A, typename B>
__device__ bool etl_greater_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a >= b; })
{
  return !etl_less(a, b);
}

template <typename A, typename B>
__device__ bool etl_null_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a == b; })
{
  return etl_equal(a, b);
}

template <typename A, typename B>
__device__ bool etl_null_equal(cuda::std::optional<A> a, cuda::std::optional<B> b)
  requires(!nullable<A> && !nullable<B> && requires { a == b; })
{
  if (a.has_value() && b.has_value()) { return etl_equal(*a, *b); }
  return a.has_value() == b.has_value();
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
