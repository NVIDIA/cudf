/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>
#include <cuda/std/type_traits>

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

/**
 * @brief Tests inequality, treating two null values as equal.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool null_not_equal(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a != b; })
{
  return a != b;
}

template <typename A, typename B>
__device__ bool null_not_equal(cuda::std::optional<A> a, cuda::std::optional<B> b)
  requires(!nullable<A> && !nullable<B> && requires { null_equal(a, b); })
{
  return !null_equal(a, b);
}

namespace detail {

template <bool Maximum, typename A, typename B, typename C = cuda::std::common_type_t<A, B>>
__device__ C extreme(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires(C lhs, C rhs) {
    lhs < rhs;
    lhs > rhs;
  })
{
  auto const lhs = static_cast<C>(a);
  auto const rhs = static_cast<C>(b);
  if constexpr (Maximum) {
    return lhs > rhs ? lhs : rhs;
  } else {
    return lhs < rhs ? lhs : rhs;
  }
}

template <bool Maximum, typename A, typename B, typename C = cuda::std::common_type_t<A, B>>
__device__ cuda::std::optional<C> null_extreme(cuda::std::optional<A> a, cuda::std::optional<B> b)
  requires requires { extreme<Maximum>(a.value(), b.value()); }
{
  if (a.has_value() && b.has_value()) { return extreme<Maximum>(*a, *b); }
  if (a.has_value()) { return static_cast<C>(*a); }
  if (b.has_value()) { return static_cast<C>(*b); }
  return cuda::std::nullopt;
}

}  // namespace detail

/**
 * @brief Returns the greater operand, or the non-null operand when only one is valid.
 *
 * Two null operands produce null. Equal or unordered valid operands select the right operand.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ auto null_max(A a, B b) -> decltype(detail::extreme<true>(a, b))
{
  return detail::extreme<true>(a, b);
}

template <typename A, typename B>
__device__ auto null_max(cuda::std::optional<A> a, cuda::std::optional<B> b)
  -> decltype(detail::null_extreme<true>(a, b))
{
  return detail::null_extreme<true>(a, b);
}

/**
 * @brief Returns the lesser operand, or the non-null operand when only one is valid.
 *
 * Two null operands produce null. Equal or unordered valid operands select the right operand.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ auto null_min(A a, B b) -> decltype(detail::extreme<false>(a, b))
{
  return detail::extreme<false>(a, b);
}

template <typename A, typename B>
__device__ auto null_min(cuda::std::optional<A> a, cuda::std::optional<B> b)
  -> decltype(detail::null_extreme<false>(a, b))
{
  return detail::null_extreme<false>(a, b);
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
