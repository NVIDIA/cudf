/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Computes logical AND.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool logical_and(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a && b; })
{ return a && b; }

/**
 * @brief Computes logical OR.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool logical_or(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { a || b; })
{ return a || b; }

/**
 * @brief Computes logical NOT.
 *
 * @tparam T Value type.
 * @param a Input operand.
 */
template <typename T>
__device__ bool logical_not(T a)
  requires(!nullable<T> && requires { !a; })
{ return !a; }

/**
 * @brief Computes logical AND with null-aware semantics.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool null_logical_and(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { logical_and(a, b); })
{ return logical_and(a, b); }

template <typename A, typename B>
__device__ cuda::std::optional<bool> null_logical_and(cuda::std::optional<A> a,
                                                      cuda::std::optional<B> b)
  requires(!nullable<A> && !nullable<B> && requires { null_logical_and(a.value(), b.value()); })
{
  if (a.has_value() && b.has_value()) {
    return null_logical_and(a.value(), b.value());
  } else if (!a.has_value() && !b.has_value()) {
    return {};
  } else {
    if (a.has_value() ? *a : *b) {
      return {};
    } else {
      return false;
    }
  }
}

/**
 * @brief Computes logical OR with null-aware semantics.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename A, typename B>
__device__ bool null_logical_or(A a, B b)
  requires(!nullable<A> && !nullable<B> && requires { logical_or(a, b); })
{ return logical_or(a, b); }

template <typename A, typename B>
__device__ cuda::std::optional<bool> null_logical_or(cuda::std::optional<A> a,
                                                     cuda::std::optional<B> b)
  requires(!nullable<A> && !nullable<B> && requires { null_logical_or(a.value(), b.value()); })
{
  if (a.has_value() && b.has_value()) {
    return null_logical_or(a.value(), b.value());
  } else if (!a.has_value() && !b.has_value()) {
    return {};
  } else {
    if (a.has_value() ? *a : *b) {
      return true;
    } else {
      return {};
    }
  }
}

/**
 * @brief Selects one of two values based on a predicate.
 *
 * @tparam A True value type.
 * @tparam B False value type.
 * @tparam Predicate Selection predicate type.
 * @param true_value Value selected when @p pred is true.
 * @param false_value Value selected when @p pred is false.
 * @param pred Selection predicate.
 */
template <typename A, typename B, cuda::std::integral Predicate>
__device__ A if_else(A true_value, B false_value, Predicate pred)
  requires(!nullable<A> && cuda::std::same_as<A, B>)
{ return pred ? true_value : false_value; }

template <typename A, typename B, cuda::std::integral Predicate>
__device__ cuda::std::optional<A> if_else(cuda::std::optional<A> true_value,
                                          cuda::std::optional<B> false_value,
                                          cuda::std::optional<Predicate> pred)
  requires(!nullable<A> && cuda::std::same_as<A, B>)
{ return pred.value_or(false) ? true_value : false_value; }

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
