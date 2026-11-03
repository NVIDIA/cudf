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
 * @brief Tests whether an input value is null.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <typename T>
__device__ bool is_null(T a)
  requires(!nullable<T>)
{
  return false;
}

template <typename T>
__device__ bool is_null(T a)
  requires(nullable<T>)
{
  return !a.has_value();
}

/**
 * @brief Returns the first non-null of two values.
 *
 * @tparam A First value type.
 * @tparam B Second value type.
 * @param a First value.
 * @param b Second value.
 */
template <typename A, typename B>
__device__ A coalesce(A a, B b)
  requires(!nullable<A> && cuda::std::same_as<A, B>)
{
  return a;
}

template <typename A, typename B>
__device__ cuda::std::optional<A> coalesce(cuda::std::optional<A> a, cuda::std::optional<B> b)
  requires(!nullable<A> && cuda::std::same_as<A, B>)
{
  if (a.has_value()) {
    return a.value();
  } else if (b.has_value()) {
    return b.value();
  } else {
    return {};
  }
}

/**
 * @brief Converts an optional predicate to a non-nullable predicate.
 *
 * @param a Input boolean predicate.
 */
template <cuda::std::same_as<bool> T>
__device__ inline bool predicate(T a)
{
  return a;
}

template <cuda::std::same_as<bool> T>
__device__ inline bool predicate(cuda::std::optional<T> a)
{
  return a.value_or(false);
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
