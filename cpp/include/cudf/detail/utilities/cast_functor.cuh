/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @brief A casting functor wrapping another functor.
 * @file
 */

#include <cudf/types.hpp>

#include <cuda/functional>

#include <type_traits>
#include <utility>

namespace cudf {
namespace detail {

/**
 * @brief Functor that casts a primitive input value to a specified type
 */
template <typename T>
struct cast_fn {
  template <typename U>
  CUDF_HOST_DEVICE constexpr T operator()(U&& val) const
  { return static_cast<T>(cuda::std::forward<U>(val)); }

  CUDF_HOST_DEVICE constexpr T&& operator()(T&& val) const noexcept
  { return cuda::std::forward<T>(val); }
};

/**
 * @brief Functor that casts another functor's result to a specified type.
 *
 * CUB 2.0.0 reductions require that the binary operator returns the same type
 * as the initial value type, so we wrap binary operators with this when used
 * by CUB.
 */
template <typename ResultType, typename F>
struct cast_functor_fn {
  F f;

  template <typename... Ts>
  CUDF_HOST_DEVICE inline ResultType operator()(Ts&&... args)
  { return static_cast<ResultType>(f(std::forward<Ts>(args)...)); }
};

/**
 * @brief Function creating a casting functor.
 */
template <typename ResultType, typename F>
inline cast_functor_fn<ResultType, std::decay_t<F>> cast_functor(F&& f)
{ return cast_functor_fn<ResultType, std::decay_t<F>>{std::forward<F>(f)}; }

}  // namespace detail

}  // namespace cudf
