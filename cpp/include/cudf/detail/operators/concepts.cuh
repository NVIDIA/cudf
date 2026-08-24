/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/concepts>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

template <typename T>
concept integer =
  cuda::std::is_integral_v<T> && !cuda::std::is_same_v<cuda::std::remove_cv_t<T>, bool>;

template <typename T>
concept signed_integer = integer<T> && cuda::std::is_signed_v<cuda::std::remove_cv_t<T>>;

template <typename T>
concept unsigned_integer = integer<T> && cuda::std::is_unsigned_v<cuda::std::remove_cv_t<T>>;

template <typename T>
concept fixed_point = cudf::is_fixed_point<T>();

template <typename T>
concept floating_point = cuda::std::is_floating_point_v<T>;

template <typename T>
constexpr bool is_nullable = false;

template <typename T>
constexpr bool is_nullable<cuda::std::optional<T>> = true;

template <typename T>
concept nullable = is_nullable<T>;

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
