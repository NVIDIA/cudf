/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/concepts>
#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

template <typename T>
concept unsigned_integer = concepts::integral_not_bool<T> && cuda::std::unsigned_integral<T>;

template <typename T>
constexpr bool is_nullable = false;

template <typename T>
constexpr bool is_nullable<cuda::std::optional<T>> = true;

template <typename T>
concept nullable = is_nullable<T>;

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
