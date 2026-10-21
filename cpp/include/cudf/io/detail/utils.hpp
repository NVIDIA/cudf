/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace io::detail {
/**
 * @brief Whether writer writes in chunks or all at once
 */
enum class single_write_mode : bool { YES, NO };

template <typename T>
constexpr bool is_convertible_to_string_column()
{
  // Note: the case (not std::is_same_v<T, bool>)
  // is already covered by is_integral
  return std::is_same_v<T, cudf::string_view> || cuda::std::is_integral_v<T> ||
         std::is_floating_point_v<T> || cudf::is_fixed_point<T>() || cudf::is_timestamp<T>() ||
         cudf::is_duration<T>();
}

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
