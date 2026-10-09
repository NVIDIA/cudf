/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/utilities/traits.hpp>

#include <memory>

namespace cudf {
namespace detail {

/**
 * @brief Returns true if the top-k of `col` can be selected with `cub::DeviceTopK`
 */
[[nodiscard]] inline bool is_cub_top_k_supported(column_view const& col)
{
  return not col.has_nulls() and cudf::is_fixed_width(col.type()) and
         not cudf::is_floating_point(col.type());  // needs special NaN handling
}

/**
 * @brief Dispatches `fn<KeyT>()` with the `cub::DeviceTopK` key type for each supported type
 *
 * @tparam Fn Callable with a templated call operator returning a `std::unique_ptr<column>`
 */
template <typename Fn>
struct dispatch_cub_top_k_key {
  Fn fn;

  template <typename T>
  std::unique_ptr<column> operator()()
  {
    if constexpr (cudf::is_chrono<T>()) {
      return fn.template operator()<typename T::rep>();
    } else if constexpr (cudf::is_fixed_width<T>() and not cudf::is_floating_point<T>()) {
      return fn.template operator()<T>();
    } else {
      CUDF_UNREACHABLE("unexpected type for the cub::DeviceTopK path");
    }
  }
};

}  // namespace detail
}  // namespace cudf
