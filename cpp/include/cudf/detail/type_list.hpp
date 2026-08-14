/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/tuple>
#include <cuda/std/utility>

namespace cudf::detail {

/**
 * @brief A list of types with helpers for operating on the contained types.
 */
template <typename... T>
struct type_list {
  static constexpr int size = sizeof...(T);

  using tuple = cuda::std::tuple<T...>;

  template <int Index>
  using at = cuda::std::tuple_element_t<Index, tuple>;

  static constexpr cuda::std::make_integer_sequence<int, size> indexed{};

  template <typename Fn>
  static CUDF_HOST_DEVICE constexpr decltype(auto) map(Fn&& fn)
  {
    return fn.template operator()<T...>();
  }
};

}  // namespace cudf::detail
