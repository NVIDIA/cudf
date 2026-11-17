/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/std/tuple>
#include <cuda/std/utility>

namespace cudf {
namespace jit {

/**
 * @brief A list of types with some helper functions to operate on them.
 *
 */
template <typename... T>
struct type_list {
  static constexpr int size = sizeof...(T);

  using tuple = cuda::std::tuple<T...>;

  template <int Index>
  using at = cuda::std::tuple_element_t<Index, tuple>;

  static constexpr cuda::std::make_integer_sequence<int, size> indexed{};

  template <typename Fn>
  static constexpr __device__ decltype(auto) map(Fn&& fn)
  {
    return fn.template operator()<T...>();
  }
};

}  // namespace jit
}  // namespace cudf
