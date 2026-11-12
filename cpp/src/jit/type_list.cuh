/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cudf {
namespace jit {

/**
 * @brief A list of types with some helper functions to operate on them.
 *
 */
template <typename... T>
struct type_list {
  template <typename Fn>
  static constexpr __device__ decltype(auto) map(Fn&& fn)
  {
    return fn.template operator()<T...>();
  }
};

}  // namespace jit
}  // namespace cudf
