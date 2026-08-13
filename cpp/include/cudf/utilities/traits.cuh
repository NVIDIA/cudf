/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/atomic>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup utility_types
 * @{
 * @file
 * @brief Device-side type traits for querying atomic operation support on cudf column types.
 */

/**
 * @brief Indicates whether the type `T` has support for atomics
 *
 * @tparam T     The type to verify
 * @return true  `T` has support for atomics
 * @return false `T` no support for atomics
 */
template <typename T>
constexpr inline bool has_atomic_support()
{
  return cuda::std::atomic<T>::is_always_lock_free;
}

struct has_atomic_support_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return has_atomic_support<T>();
  }
};

/**
 * @brief Indicates whether `type` has support for atomics
 *
 * @param type   The `data_type` to verify
 * @return true  `type` has support for atomics
 * @return false `type` no support for atomics
 */
constexpr inline bool has_atomic_support(data_type type)
{
  return cudf::type_dispatcher(type, has_atomic_support_impl{});
}

/** @} */

}  // namespace CUDF_EXPORT cudf
