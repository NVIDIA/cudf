/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/type_traits>

#include <format>
#include <string>

namespace cudf::jit {
template <typename T>
  requires(cuda::std::is_integral_v<T> && sizeof(T) <= 16)
std::string nvvm_type_impl()
{
  return std::format("i{}", sizeof(T) * 8);
}

template <typename T>
  requires(cuda::std::is_floating_point_v<T> && sizeof(T) <= 8)
std::string nvvm_type_impl()
{
  if constexpr (sizeof(T) == 2) {
    return "half";
  } else if constexpr (sizeof(T) == 4) {
    return "float";
  } else {
    return "double";
  }
}

template <typename T>
  requires(cuda::std::is_pointer_v<T>)
std::string nvvm_type_impl()
{
  return "i8*";
}

template <typename T>
  requires(cuda::std::is_class_v<T> && alignof(T) <= 16)
std::string nvvm_type_impl()
{
  return std::format("[{} x i{}]", sizeof(T) / alignof(T), alignof(T) * 8);
}

template <typename T>
auto nvvm_type() -> decltype(nvvm_type_impl<cuda::std::remove_cv_t<T>>())
{
  return nvvm_type_impl<cuda::std::remove_cv_t<T>>();
}

}  // namespace cudf::jit
