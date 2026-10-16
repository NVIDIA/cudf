/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>

#include <cuda/std/optional>

namespace cudf {
namespace jit {

/**
 * @brief A column wrapper type that treats a column as a vector of elements.
 *
 */
struct mutable_vector_device_view : private mutable_column_device_view_core {
  using base = mutable_column_device_view_core;

  CUDF_HOST_DEVICE constexpr mutable_vector_device_view(base const& src) : base{src} {}
  ~mutable_vector_device_view()                                            = default;
  mutable_vector_device_view(mutable_vector_device_view const&)            = default;
  mutable_vector_device_view(mutable_vector_device_view&&)                 = default;
  mutable_vector_device_view& operator=(mutable_vector_device_view const&) = default;
  mutable_vector_device_view& operator=(mutable_vector_device_view&&)      = default;

  using base::nullable;
  using base::offset;
  using base::size;
  using base::type;

  template <typename T>
  CUDF_HOST_DEVICE T* __restrict__ data() const noexcept
  {
    return static_cast<T*>(const_cast<void*>(_data)) + _offset;
  }

  using base::is_null;
  using base::is_valid;
  using base::null_mask;

  template <typename T>
  [[nodiscard]] __device__ decltype(auto) element(size_type element_index) const noexcept
  {
    return data<T>()[element_index];
  }

  template <typename T>
  [[nodiscard]] __device__ cuda::std::optional<T> nullable_element(
    size_type element_index) const noexcept
  {
    if (is_null(element_index)) { return cuda::std::nullopt; }
    return element<T>(element_index);
  }

  template <typename T>
  __device__ void assign(size_type row, T value) const noexcept
  {
    data<T>()[row] = value;
  }
};

}  // namespace jit
}  // namespace cudf
