/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>
#include <cuda/std/span>

namespace cudf {
namespace jit {

/** @brief Reads fixed-width list rows as spans of their device storage type. */
struct lists_column_device_view : private column_device_view_core {
  using base = column_device_view_core;
  using base::base;
  using base::is_null;
  using base::is_valid;
  using base::null_mask;
  using base::nullable;
  using base::offset;
  using base::size;
  using base::type;

  template <typename Span>
  [[nodiscard]] __device__ Span element(size_type row) const noexcept
  {
    auto offsets = child(0);
    auto begin   = offsets.element<size_type>(row + offset());
    auto end     = offsets.element<size_type>(row + offset() + 1);
    auto values  = child(1);
    auto data    = static_cast<typename Span::pointer>(values.head());
    if (values.offset() != 0) { data += values.offset(); }
    return {begin == 0 ? data : data + begin, static_cast<size_t>(end - begin)};
  }

  template <typename Span>
  [[nodiscard]] __device__ cuda::std::optional<Span> nullable_element(size_type row) const noexcept
  {
    if (is_null(row)) { return cuda::std::nullopt; }
    return element<Span>(row);
  }
};

/** @brief Writes list rows directly into storage allocated from supplied offsets. */
struct mutable_lists_column_device_view : private mutable_column_device_view_core {
  using base = mutable_column_device_view_core;
  using base::base;
  using base::is_null;
  using base::is_valid;
  using base::null_mask;
  using base::nullable;
  using base::offset;
  using base::size;
  using base::type;

  template <typename Span>
  [[nodiscard]] __device__ Span element(size_type row) const noexcept
  {
    auto offsets = child(0);
    auto begin   = offsets.element<size_type>(row + offset());
    auto end     = offsets.element<size_type>(row + offset() + 1);
    auto values  = child(1);
    auto data    = static_cast<typename Span::pointer>(const_cast<void*>(values.head()));
    if (values.offset() != 0) { data += values.offset(); }
    return {begin == 0 ? data : data + begin, static_cast<size_t>(end - begin)};
  }

  template <typename Span>
  __device__ void assign(size_type, Span) const noexcept
  {
    // The UDF writes directly into its allocated span.
  }
};

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

/**
 * @brief A column wrapper type that treats a column as a column of mutable strings.
 * The offsets will have been pre-initialized and the chars will have been pre-allocated.
 */
struct mutable_strings_column_device_view : private mutable_column_device_view_core {
  using base = mutable_column_device_view_core;

  CUDF_HOST_DEVICE constexpr mutable_strings_column_device_view(base const& src) : base{src} {}

  ~mutable_strings_column_device_view()                                         = default;
  mutable_strings_column_device_view(mutable_strings_column_device_view const&) = default;
  mutable_strings_column_device_view(mutable_strings_column_device_view&&)      = default;
  mutable_strings_column_device_view& operator=(mutable_strings_column_device_view const&) =
    default;
  mutable_strings_column_device_view& operator=(mutable_strings_column_device_view&&) = default;

  using base::is_null;
  using base::is_valid;
  using base::null_mask;
  using base::nullable;
  using base::offset;
  using base::size;
  using base::type;

  template <typename T = cuda::std::span<char>>
  [[nodiscard]] __device__ cuda::std::span<char> element(size_type element_index) const noexcept
    requires(cuda::std::is_same_v<T, cuda::std::span<char>>)
  {
    auto index             = element_index + offset();
    auto chars             = static_cast<char*>(const_cast<void*>(_data));
    auto offsets           = child(offsets_column_index);
    auto itr               = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
    auto beg               = itr[index];
    auto end               = itr[index + 1];
    auto* __restrict__ str = chars + beg;
    return cuda::std::span<char>{str, static_cast<size_t>(end - beg)};
  }

  template <typename T = cuda::std::span<char>>
  [[nodiscard]] __device__ cuda::std::optional<cuda::std::span<char>> nullable_element(
    size_type element_index) const noexcept
    requires(cuda::std::is_same_v<T, cuda::std::span<char>>)
  {
    if (is_null(element_index)) { return cuda::std::nullopt; }
    return element<T>(element_index);
  }

  template <typename T = cuda::std::span<char>>
  __device__ void assign(size_type row, cuda::std::span<char> value) const noexcept
    requires(cuda::std::is_same_v<T, cuda::std::span<char>>)
  {
    // no-op since we assume the chars have already been pre-allocated and they are mutated
    // in-place
    return;
  }
};

}  // namespace jit
}  // namespace cudf
