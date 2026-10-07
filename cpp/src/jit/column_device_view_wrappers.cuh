/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/strings/mutable_string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

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

  template <typename T = cudf::mutable_string_view>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
    requires(cuda::std::is_same_v<T, cudf::mutable_string_view>)
  {
    auto index             = element_index + offset();
    auto chars             = static_cast<char*>(const_cast<void*>(_data));
    auto offsets           = child(offsets_column_index);
    auto itr               = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
    auto beg               = itr[index];
    auto end               = itr[index + 1];
    auto* __restrict__ str = chars + beg;
    return mutable_string_view{str, static_cast<size_type>(end - beg)};
  }

  template <typename T = cudf::mutable_string_view>
  [[nodiscard]] __device__ cuda::std::optional<T> nullable_element(
    size_type element_index) const noexcept
    requires(cuda::std::is_same_v<T, cudf::mutable_string_view>)
  {
    if (is_null(element_index)) { return cuda::std::nullopt; }
    return element<T>(element_index);
  }

  template <typename T = cudf::mutable_string_view>
  __device__ void assign(size_type row, cudf::mutable_string_view value) const noexcept
    requires(cuda::std::is_same_v<T, cudf::mutable_string_view>)
  {
    // no-op since we assume the chars have already been pre-allocated and they are mutated
    // in-place
    return;
  }
};

/** @brief Reads fixed-width list rows as list_element views of their device storage type. */
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

  using base::element;
  using base::nullable_element;
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

  template <
    typename L,
    CUDF_ENABLE_IF(is_list_element<L>&& is_rep_layout_compatible<typename L::element_type>())>
  [[nodiscard]] __device__ L element(size_type element_index) const noexcept
  {
    auto const& offsets = child(list_offsets_column_index);
    auto const begin    = offsets.template element<size_type>(element_index + offset());
    auto const end      = offsets.template element<size_type>(element_index + offset() + 1);
    auto const& values  = child(list_values_column_index);
    return L{values.data<typename L::element_type>() + begin, end - begin};
  }

  template <typename T>
  [[nodiscard]] __device__ cuda::std::optional<T> nullable_element(
    size_type element_index) const noexcept
    requires(cudf::is_list_element<T>)
  {
    if (is_null(element_index)) { return cuda::std::nullopt; }
    return element<T>(element_index);
  }

  template <typename T>
  __device__ void assign(size_type, T) const noexcept
    requires(cudf::is_list_element<T>)
  {
    // The UDF writes directly into its allocated span.
  }
};

}  // namespace jit
}  // namespace cudf
