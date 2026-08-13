/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/algorithm>

#include <cstdint>

/**
 * @file
 * @brief Class definition for `cudf::binary_view`.
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief A non-owning, immutable view of a variable-length sequence of bytes.
 *
 * The caller must maintain the device memory for the lifetime of this object.
 * Binary values have no character encoding; comparisons use unsigned-byte
 * lexicographic ordering.
 */
class binary_view {
 public:
  using value_type     = uint8_t;
  using const_iterator = value_type const*;

  CUDF_HOST_DEVICE constexpr binary_view() = default;

  /**
   * @brief Constructs a view over `size` bytes beginning at `data`.
   */
  CUDF_HOST_DEVICE constexpr binary_view(value_type const* data, size_type size)
    : _data{data}, _size{size}
  {
  }

  /**
   * @brief Returns the number of bytes in this value.
   */
  CUDF_HOST_DEVICE [[nodiscard]] constexpr size_type size_bytes() const noexcept { return _size; }

  /**
   * @brief Returns a pointer to the first byte.
   */
  CUDF_HOST_DEVICE [[nodiscard]] constexpr value_type const* data() const noexcept
  {
    return _data;
  }

  /**
   * @brief Returns whether this value contains no bytes.
   */
  CUDF_HOST_DEVICE [[nodiscard]] constexpr bool empty() const noexcept { return _size == 0; }

  CUDF_HOST_DEVICE [[nodiscard]] constexpr const_iterator begin() const noexcept { return _data; }
  CUDF_HOST_DEVICE [[nodiscard]] constexpr const_iterator end() const noexcept
  {
    return _data + _size;
  }

  /**
   * @brief Returns the byte at `index`.
   */
  CUDF_HOST_DEVICE [[nodiscard]] constexpr value_type operator[](size_type index) const noexcept
  {
    return _data[index];
  }

  /**
   * @brief Compares values using unsigned-byte lexicographic ordering.
   */
  CUDF_HOST_DEVICE [[nodiscard]] constexpr int compare(binary_view rhs) const noexcept
  {
    auto const count = cuda::std::min(_size, rhs._size);
    for (size_type i = 0; i < count; ++i) {
      if (_data[i] < rhs._data[i]) { return -1; }
      if (_data[i] > rhs._data[i]) { return 1; }
    }
    return (_size > rhs._size) - (_size < rhs._size);
  }

  CUDF_HOST_DEVICE [[nodiscard]] constexpr bool operator==(binary_view rhs) const noexcept
  {
    return compare(rhs) == 0;
  }

  CUDF_HOST_DEVICE [[nodiscard]] constexpr bool operator!=(binary_view rhs) const noexcept
  {
    return not(*this == rhs);
  }

  CUDF_HOST_DEVICE [[nodiscard]] constexpr bool operator<(binary_view rhs) const noexcept
  {
    return compare(rhs) < 0;
  }

  CUDF_HOST_DEVICE [[nodiscard]] constexpr bool operator>(binary_view rhs) const noexcept
  {
    return compare(rhs) > 0;
  }

  CUDF_HOST_DEVICE [[nodiscard]] constexpr bool operator<=(binary_view rhs) const noexcept
  {
    return compare(rhs) <= 0;
  }

  CUDF_HOST_DEVICE [[nodiscard]] constexpr bool operator>=(binary_view rhs) const noexcept
  {
    return compare(rhs) >= 0;
  }

 private:
  value_type const* _data{};
  size_type _size{};
};

}  // namespace CUDF_EXPORT cudf
