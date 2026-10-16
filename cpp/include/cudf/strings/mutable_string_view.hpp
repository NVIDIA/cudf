/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/string_view.hpp>

namespace CUDF_EXPORT cudf {

/**
 * @brief A non-owning, mutable view of device data representing a UTF-8 string.
 *
 * @ingroup strings_classes
 *
 * The caller must maintain the device memory for the lifetime of this instance.
 * Writes must stay within the existing byte range and preserve valid UTF-8 encoding.
 * Access to the device memory must occur in a kernel.
 */
class mutable_string_view {
 public:
  /** @brief Default constructor represents an empty string. */
  CUDF_HOST_DEVICE mutable_string_view() {}

  /**
   * @brief Create a view of an existing device char array.
   *
   * @param data Device char array encoded in UTF-8
   * @param bytes Number of bytes in the array
   */
  CUDF_HOST_DEVICE mutable_string_view(char* data, size_type bytes) : _data{data}, _bytes{bytes} {}

  /**
   * @brief Return a pointer to the mutable device array.
   * @return A pointer to the mutable device array
   */
  CUDF_HOST_DEVICE [[nodiscard]] char* data() const { return _data; }

  /**
   * @brief Return the number of bytes in this string.
   * @return The number of bytes in this string
   */
  CUDF_HOST_DEVICE [[nodiscard]] size_type size_bytes() const { return _bytes; }

  /**
   * @brief Return the number of UTF-8 characters in this string.
   *
   * The stored character count is refreshed on each call to account for mutations
   * through the device pointer returned by data().
   *
   * @return The number of UTF-8 characters in this string
   */
  [[nodiscard]] __device__ inline size_type length() const;

  /**
   * @brief Return whether this string is empty.
   * @return true if the string has no bytes
   */
  CUDF_HOST_DEVICE [[nodiscard]] bool empty() const { return size_bytes() == 0; }

  /**
   * @brief Return an immutable view of the same string.
   * @return A string_view of the same device array and byte range
   */
  CUDF_HOST_DEVICE operator string_view() const { return string_view{_data, _bytes}; }

 private:
  char* _data{};                ///< Pointer to the device char array
  size_type _bytes{};           ///< Number of bytes in the array
  mutable size_type _length{};  ///< Most recently computed UTF-8 character count
};

}  // namespace CUDF_EXPORT cudf
