/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/span.hpp>

#include <cuda/std/limits>

namespace cudf::io::statistics {

/**
 * @brief Wrapper for a row of a list<int8> or list<uint8> column. This is analogous to
 * `string_view` in type. It was created due to the need for comparison operators for cub reduce on
 * statistics. Otherwise, it is a device_span in all but name.
 *
 */
class byte_array_view {
 public:
  using element_type = std::byte const;  ///< The type of the elements in the byte array

  CUDF_HOST_DEVICE constexpr byte_array_view() noexcept {}
  /**
   * @brief Constructs a byte_array_view from a pointer and a size.
   *
   * @param data Pointer to the first element in the byte array.
   * @param size The number of elements in the byte array.
   */
  CUDF_HOST_DEVICE constexpr byte_array_view(element_type* data, std::size_t size)
    : _data(data, size)
  {
  }
  CUDF_HOST_DEVICE constexpr byte_array_view(byte_array_view const&) noexcept =
    default;  ///< Copy constructor
  /**
   * @brief Copy assignment operator.
   *
   * @return Reference to this byte_array_view.
   */
  constexpr byte_array_view& operator=(byte_array_view const&) noexcept = default;

  /**
   * @brief Returns a reference to the idx-th element of the byte_array_view.
   *
   * The behavior is undefined if idx is out of range (i.e., if it is greater than or equal to
   * size()).
   *
   * @param idx The index of the element to access.
   * @return A reference to the idx-th element of the byte_array_view, i.e., `_data.data()[idx]`.
   */
  [[nodiscard]] __device__ constexpr element_type& operator[](std::size_t idx) const
  { return _data[idx]; }

  /**
   * @brief Returns a pointer to the beginning of the byte_array_view.
   *
   * @return A pointer to the first element of the byte_array_view.
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr element_type* data() const noexcept
  { return _data.data(); }

  /**
   * @brief Returns the number of elements in the byte_array_view.
   *
   * @return The number of elements in the byte_array_view.
   */
  [[nodiscard]] constexpr std::size_t size() const noexcept { return _data.size(); }

  /**
   * @brief Returns the size of the byte_array_view in bytes.
   *
   * @return The size of the byte_array_view in bytes
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr std::size_t size_bytes() const noexcept
  { return _data.size_bytes(); }

  /**
   * @brief Comparing target byte_array_view with this byte_array_view. Each byte in the array is
   * compared.
   *
   * @param rhs Target byte_array_view to compare with this byte_array_view.
   * @return 0  If they compare equal.
   *         <0 Either the value of the first byte of this byte_array_view that does not match is
   * lower in the arg byte_array_view, or all compared bytes match but the arg byte_array_view is
   * shorter. >0 Either the value of the first byte of this byte_array_view that does not match is
   * greater in the arg byte_array_view, or all compared bytes match but the arg byte_array_view is
   * longer.
   */
  [[nodiscard]] __device__ inline int32_t compare(byte_array_view const& rhs) const
  {
    auto const len1  = size_bytes();
    auto const len2  = rhs.size_bytes();
    auto const* ptr1 = this->data();
    auto const* ptr2 = rhs.data();
    if ((ptr1 == ptr2) && (len1 == len2)) { return 0; }
    // if I am max, I am greater than the argument
    if (ptr1 == nullptr && len1 == cuda::std::numeric_limits<std::size_t>::max()) { return 1; }
    // if the argument is max, it is greater than me
    if (ptr2 == nullptr && len2 == cuda::std::numeric_limits<std::size_t>::max()) { return -1; }
    std::size_t idx = 0;
    for (; (idx < len1) && (idx < len2); ++idx) {
      if (ptr1[idx] != ptr2[idx]) {
        return static_cast<int32_t>(ptr1[idx]) - static_cast<int32_t>(ptr2[idx]);
      }
    }
    // if the argument ran out of data, it is less than me
    if (idx < len1) return 1;
    // if I ran out of data first, I am less than the argument
    if (idx < len2) return -1;
    return 0;
  }

  /**
   * @brief Returns true if this byte_array_view is ordered before rhs.
   *
   * @param rhs Target byte_array_view to compare with this byte_array_view.
   * @return true if this byte_array_view is ordered before rhs
   */
  [[nodiscard]] __device__ inline bool operator<(byte_array_view const& rhs) const
  { return compare(rhs) < 0; }
  /**
   * @brief Returns true if rhs is ordered before this byte_array_view.
   *
   * @param rhs Target byte_array_view to compare with this byte_array_view.
   * @return true if rhs is ordered before this byte_array_view
   */
  [[nodiscard]] __device__ inline bool operator>(byte_array_view const& rhs) const
  { return compare(rhs) > 0; }

  /**
   * @brief Returns true if this byte_array_view is ordered before rhs.
   *
   * @param rhs Target byte_array_view to compare with this byte_array_view.
   * @return true if this byte_array_view is ordered before rhs
   */
  [[nodiscard]] __device__ inline bool operator<=(byte_array_view const& rhs) const
  { return compare(rhs) <= 0; }
  /**
   * @brief Returns true if rhs is ordered before this byte_array_view.
   *
   * @param rhs Target byte_array_view to compare with this byte_array_view.
   * @return true if rhs is ordered before this byte_array_view
   */
  [[nodiscard]] __device__ inline bool operator>=(byte_array_view const& rhs) const
  { return compare(rhs) >= 0; }

  /**
   * @brief Return minimum value associated with the byte_array_view type
   *
   * @return An empty byte_array_view
   */
  [[nodiscard]] __device__ inline static byte_array_view min() { return {}; }

  /**
   * @brief Return a byte_array_view to interpret as maximum value
   *
   * @return A byte_array_view value which represents the largest possible byte_array_view
   */
  [[nodiscard]] __device__ inline static byte_array_view max()
  { return {nullptr, cuda::std::numeric_limits<std::size_t>::max()}; }

 private:
  device_span<element_type> _data{};
};

}  // namespace cudf::io::statistics
