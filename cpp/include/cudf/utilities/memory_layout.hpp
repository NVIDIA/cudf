/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cstddef>

namespace cudf::detail {

/**
 * @brief Represents the memory layout of an object.
 */
struct [[nodiscard]] memory_layout {
  size_t alignment = 1;  ///< Nonzero power-of-two base alignment
  size_t size      = 0;  ///< Extent before optional aggregate tail padding

  /// @brief Appends an already complete object layout at its next aligned offset.
  /// @param other Memory layout of the object to append
  /// @return Memory layout after appending the object (excluding tail padding)
  constexpr memory_layout append(memory_layout const& other) const
  {
    auto const offset = (size + other.alignment - 1) & ~(other.alignment - 1);
    return {.alignment = std::max(alignment, other.alignment), .size = offset + other.size};
  }

  /// @brief Includes tail padding so size is a multiple of alignment.
  /// @return Memory layout with tail padding included
  constexpr memory_layout aligned() const
  {
    return {.alignment = alignment, .size = (size + alignment - 1) & ~(alignment - 1)};
  }
};

/// @brief Get the memory layout of type T.
template <typename T>
inline constexpr memory_layout memory_layout_of{.alignment = alignof(T), .size = sizeof(T)};

}  // namespace cudf::detail
