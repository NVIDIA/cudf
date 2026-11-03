/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cstddef>

namespace cudf::detail {

/**
 * @brief Byte extent and alignment for sequentially placed objects.
 *
 * append() aligns each new object; aligned() adds aggregate tail padding for storage/array
 * stride. Finish a nested aggregate with aligned() before appending it to its parent.
 * The default layout is empty; memory_layout_of<T> includes sizeof(T)'s tail padding.
 */
struct memory_layout {
  size_t alignment = 1;  ///< Nonzero power-of-two base alignment
  size_t size      = 0;  ///< Extent before optional aggregate tail padding

  /// @brief Appends an already complete object layout at its next aligned offset.
  constexpr memory_layout append(memory_layout const& other) const
  {
    auto const new_alignment = std::max(alignment, other.alignment);
    auto const offset        = (size + other.alignment - 1) & ~(other.alignment - 1);
    return {.alignment = new_alignment, .size = offset + other.size};
  }

  /// @brief Includes tail padding so size is a multiple of alignment.
  constexpr memory_layout aligned() const
  {
    return {.alignment = alignment, .size = (size + alignment - 1) & ~(alignment - 1)};
  }
};

/// @brief Complete compiler-provided size and alignment of T, including empty classes.
template <typename T>
inline constexpr memory_layout memory_layout_of{.alignment = alignof(T), .size = sizeof(T)};

}  // namespace cudf::detail
