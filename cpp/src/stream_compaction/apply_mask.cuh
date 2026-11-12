/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

namespace cudf::detail {

/**
 * @brief Functor that returns whether a boolean mask keeps the element at a given index
 *
 * @tparam MaskIterator Iterator over the mask values with null elements already replaced
 */
template <typename MaskIterator>
struct mask_filter {
  MaskIterator mask;  ///< Iterator over the mask values
  bool is_retention;  ///< Whether `true` mask values keep (retention) or drop (deletion) elements

  /**
   * @brief Returns whether the mask keeps the element at index `i`
   *
   * @param i Index of the element
   * @return `true` if the element is kept, `false` otherwise
   */
  __device__ inline bool operator()(size_type i) const noexcept { return mask[i] == is_retention; }
};

}  // namespace cudf::detail
