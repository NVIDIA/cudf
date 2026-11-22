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
  bool is_retention;  ///< Whether `true` mask values retain or drop values

  __device__ bool operator()(size_type i) const noexcept { return mask[i] == is_retention; }
};

}  // namespace cudf::detail
