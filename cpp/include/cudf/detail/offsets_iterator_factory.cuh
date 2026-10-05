/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/offsets_iterator.cuh>

namespace cudf {
namespace detail {

/**
 * @brief Use this class to create an offsetalator instance.
 */
struct offsetalator_factory {
  /**
   * @brief Create an input offsetalator instance from an offsets column
   *
   * @param offsets Column to wrap with an offsetalator
   * @param offset Index value within `offsets` to use as the beginning of the iterator
   */
  static input_offsetalator make_input_iterator(column_view const& offsets, size_type offset = 0)
  { return input_offsetalator(offsets.head(), offsets.type(), offset); }

  /**
   * @brief Create an output offsetalator instance from an offsets column
   *
   * @param offsets Column to wrap with an offsetalator
   */
  static output_offsetalator make_output_iterator(mutable_column_view const& offsets)
  { return output_offsetalator(offsets.head(), offsets.type()); }
};

}  // namespace detail
}  // namespace cudf
