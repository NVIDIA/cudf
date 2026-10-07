/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf::groupby::detail {

/**
 * @brief Value accessor for column which supports dictionary column too.
 *
 * This is similar to `value_accessor` in `column_device_view.cuh` but with support of dictionary
 * type.
 *
 * @tparam T Type of the underlying column. For dictionary column, type of the key column.
 */
template <typename T>
struct value_accessor {
  column_device_view col;
  bool is_dict;

  value_accessor(column_device_view const& col) : col(col), is_dict(cudf::is_dictionary(col.type()))
  {
  }

  __device__ T value(size_type i) const
  {
    if (is_dict) {
      auto keys = col.child(dictionary_column_view::keys_column_index);
      return keys.element<T>(static_cast<size_type>(col.element<dictionary32>(i)));
    } else {
      return col.element<T>(i);
    }
  }

  __device__ auto operator()(size_type i) const { return value(i); }
};

}  // namespace cudf::groupby::detail
