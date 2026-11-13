/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <type_traits>

namespace cudf {
namespace detail {

/**
 * @brief Binary `argmin`/`argmax` operator
 *
 * Equal values select the smaller row index, independently of reduction order.
 *
 * @tparam T Type of the underlying column. Must support '<' and '==' operators.
 */
template <typename T>
struct element_argminmax_fn {
  column_device_view const d_col;
  bool const has_nulls;
  bool const arg_min;

  __device__ inline bool out_of_bounds_or_null(size_type idx) const
  {
    // The extra bounds checking is due to issue github.com/NVIDIA/cudf/9156 and
    // github.com/NVIDIA/thrust/issues/1525
    // where invalid random values may be passed here by thrust::reduce_by_key
    return idx < 0 || idx >= d_col.size() || (has_nulls && d_col.is_null_nocheck(idx));
  }

  __attribute__((noinline)) __device__ size_type operator()(size_type lhs_idx,
                                                            size_type rhs_idx) const
  {
    if (out_of_bounds_or_null(lhs_idx)) { return rhs_idx; }
    if (out_of_bounds_or_null(rhs_idx)) { return lhs_idx; }

    auto const lhs = d_col.type().id() == type_id::DICTIONARY32
                       ? d_col.child(dictionary_column_view::keys_column_index)
                           .element<T>(d_col.element<dictionary32>(lhs_idx).value())
                       : d_col.element<T>(lhs_idx);
    auto const rhs = d_col.type().id() == type_id::DICTIONARY32
                       ? d_col.child(dictionary_column_view::keys_column_index)
                           .element<T>(d_col.element<dictionary32>(rhs_idx).value())
                       : d_col.element<T>(rhs_idx);

    if constexpr (std::is_same_v<T, string_view>) {
      // Ordering and equality share one comparison, including for dictionary keys.
      auto const comparison = lhs.compare(rhs);
      if (comparison == 0) { return lhs_idx < rhs_idx ? lhs_idx : rhs_idx; }
      return (comparison < 0) == arg_min ? lhs_idx : rhs_idx;
    } else {
      if (lhs == rhs) { return lhs_idx < rhs_idx ? lhs_idx : rhs_idx; }
      return (lhs < rhs) == arg_min ? lhs_idx : rhs_idx;
    }
  }
};

}  // namespace detail
}  // namespace cudf
