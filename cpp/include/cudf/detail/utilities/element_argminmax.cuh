/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/type_traits>

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
  /// Order NaN after all other values (`nan_semantics::TOTALLY_ORDERED`) for argmin, instead of
  /// selecting the first NaN (`nan_semantics::UNORDERED`)
  bool const totally_ordered = false;

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
    } else if constexpr (cuda::std::is_floating_point_v<T>) {
      // Select the row holding the MIN/MAX result, with -0 less than +0: the first NaN row for
      // argmax and for UNORDERED argmin (see ieee_extremum_precedes). With TOTALLY_ORDERED, NaN is
      // greater than every other value, which only changes argmin: a NaN row is selected only
      // when every row is NaN (see totally_ordered_minimum_precedes).
      auto const precedes = [&](T a, T b) {
        if (!arg_min) { return cudf::detail::ieee_extremum_precedes<false>(a, b); }
        return totally_ordered ? cudf::detail::totally_ordered_minimum_precedes(a, b)
                               : cudf::detail::ieee_extremum_precedes<true>(a, b);
      };
      if (precedes(lhs, rhs)) { return lhs_idx; }
      if (precedes(rhs, lhs)) { return rhs_idx; }
      return lhs_idx < rhs_idx ? lhs_idx : rhs_idx;
    } else {
      if (lhs == rhs) { return lhs_idx < rhs_idx ? lhs_idx : rhs_idx; }
      return (lhs < rhs) == arg_min ? lhs_idx : rhs_idx;
    }
  }
};

}  // namespace detail
}  // namespace cudf
