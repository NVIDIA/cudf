/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "contains_table_impl.cuh"

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/detail/search.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>
#include <cuda/stream>

#include <algorithm>

namespace cudf::detail {

rmm::device_uvector<bool> contains(table_view const& haystack,
                                   table_view const& needles,
                                   null_equality compare_nulls,
                                   nan_equality compare_nans,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::have_same_types(haystack, needles), "Column types mismatch");

  auto const haystack_has_nulls = has_nested_nulls(haystack);
  auto const needles_has_nulls  = has_nested_nulls(needles);
  auto const has_any_nulls      = haystack_has_nulls || needles_has_nulls;

  auto const preprocessed_needles =
    cudf::detail::row::equality::preprocessed_table::create(needles, stream);
  auto const preprocessed_haystack =
    cudf::detail::row::equality::preprocessed_table::create(haystack, stream);

  // The output vector.
  auto contained = rmm::device_uvector<bool>(needles.num_rows(), stream, mr);

  // Only use primitive row operators for non-floating-point types since they don't handle NaN
  // equality
  auto const has_floating_point =
    std::any_of(haystack.begin(), haystack.end(), [](auto const& col) {
      return cudf::is_floating_point(col.type());
    });
  if (cudf::detail::is_primitive_row_op_compatible(haystack) && !has_floating_point) {
    auto const d_haystack_hasher = cudf::detail::row::primitive::row_hasher{
      nullate::DYNAMIC{has_any_nulls}, preprocessed_haystack};
    auto const d_needle_hasher = cudf::detail::row::primitive::row_hasher{
      nullate::DYNAMIC{has_any_nulls}, preprocessed_needles};
    auto const d_hasher     = hasher_adapter{d_haystack_hasher, d_needle_hasher};
    auto const d_self_equal = cudf::detail::row::primitive::row_equality_comparator{
      nullate::DYNAMIC{has_any_nulls}, preprocessed_haystack, preprocessed_haystack, compare_nulls};
    auto const d_two_table_equal = cudf::detail::row::primitive::row_equality_comparator{
      nullate::DYNAMIC{has_any_nulls}, preprocessed_needles, preprocessed_haystack, compare_nulls};
    auto const d_equal = comparator_adapter{d_self_equal, d_two_table_equal};
    perform_contains(haystack,
                     needles,
                     haystack_has_nulls,
                     needles_has_nulls,
                     compare_nulls,
                     d_equal,
                     cuco::linear_probing<1, decltype(d_hasher)>{d_hasher},
                     contained,
                     stream);
  } else {
    auto const haystack_hasher   = cudf::detail::row::hash::row_hasher(preprocessed_haystack);
    auto const d_haystack_hasher = haystack_hasher.device_hasher(nullate::DYNAMIC{has_any_nulls});
    auto const needle_hasher     = cudf::detail::row::hash::row_hasher(preprocessed_needles);
    auto const d_needle_hasher   = needle_hasher.device_hasher(nullate::DYNAMIC{has_any_nulls});
    auto const d_hasher          = hasher_adapter{d_haystack_hasher, d_needle_hasher};

    auto const self_equal = cudf::detail::row::equality::self_comparator(preprocessed_haystack);
    auto const two_table_equal = cudf::detail::row::equality::two_table_comparator(
      preprocessed_needles, preprocessed_haystack);

    if (cudf::detail::has_nested_columns(haystack)) {
      dispatch_nan_comparator<true>(haystack,
                                    needles,
                                    compare_nulls,
                                    compare_nans,
                                    haystack_has_nulls,
                                    needles_has_nulls,
                                    has_any_nulls,
                                    self_equal,
                                    two_table_equal,
                                    d_hasher,
                                    contained,
                                    stream);
    } else {
      dispatch_nan_comparator<false>(haystack,
                                     needles,
                                     compare_nulls,
                                     compare_nans,
                                     haystack_has_nulls,
                                     needles_has_nulls,
                                     has_any_nulls,
                                     self_equal,
                                     two_table_equal,
                                     d_hasher,
                                     contained,
                                     stream);
    }
  }
  return contained;
}

}  // namespace cudf::detail
