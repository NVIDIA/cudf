/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cudf {
namespace detail {

/**
 * @copydoc
 * sorted_order(table_view&,std::vector<order>,std::vector<null_order>,cuda::stream_ref,
 * memory_resources)
 *
 * @tparam stable Whether to use stable sort
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <sort_method method>
std::unique_ptr<column> sorted_order(table_view input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  auto const output_mr = mr.get_output_mr();
  auto const temp_mr   = mr.get_temporary_mr();
  if (input.num_rows() == 0 or input.num_columns() == 0) {
    return cudf::make_numeric_column(
      data_type(type_to_id<size_type>()), 0, mask_state::UNALLOCATED, stream, output_mr);
  }

  if (not column_order.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(input.num_columns()) == column_order.size(),
                 "Mismatch between number of columns and column order.");
  }

  if (not null_precedence.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(input.num_columns()) == null_precedence.size(),
                 "Mismatch between number of columns and null_precedence size.");
  }

  // fast-path for single column sort
  if (input.num_columns() == 1 and not cudf::is_nested(input.column(0).type())) {
    auto const single_col = input.column(0);
    auto const col_order  = column_order.empty() ? order::ASCENDING : column_order.front();
    auto const null_prec  = null_precedence.empty() ? null_order::BEFORE : null_precedence.front();
    return sorted_order<method>(single_col, col_order, null_prec, stream, mr);
  }

  std::unique_ptr<column> sorted_indices =
    cudf::make_numeric_column(data_type(type_to_id<size_type>()),
                              input.num_rows(),
                              mask_state::UNALLOCATED,
                              stream,
                              output_mr);
  mutable_column_view mutable_indices_view = sorted_indices->mutable_view();
  thrust::sequence(rmm::exec_policy_nosync(stream, temp_mr),
                   mutable_indices_view.begin<size_type>(),
                   mutable_indices_view.end<size_type>(),
                   0);

  auto const do_sort = [&](auto const comparator) {
    // Compiling `thrust::*sort*` APIs is expensive.
    // Thus, we should optimize that by using constexpr condition to only compile what we need.
    if constexpr (method == sort_method::STABLE) {
      thrust::stable_sort(rmm::exec_policy_nosync(stream, temp_mr),
                          mutable_indices_view.begin<size_type>(),
                          mutable_indices_view.end<size_type>(),
                          comparator);
    } else {
      thrust::sort(rmm::exec_policy_nosync(stream, temp_mr),
                   mutable_indices_view.begin<size_type>(),
                   mutable_indices_view.end<size_type>(),
                   comparator);
    }
  };

  // Preprocessing here allocates from the current device resource; the comparator does not
  // yet accept a memory resource.
  auto const comp =
    cudf::detail::row::lexicographic::self_comparator(input, column_order, null_precedence, stream);
  if (cudf::detail::has_nested_columns(input)) {
    auto const comparator = comp.less<true>(nullate::DYNAMIC{has_nested_nulls(input)});
    do_sort(comparator);
  } else {
    auto const comparator = comp.less<false>(nullate::DYNAMIC{has_nested_nulls(input)});
    do_sort(comparator);
  }

  return sorted_indices;
}

}  // namespace detail
}  // namespace cudf
