/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_impl.cuh"
#include "sort_radix.hpp"

#include <cudf/column/column.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {
std::unique_ptr<column> sorted_order(table_view const& input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  return sorted_order<sort_method::UNSTABLE>(input, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> sort_by_key(table_view const& values,
                                   table_view const& keys,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(values.num_rows() == keys.num_rows(),
               "Mismatch in number of rows for values and keys");

  auto sorted_order = detail::sorted_order(
    keys, column_order, null_precedence, stream, cudf::get_current_device_resource_ref());

  return detail::gather(values,
                        sorted_order->view(),
                        out_of_bounds_policy::DONT_CHECK,
                        negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

std::unique_ptr<table> sort(table_view const& input,
                            std::vector<order> const& column_order,
                            std::vector<null_order> const& null_precedence,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr)
{
  // fast-path sort conditions: single, fixed-width column with no nulls
  if (input.num_columns() == 1 && is_radix_sortable(input.column(0))) {
    auto order  = (column_order.empty() ? order::ASCENDING : column_order.front());
    auto output = sort_radix(input.column(0), order == order::ASCENDING, stream, mr);
    std::vector<std::unique_ptr<column>> columns;
    columns.emplace_back(std::move(output));
    return std::make_unique<table>(std::move(columns));
  }
  return detail::sort_by_key(input, input, column_order, null_precedence, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> sorted_order(table_view const& input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sorted_order(input, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> sort(table_view const& input,
                            std::vector<order> const& column_order,
                            std::vector<null_order> const& null_precedence,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort(input, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> sort_by_key(table_view const& values,
                                   table_view const& keys,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort_by_key(values, keys, column_order, null_precedence, stream, mr);
}

}  // namespace cudf
