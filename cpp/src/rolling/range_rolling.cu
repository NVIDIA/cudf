/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/range_rolling.hpp"
#include "detail/rolling.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

namespace CUDF_EXPORT cudf {
namespace detail {

rmm::device_uvector<cudf::size_type> nulls_per_group(column_view const& orderby,
                                                     rmm::device_uvector<size_type> const& offsets,
                                                     rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  auto d_orderby        = column_device_view::create(orderby, stream);
  auto const num_groups = offsets.size() - 1;
  std::size_t bytes{0};
  auto is_null_it = cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0},
    cuda::proclaim_return_type<size_type>(
      [orderby = *d_orderby] __device__(size_type i) -> size_type {
        return static_cast<size_type>(orderby.is_null_nocheck(i));
      }));
  rmm::device_uvector<cudf::size_type> null_counts{num_groups, stream};
  cub::DeviceSegmentedReduce::Sum(nullptr,
                                  bytes,
                                  is_null_it,
                                  null_counts.begin(),
                                  num_groups,
                                  offsets.begin(),
                                  offsets.begin() + 1,
                                  stream.value());
  auto tmp = rmm::device_buffer(bytes, stream);
  cub::DeviceSegmentedReduce::Sum(tmp.data(),
                                  bytes,
                                  is_null_it,
                                  null_counts.begin(),
                                  num_groups,
                                  offsets.begin(),
                                  offsets.begin() + 1,
                                  stream.value());
  return null_counts;
}

std::unique_ptr<column> make_range_window(
  column_view const& orderby,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  rolling::direction direction,
  order order,
  null_order null_order,
  range_window_type window,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  bool const nulls_at_start = (order == order::ASCENDING && null_order == null_order::BEFORE) ||
                              (order == order::DESCENDING && null_order == null_order::AFTER);

  return std::visit(
    [&](auto const& window_bound) -> std::unique_ptr<column> {
      return dispatch_range_window(window_bound,
                                   orderby,
                                   direction,
                                   order,
                                   grouping,
                                   nulls_at_start,
                                   window_bound.delta(),
                                   stream,
                                   mr);
    },
    window);
}

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  null_order null_order,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (group_keys.num_columns() > 0) {
    using sort_helper = cudf::groupby::detail::sort::sort_groupby_helper;
    sort_helper helper{group_keys, null_policy::INCLUDE, sorted::YES, {}};
    auto const& labels   = helper.group_labels(stream);
    auto const& offsets  = helper.group_offsets(stream);
    auto per_group_nulls = orderby.has_nulls() ? nulls_per_group(orderby, offsets, stream)
                                               : rmm::device_uvector<size_type>{0, stream};
    auto grouping = detail::rolling::preprocessed_group_info{labels, offsets, per_group_nulls};
    return {
      make_range_window(
        orderby, grouping, rolling::direction::PRECEDING, order, null_order, preceding, stream, mr),
      make_range_window(orderby,
                        grouping,
                        rolling::direction::FOLLOWING,
                        order,
                        null_order,
                        following,
                        stream,
                        mr)};
  } else {
    return {make_range_window(orderby,
                              std::nullopt,
                              rolling::direction::PRECEDING,
                              order,
                              null_order,
                              preceding,
                              stream,
                              mr),
            make_range_window(orderby,
                              std::nullopt,
                              rolling::direction::FOLLOWING,
                              order,
                              null_order,
                              following,
                              stream,
                              mr)};
  }
}

}  // namespace detail

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  null_order null_order,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    group_keys.num_columns() == 0 || group_keys.num_rows() == orderby.size(),
    "If a grouping table is provided, it must have same number of rows as the orderby column.");
  return detail::make_range_windows(
    group_keys, orderby, order, null_order, preceding, following, stream, mr);
}

}  // namespace CUDF_EXPORT cudf
