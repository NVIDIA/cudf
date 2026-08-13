/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/optimized_unbounded_window.hpp"
#include "detail/range_window_bounds.hpp"
#include "detail/rolling.cuh"
#include "detail/rolling_udf.cuh"
#include "detail/rolling_utils.cuh"

#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <cuda/std/functional>

#include <concepts>
#include <span>

namespace cudf {

namespace detail {

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               column_view const& default_outputs,
                                               window_bounds preceding_window_bounds,
                                               window_bounds following_window_bounds,
                                               size_type min_periods,
                                               rolling_aggregation const& aggr,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) { return cudf::detail::empty_output_for_rolling_aggregation(input, aggr); }

  CUDF_EXPECTS((group_keys.num_columns() == 0 || group_keys.num_rows() == input.size()),
               "Size mismatch between group_keys and input vector.");

  CUDF_EXPECTS((min_periods >= 0), "min_periods must be non-negative");

  CUDF_EXPECTS((default_outputs.is_empty() || default_outputs.size() == input.size()),
               "Defaults column must be either empty or have as many rows as the input column.");

  // Detect and bypass fully UNBOUNDED windows.
  if (can_optimize_unbounded_window(preceding_window_bounds.is_unbounded(),
                                    following_window_bounds.is_unbounded(),
                                    min_periods,
                                    aggr)) {
    return optimized_unbounded_window(group_keys, input, aggr, stream, mr);
  }

  auto const preceding_window = preceding_window_bounds.value();
  auto const following_window = following_window_bounds.value();

  CUDF_EXPECTS(-(preceding_window - 1) <= following_window,
               "Preceding window bounds must precede the following window bounds.");

  if (group_keys.num_columns() == 0) {
    // No Groupby columns specified. Treat as one big group.
    return detail::rolling_window(
      input, default_outputs, preceding_window, following_window, min_periods, aggr, stream, mr);
  }

  using sort_groupby_helper = cudf::groupby::detail::sort::sort_groupby_helper;

  sort_groupby_helper helper{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES, {}};
  auto const& group_offsets{helper.group_offsets(stream)};
  auto const& group_labels{helper.group_labels(stream)};

  // `group_offsets` are interpreted in adjacent pairs, each pair representing the offsets
  // of the first, and one past the last elements in a group.
  //
  // If `group_offsets` is not empty, it must contain at least two offsets:
  //   a. 0, indicating the first element in `input`
  //   b. input.size(), indicating one past the last element in `input`.
  //
  // Thus, for an input of 1000 rows,
  //   0. [] indicates a single group, spanning the entire column.
  //   1  [10] is invalid.
  //   2. [0, 1000] indicates a single group, spanning the entire column (thus, equivalent to no
  //   groups.)
  //   3. [0, 500, 1000] indicates two equal-sized groups: [0,500), and [500,1000).

  if (aggr.kind == aggregation::CUDA || aggr.kind == aggregation::PTX) {
    cudf::detail::preceding_window_wrapper grouped_preceding_window{
      group_offsets.data(), group_labels.data(), preceding_window};

    cudf::detail::following_window_wrapper grouped_following_window{
      group_offsets.data(), group_labels.data(), following_window};

    return cudf::detail::rolling_window_udf(
      input,
      cudf::detail::preceding_window_wrapper{
        group_offsets.data(), group_labels.data(), preceding_window},
      cudf::detail::following_window_wrapper{
        group_offsets.data(), group_labels.data(), following_window},
      min_periods,
      aggr,
      stream,
      mr);
  } else {
    namespace utils = cudf::detail::rolling;
    auto groups     = utils::grouped{group_labels.data(), group_offsets.data()};
    auto preceding =
      utils::make_clamped_window_iterator<utils::direction::PRECEDING>(preceding_window, groups);
    auto following =
      utils::make_clamped_window_iterator<utils::direction::FOLLOWING>(following_window, groups);
    return cudf::detail::rolling_window(
      input, default_outputs, preceding, following, min_periods, aggr, stream, mr);
  }
}

}  // namespace detail

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               column_view const& default_outputs,
                                               window_bounds preceding_window_bounds,
                                               window_bounds following_window_bounds,
                                               size_type min_periods,
                                               rolling_aggregation const& aggr,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return detail::grouped_rolling_window(group_keys,
                                        input,
                                        default_outputs,
                                        preceding_window_bounds,
                                        following_window_bounds,
                                        min_periods,
                                        aggr,
                                        stream,
                                        mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               size_type preceding_window,
                                               size_type following_window,
                                               size_type min_periods,
                                               rolling_aggregation const& aggr,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return grouped_rolling_window(group_keys,
                                input,
                                window_bounds::get(preceding_window),
                                window_bounds::get(following_window),
                                min_periods,
                                aggr,
                                stream,
                                mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               window_bounds preceding_window,
                                               window_bounds following_window,
                                               size_type min_periods,
                                               rolling_aggregation const& aggr,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return detail::grouped_rolling_window(group_keys,
                                        input,
                                        empty_like(input)->view(),
                                        preceding_window,
                                        following_window,
                                        min_periods,
                                        aggr,
                                        stream,
                                        mr);
}

std::unique_ptr<column> grouped_rolling_window(table_view const& group_keys,
                                               column_view const& input,
                                               column_view const& default_outputs,
                                               size_type preceding_window,
                                               size_type following_window,
                                               size_type min_periods,
                                               rolling_aggregation const& aggr,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return detail::grouped_rolling_window(group_keys,
                                        input,
                                        default_outputs,
                                        window_bounds::get(preceding_window),
                                        window_bounds::get(following_window),
                                        min_periods,
                                        aggr,
                                        stream,
                                        mr);
}

namespace detail {

namespace {

template <typename MakeWindows>
  requires(std::invocable<MakeWindows&>)
std::unique_ptr<table> grouped_range_rolling_window_impl(table_view const& group_keys,
                                                         std::span<rolling_request const> requests,
                                                         range_window_type preceding,
                                                         range_window_type following,
                                                         size_type orderby_size,
                                                         MakeWindows&& make_windows,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> results;
  results.reserve(requests.size());
  // Can we avoid making the window bounds?
  if (std::all_of(requests.begin(), requests.end(), [](rolling_request const& req) {
        return req.values.is_empty();
      })) {
    std::transform(requests.begin(),
                   requests.end(),
                   std::back_inserter(results),
                   [](rolling_request const& req) {
                     return cudf::detail::empty_output_for_rolling_aggregation(req.values,
                                                                               *req.aggregation);
                   });
    return std::make_unique<table>(std::move(results));
  }
  CUDF_EXPECTS(std::all_of(requests.begin(),
                           requests.end(),
                           [orderby_size](rolling_request const& req) {
                             return req.values.size() == orderby_size;
                           }),
               "Size mismatch between request columns and orderby.");

  // Can we do an optimized fully unbounded aggregation in all cases?
  if (std::all_of(requests.begin(), requests.end(), [&](rolling_request const& req) {
        return can_optimize_unbounded_window(std::holds_alternative<unbounded>(preceding),
                                             std::holds_alternative<unbounded>(following),
                                             req.min_periods,
                                             *req.aggregation);
      })) {
    std::transform(requests.begin(),
                   requests.end(),
                   std::back_inserter(results),
                   [&](rolling_request const& req) {
                     return optimized_unbounded_window(
                       group_keys, req.values, *req.aggregation, stream, mr);
                   });
    return std::make_unique<table>(std::move(results));
  }
  // OK, need to do the more complicated thing
  auto [preceding_column, following_column] = make_windows();
  auto const& preceding_view                = preceding_column->view();
  auto const& following_view                = following_column->view();
  std::transform(
    requests.begin(), requests.end(), std::back_inserter(results), [&](rolling_request const& req) {
      if (can_optimize_unbounded_window(std::holds_alternative<unbounded>(preceding),
                                        std::holds_alternative<unbounded>(following),
                                        req.min_periods,
                                        *req.aggregation)) {
        return optimized_unbounded_window(group_keys, req.values, *req.aggregation, stream, mr);
      } else {
        return detail::rolling_window(req.values,
                                      preceding_view,
                                      following_view,
                                      req.min_periods,
                                      *req.aggregation,
                                      stream,
                                      mr);
      }
    });
  return std::make_unique<table>(std::move(results));
}

}  // namespace

std::unique_ptr<table> grouped_range_rolling_window(table_view const& group_keys,
                                                    column_view const& orderby,
                                                    order order,
                                                    null_order null_order,
                                                    range_window_type preceding,
                                                    range_window_type following,
                                                    std::span<rolling_request const> requests,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  return grouped_range_rolling_window_impl(
    group_keys,
    requests,
    preceding,
    following,
    orderby.size(),
    [&] {
      return make_range_windows(group_keys,
                                orderby,
                                order,
                                null_order,
                                preceding,
                                following,
                                stream,
                                cudf::get_current_device_resource_ref());
    },
    stream,
    mr);
}

std::unique_ptr<table> grouped_range_rolling_window(table_view const& group_keys,
                                                    table_view const& orderby,
                                                    host_span<order const> orders,
                                                    host_span<null_order const> null_orders,
                                                    range_window_type preceding,
                                                    range_window_type following,
                                                    host_span<rolling_request const> requests,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(orderby.num_columns() > 0, "orderby must be non-empty");
  CUDF_EXPECTS(orderby.num_columns() == static_cast<size_type>(orders.size()),
               "ORDER BY column count must match order vector");
  CUDF_EXPECTS(orderby.num_columns() == static_cast<size_type>(null_orders.size()),
               "ORDER BY column count must match null-order vector");
  if (orderby.num_columns() == 1) {
    return detail::grouped_range_rolling_window(group_keys,
                                                orderby.column(0),
                                                orders.front(),
                                                null_orders.front(),
                                                preceding,
                                                following,
                                                requests,
                                                stream,
                                                mr);
  }
  return grouped_range_rolling_window_impl(
    group_keys,
    requests,
    preceding,
    following,
    orderby.num_rows(),
    [&] {
      return detail::make_range_windows(group_keys,
                                        orderby,
                                        orders,
                                        null_orders,
                                        preceding,
                                        following,
                                        stream,
                                        cudf::get_current_device_resource_ref());
    },
    stream,
    mr);
}

[[nodiscard]] static null_order deduce_null_order(
  column_view const& orderby,
  order order,
  rmm::device_uvector<size_type> const& offsets,
  rmm::device_uvector<size_type> const& per_group_nulls,
  rmm::cuda_stream_view stream)
{
  auto d_orderby = column_device_view::create(orderby, stream);
  if (order == order::ASCENDING) {
    // Sort order is ASCENDING
    // null_order was either BEFORE or AFTER
    // If at least one group has a null at the beginning and that
    // group has more entries than the null count of the group, must
    // be nulls at starts of groups (BEFORE),otherwise must be nulls at end (AFTER)
    auto it = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<bool>(
        [d_orderby       = *d_orderby,
         d_offsets       = offsets.data(),
         nulls_per_group = per_group_nulls.data()] __device__(size_type i) -> bool {
          return nulls_per_group[i] < (d_offsets[i + 1] - d_offsets[i]) &&
                 d_orderby.is_null_nocheck(d_offsets[i]);
        }));
    auto is_before =
      thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     it,
                     it + offsets.size() - 1,
                     false,
                     cuda::std::logical_or<>{});
    return is_before ? null_order::BEFORE : null_order::AFTER;
  } else {
    // Sort order is DESCENDING
    // null_order was either BEFORE or AFTER
    // If at least one group has a null at the end and that group has
    // more entries than the null count of the group must be nulls at ends of groups (BEFORE).
    // Otherwise must be nulls at start (AFTER)
    auto it = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<bool>(
        [d_orderby       = *d_orderby,
         d_offsets       = offsets.data(),
         nulls_per_group = per_group_nulls.data()] __device__(size_type i) -> bool {
          return nulls_per_group[i] < (d_offsets[i + 1] - d_offsets[i]) &&
                 d_orderby.is_null_nocheck(d_offsets[i + 1] - 1);
        }));
    auto is_before =
      thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     it,
                     it + offsets.size() - 1,
                     false,
                     cuda::std::logical_or<>{});
    return is_before ? null_order::BEFORE : null_order::AFTER;
  }
}

/**
 * @copydoc  std::unique_ptr<column> grouped_range_rolling_window(
 *               table_view const& group_keys,
 *               column_view const& orderby_column,
 *               cudf::order const& order,
 *               column_view const& input,
 *               range_window_bounds const& preceding,
 *               range_window_bounds const& following,
 *               size_type min_periods,
 *               rolling_aggregation const& aggr,
 *               rmm::device_async_resource_ref mr );
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> grouped_range_rolling_window(table_view const& group_keys,
                                                     column_view const& order_by_column,
                                                     cudf::order const& order,
                                                     column_view const& input,
                                                     range_window_bounds const& preceding,
                                                     range_window_bounds const& following,
                                                     size_type min_periods,
                                                     rolling_aggregation const& aggr,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) { return cudf::detail::empty_output_for_rolling_aggregation(input, aggr); }

  CUDF_EXPECTS((group_keys.num_columns() == 0 || group_keys.num_rows() == input.size()),
               "Size mismatch between group_keys and input vector.");

  CUDF_EXPECTS((min_periods > 0), "min_periods must be positive");

  // Detect and bypass fully UNBOUNDED windows.
  if (can_optimize_unbounded_window(
        preceding.is_unbounded(), following.is_unbounded(), min_periods, aggr)) {
    return optimized_unbounded_window(group_keys, input, aggr, stream, mr);
  }

  auto get_window_type = [](range_window_bounds const& bound) -> range_window_type {
    if (bound.is_unbounded()) {
      return unbounded{};
    } else if (bound.is_current_row()) {
      return current_row{};
    } else {
      return bounded_closed{bound.range_scalar()};
    }
  };
  auto [preceding_column, following_column] = [&]() {
    if (group_keys.num_columns() > 0 && order_by_column.has_nulls()) {
      using sort_helper = cudf::groupby::detail::sort::sort_groupby_helper;
      sort_helper helper{group_keys, null_policy::INCLUDE, sorted::YES, {}};
      auto const& labels   = helper.group_labels(stream);
      auto const& offsets  = helper.group_offsets(stream);
      auto per_group_nulls = order_by_column.has_nulls()
                               ? detail::nulls_per_group(order_by_column, offsets, stream)
                               : rmm::device_uvector<size_type>{0, stream};
      detail::rolling::preprocessed_group_info grouping{labels, offsets, per_group_nulls};
      auto null_order = deduce_null_order(order_by_column, order, offsets, per_group_nulls, stream);
      // Don't use make_range_windows since that reconstructs the grouping info.
      // This is polyfill code anyway, so can be removed after this public API is deprecated and
      // removed.
      return std::pair{
        detail::make_range_window(order_by_column,
                                  grouping,
                                  rolling::direction::PRECEDING,
                                  order,
                                  null_order,
                                  get_window_type(preceding),
                                  stream,
                                  cudf::get_current_device_resource_ref()),
        detail::make_range_window(order_by_column,
                                  grouping,
                                  rolling::direction::FOLLOWING,
                                  order,

                                  null_order,
                                  get_window_type(following),
                                  stream,
                                  cudf::get_current_device_resource_ref()),
      };
    } else {
      auto null_order =
        order_by_column.has_nulls()
          ? (((order == order::ASCENDING && order_by_column.null_count(0, 1, stream) == 1) ||
              (order == order::DESCENDING &&
               order_by_column.null_count(
                 order_by_column.size() - 1, order_by_column.size(), stream) == 1))
               ? null_order::BEFORE
               : null_order::AFTER)
          : null_order::BEFORE;
      return make_range_windows(group_keys,
                                order_by_column,
                                order,
                                null_order,
                                get_window_type(preceding),
                                get_window_type(following),
                                stream,
                                cudf::get_current_device_resource_ref());
    }
  }();

  return detail::rolling_window(
    input, preceding_column->view(), following_column->view(), min_periods, aggr, stream, mr);
}

}  // namespace detail

/**
 * @copydoc grouped_range_rolling_window(
 *               table_view const& group_keys,
 *               column_view const& orderby_column,
 *               cudf::order const& order,
 *               column_view const& input,
 *               range_window_bounds const& preceding,
 *               range_window_bounds const& following,
 *               size_type min_periods,
 *               rolling_aggregation const& aggr,
 *               rmm::device_async_resource_ref mr );
 */
std::unique_ptr<column> grouped_range_rolling_window(table_view const& group_keys,
                                                     column_view const& timestamp_column,
                                                     cudf::order const& timestamp_order,
                                                     column_view const& input,
                                                     range_window_bounds const& preceding,
                                                     range_window_bounds const& following,
                                                     size_type min_periods,
                                                     rolling_aggregation const& aggr,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::grouped_range_rolling_window(group_keys,
                                              timestamp_column,
                                              timestamp_order,
                                              input,
                                              preceding,
                                              following,
                                              min_periods,
                                              aggr,
                                              stream,
                                              mr);
}

std::unique_ptr<table> grouped_range_rolling_window(table_view const& group_keys,
                                                    column_view const& orderby,
                                                    order order,
                                                    null_order null_order,
                                                    range_window_type preceding,
                                                    range_window_type following,
                                                    std::span<rolling_request const> requests,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(std::all_of(requests.begin(),
                           requests.end(),
                           [](rolling_request const& req) { return req.min_periods > 0; }),
               "All min_periods must be positive");
  return detail::grouped_range_rolling_window(
    group_keys, orderby, order, null_order, preceding, following, requests, stream, mr);
}

std::unique_ptr<table> grouped_range_rolling_window(table_view const& group_keys,
                                                    table_view const& orderby,
                                                    host_span<order const> orders,
                                                    host_span<null_order const> null_orders,
                                                    range_window_type preceding,
                                                    range_window_type following,
                                                    host_span<rolling_request const> requests,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(std::all_of(requests.begin(),
                           requests.end(),
                           [](rolling_request const& req) { return req.min_periods > 0; }),
               "All min_periods must be positive");
  return detail::grouped_range_rolling_window(
    group_keys, orderby, orders, null_orders, preceding, following, requests, stream, mr);
}

}  // namespace cudf
