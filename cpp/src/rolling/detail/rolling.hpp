/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/rolling.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cudf {
// helper functions - used in the rolling window implementation and tests

namespace detail {

// store functor
template <typename T, bool is_mean = false>
struct rolling_store_output_functor {
  CUDF_HOST_DEVICE inline void operator()(T& out, T& val, size_type count) { out = val; }
};

// Specialization for MEAN
template <typename _T>
struct rolling_store_output_functor<_T, true> {
  // Don't store the output if count is zero since integral division by zero is
  // undefined behaviour. The caller must ensure that the relevant row is
  // marked as invalid with a null.

  // SFINAE for non-bool, non-timestamp types
  template <typename T = _T>
  CUDF_HOST_DEVICE inline void operator()(T& out, T& val, size_type count)
    requires(!(cudf::is_boolean<T>() || cudf::is_timestamp<T>()))
  {
    if (count > 0) { out = val / count; }
  }

  // SFINAE for timestamp types
  template <typename T = _T>
  CUDF_HOST_DEVICE inline void operator()(T& out, T& val, size_type count)
    requires(cudf::is_timestamp<T>())
  {
    if (count > 0) { out = static_cast<T>(val.time_since_epoch() / count); }
  }
};

/**
 * @copydoc cudf::rolling_window(column_view const& input,
 *                               column_view const& default_outputs,
 *                               size_type preceding_window,
 *                               size_type following_window,
 *                               size_type min_periods,
 *                               rolling_aggregation const& agg,
 *                               rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream to use for device memory operations
 */
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::rolling_window(column_view const& input,
 *                               column_view const& preceding_window,
 *                               column_view const& following_window,
 *                               size_type min_periods,
 *                               rolling_aggregation const& agg,
 *                               rmm::device_async_resource_ref mr);
 *
 * @param stream CUDA stream to use for device memory operations
 */
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

bool is_valid_rolling_aggregation(data_type input_type, aggregation::Kind kind);

/**
 * @brief Construct preceding and following columns for a multi-column order-by RANGE window.
 *
 * Implementation detail of the multi-column `grouped_range_rolling_window` overload. Multi-column
 * order-by windows support only peer-frame endpoints (`unbounded` and `current_row`) since a scalar
 * delta is not well-defined across multiple order-by columns.
 *
 * @param group_keys Possibly empty table of sorted keys defining groups.
 * @param orderby Table defining sorted order-by keys. If `group_keys` is non-empty, must be sorted
 * groupwise.
 * @param orders Sort order for each order-by column.
 * @param null_orders Null sort order for each order-by column.
 * @param preceding Type of the preceding window. Must be `unbounded` or `current_row`.
 * @param following Type of the following window. Must be `unbounded` or `current_row`.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned columns' device memory.
 * @return Pair of preceding and following columns suitable for passing to `rolling_window`.
 */
[[nodiscard]] std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  table_view const& orderby,
  host_span<order const> orders,
  host_span<null_order const> null_orders,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail

}  // namespace cudf
