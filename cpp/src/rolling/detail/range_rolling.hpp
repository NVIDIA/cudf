/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <utility>

namespace cudf::detail {

/**
 * @brief Constructs preceding and following window-size columns for a single-column RANGE window.
 *
 * @param group_keys Possibly empty table of sorted keys defining groups
 * @param orderby Sorted order-by column
 * @param order Sort order of the order-by column
 * @param null_order Null sort order of the order-by column
 * @param preceding Type of the preceding window
 * @param following Type of the following window
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return Pair of preceding and following window-size columns
 */
[[nodiscard]] std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  null_order null_order,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Dispatches computation of an unbounded RANGE window-size column.
 *
 * @param window Unbounded window tag
 * @param orderby Sorted order-by column
 * @param direction Direction of the window
 * @param order Sort order of the order-by column
 * @param grouping Preprocessed grouping information, if any
 * @param nulls_at_start Whether nulls are ordered before non-null values
 * @param row_delta Must be null for an unbounded window
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column containing the window size for each row
 */
[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  unbounded window,
  column_view const& orderby,
  rolling::direction direction,
  order order,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Dispatches computation of a current-row RANGE window-size column.
 *
 * @param window Current-row window tag
 * @param orderby Sorted order-by column
 * @param direction Direction of the window
 * @param order Sort order of the order-by column
 * @param grouping Preprocessed grouping information, if any
 * @param nulls_at_start Whether nulls are ordered before non-null values
 * @param row_delta Must be null for a current-row window
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column containing the window size for each row
 */
[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  current_row window,
  column_view const& orderby,
  rolling::direction direction,
  order order,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Dispatches computation of a bounded-closed RANGE window-size column.
 *
 * @param window Bounded-closed window tag
 * @param orderby Sorted order-by column
 * @param direction Direction of the window
 * @param order Sort order of the order-by column
 * @param grouping Preprocessed grouping information, if any
 * @param nulls_at_start Whether nulls are ordered before non-null values
 * @param row_delta Must be non-null and contain the bounded-window delta
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column containing the window size for each row
 */
[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_closed window,
  column_view const& orderby,
  rolling::direction direction,
  order order,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Dispatches computation of a bounded-open RANGE window-size column.
 *
 * @param window Bounded-open window tag
 * @param orderby Sorted order-by column
 * @param direction Direction of the window
 * @param order Sort order of the order-by column
 * @param grouping Preprocessed grouping information, if any
 * @param nulls_at_start Whether nulls are ordered before non-null values
 * @param row_delta Must be non-null and contain the bounded-window delta
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column containing the window size for each row
 */
[[nodiscard]] std::unique_ptr<column> dispatch_range_window(
  bounded_open window,
  column_view const& orderby,
  rolling::direction direction,
  order order,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
