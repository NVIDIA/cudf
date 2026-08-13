/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <span>
#include <variant>

/**
 * @file
 * @brief APIs for computing rolling window aggregations over columns.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup aggregation_rolling
 * @{
 */

/**
 * @brief Strongly typed wrapper for bounded closed rolling windows.
 *
 * @param delta The scalar delta from the current row. Must be valid,
 * behaviour is undefined if not. If the scalar represents a floating
 * point type the value must be neither inf nor nan, otherwise
 * behaviour is undefined.
 *
 * The endpoints of this window are included.
 */
struct bounded_closed {
  cudf::scalar const& delta_;  ///< Delta from the current row in the window. Must be valid,
                               ///< behaviour is undefined if not.

  /**
   * @brief Construct a bounded closed rolling window.
   *
   * @param delta The scalar delta from the current row. Must be valid, behaviour is undefined if
   * not.
   */
  bounded_closed(cudf::scalar const& delta) : delta_{delta} {}
  /**
   * @brief Return pointer to the row delta scalar.
   * @return pointer to scalar, not null.
   */
  [[nodiscard]] cudf::scalar const* delta() const noexcept { return &delta_; }
};

/**
 * @brief Strongly typed wrapper for bounded open rolling windows.
 *
 * @param delta The scalar delta from the current row. Must be valid,
 * behaviour is undefined if not. If the scalar represents a floating
 * point type the value must be neither inf nor nan, otherwise
 * behaviour is undefined.
 *
 * The endpoints of this window are excluded.
 */
struct bounded_open {
  cudf::scalar const& delta_;  ///< Delta from the current row in the window. Must be valid,
                               ///< behaviour is undefined if not.

  /**
   * @brief Construct a bounded open rolling window.
   *
   * @param delta The scalar delta from the current row. Must be valid, behaviour is undefined if
   * not.
   */
  bounded_open(cudf::scalar const& delta) : delta_{delta} {}

  /**
   * @brief Return pointer to the row delta scalar.
   * @return pointer to scalar, not null.
   */
  [[nodiscard]] cudf::scalar const* delta() const noexcept { return &delta_; }
};

/**
 * @brief Strongly typed wrapper for unbounded rolling windows.
 *
 * This window runs to the begin/end of the current row's group.
 */
struct unbounded {
  /**
   * @brief Return a null row delta
   * @return nullptr
   */
  [[nodiscard]] constexpr cudf::scalar const* delta() const noexcept { return nullptr; }
};
/**
 * @brief Strongly typed wrapper for current_row rolling windows.
 *
 * This window contains all rows that are equal to the current row.
 */
struct current_row {
  /**
   * @brief Return a null row delta
   * @return nullptr
   */
  [[nodiscard]] constexpr cudf::scalar const* delta() const noexcept { return nullptr; }
};

/**
 * @brief The type of the range-based rolling window endpoint.
 */
using range_window_type = std::variant<unbounded, current_row, bounded_closed, bounded_open>;

/**
 * @brief A request for a rolling aggregation on a column.
 */
struct rolling_request {
  column_view values;     ///< Elements to aggregate
  size_type min_periods;  ///< Minimum number of observations required for the window to be valid
  std::unique_ptr<rolling_aggregation> aggregation;  ///< Desired aggregation
};

/**
 * @brief Constructs preceding and following columns given window range specifications.
 *
 * @param group_keys Possibly empty table of sorted keys defining groups.
 * @param orderby Column defining window ranges. Must be sorted. If `group_keys` is non-empty, must
 * be sorted groupwise.
 * @param order Sort order of the `orderby` column.
 * @param null_order Null sort order in the sorted `orderby` column. Apples groupwise if
 * `group_keys` is non-empty.
 * @param preceding Type of the preceding window.
 * @param following Type of the following window.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return pair of preceding and following columns that define the window bounds for each row,
 * suitable for passing to `rolling_window`.
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  null_order null_order,
  range_window_type preceding,
  range_window_type following,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Applies a fixed-size rolling window function to the values in a column.
 *
 * This function aggregates values in a window around each element i of the input column, and
 * invalidates the bit mask for element i if there are not enough observations. The window size is
 * static (the same for each element). This matches Pandas' API for DataFrame.rolling with a few
 * notable differences:
 * - instead of the center flag it uses a two-part window to allow for more flexible windows.
 *   The total window size = `preceding_window + following_window`. Element `i` uses elements
 *   `[i-preceding_window+1, i+following_window]` to do the window computation.
 * - instead of storing NA/NaN for output rows that do not meet the minimum number of observations
 *   this function updates the valid bitmask of the column to indicate which elements are valid.
 *
 * @note Windows near the endpoints of the input are automatically clamped to be in-bounds.
 *
 * Notes on return column types:
 * - The returned column for count aggregation always has `INT32` type.
 * - The returned column for VARIANCE/STD aggregations always has `FLOAT64` type.
 * - All other operators return a column of the same type as the input. Therefore
 *   it is suggested to convert integer column types (especially low-precision integers)
 *   to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 *
 * @param[in] input The input column
 * @param[in] preceding_window The static rolling window size in the backward direction
 * @param[in] following_window The static rolling window size in the forward direction
 * @param[in] min_periods Minimum number of valid (non-null) observations in window required to
 *                        produce a valid result. If the number of valid observations is less than
 *                        `min_periods`, the result for element `i` is null, except when
 *                        `min_periods=0`. When `min_periods=0`, aggregations return identity values
 *                        for windows with insufficient observations: SUM and COUNT return 0, MIN
 *                        returns the maximum value for the type, MAX returns the minimum value for
 *                        the type. Note: MEAN behavior is undefined for empty windows.
 * @param[in] agg The rolling window aggregation type (SUM, MAX, MIN, etc.)
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns   A nullable output column containing the rolling window results
 */
std::unique_ptr<column> rolling_window(
  column_view const& input,
  size_type preceding_window,
  size_type following_window,
  size_type min_periods,
  rolling_aggregation const& agg,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  @copybrief rolling_window
 * @details  @copydetails rolling_window(
 *            column_view const& input,
 *            size_type preceding_window,
 *            size_type following_window,
 *            size_type min_periods,
 *            rolling_aggregation const& agg,
 *            rmm::cuda_stream_view stream,
 *            rmm::device_async_resource_ref mr)
 *
 * @param default_outputs A column of per-row default values to be returned instead
 *                        of nulls. Used for LEAD()/LAG(), if the row offset crosses
 *                        the boundaries of the column.
 */
std::unique_ptr<column> rolling_window(
  column_view const& input,
  column_view const& default_outputs,
  size_type preceding_window,
  size_type following_window,
  size_type min_periods,
  rolling_aggregation const& agg,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Abstraction for window boundary sizes
 */
struct window_bounds {
 public:
  /**
   * @brief Construct bounded window boundary.
   *
   * @param value Finite window boundary (in days or rows)
   * @returns A window boundary
   */
  static window_bounds get(size_type value) { return window_bounds(false, value); }

  /**
   * @brief Construct unbounded window boundary.
   *
   * @return window_bounds
   */
  static window_bounds unbounded()
  {
    return window_bounds(true, std::numeric_limits<cudf::size_type>::max());
  }

  /**
   * Whether the window_bounds is unbounded.
   *
   * @return true if the window bounds is unbounded.
   * @return false if the window bounds has a finite row boundary.
   */
  [[nodiscard]] bool is_unbounded() const { return _is_unbounded; }

  /**
   * @brief Gets the row-boundary for this window_bounds.
   *
   * @return the row boundary value (in days or rows)
   */
  [[nodiscard]] size_type value() const { return _value; }

 private:
  explicit window_bounds(bool is_unbounded_, size_type value_ = 0)
    : _is_unbounded{is_unbounded_}, _value{value_}
  {
  }

  bool const _is_unbounded;  ///< Whether the window boundary is unbounded
  size_type const _value;    ///< Finite window boundary value (in days or rows)
};

/**
 * @brief  Applies a grouping-aware, fixed-size rolling window function to the values in a column.
 *
 * Like `rolling_window()`, this function aggregates values in a window around each
 * element of a specified `input` column. It differs from `rolling_window()` in that elements of the
 * `input` column are grouped into distinct groups (e.g. the result of a groupby). The window
 * aggregation cannot cross the group boundaries. For a row `i` of `input`, the group is determined
 * from the corresponding (i.e. i-th) values of the columns under `group_keys`.
 *
 * Note: This method requires that the rows are presorted by the `group_key` values.
 *
 * @code{.pseudo}
 * Example: Consider a user-sales dataset, where the rows look as follows:
 * { "user_id", sales_amt, day }
 *
 * The `grouped_rolling_window()` method enables windowing queries such as grouping a dataset by
 * `user_id`, and summing up the `sales_amt` column over a window of 3 rows (2 preceding (including
 * current row), 1 row following).
 *
 * In this example,
 *    1. `group_keys == [ user_id ]`
 *    2. `input == sales_amt`
 * The data are grouped by `user_id`, and ordered by `day`-string. The aggregation
 * (SUM) is then calculated for a window of 3 values around (and including) each row.
 *
 * For the following input:
 *
 *  [ // user,  sales_amt
 *    { "user1",   10      },
 *    { "user2",   20      },
 *    { "user1",   20      },
 *    { "user1",   10      },
 *    { "user2",   30      },
 *    { "user2",   80      },
 *    { "user1",   50      },
 *    { "user1",   60      },
 *    { "user2",   40      }
 *  ]
 *
 * Partitioning (grouping) by `user_id` yields the following `sales_amt` vector
 * (with 2 groups, one for each distinct `user_id`):
 *
 *    [ 10,  20,  10,  50,  60,  20,  30,  80,  40 ]
 *      <-------user1-------->|<------user2------->
 *
 * The SUM aggregation is applied with 1 preceding and 1 following
 * row, with a minimum of 1 period. The aggregation window is thus 3 rows wide,
 * yielding the following column:
 *
 *    [ 30, 40,  80, 120, 110,  50, 130, 150, 120 ]
 *
 * Note: The SUMs calculated at the group boundaries (i.e. indices 0, 4, 5, and 8)
 * consider only 2 values each, in spite of the window-size being 3.
 * Each aggregation operation cannot cross group boundaries.
 * @endcode
 *
 * The returned column for `op == COUNT` always has `INT32` type. All other operators return a
 * column of the same type as the input. Therefore it is suggested to convert integer column types
 * (especially low-precision integers) to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 *
 * Note: `preceding_window` and `following_window` could well have negative values. This yields
 * windows where the current row might not be included at all. For instance, consider a window
 * defined as (preceding=3, following=-1). This produces a window from 2 (i.e. 3-1) rows preceding
 * the current row, and 1 row *preceding* the current row. For the example above, the window for
 * row#3 is:
 *
 *    [ 10,  20,  10,  50,  60,  20,  30,  80,  40 ]
 *      <--window-->   ^
 *                     |
 *               current_row
 *
 * Similarly, `preceding` could have a negative value, indicating that the window begins at a
 * position after the current row.  It differs slightly from the semantics for `following`, because
 * `preceding` includes the current row. Therefore:
 *   1. preceding=1  => Window starts at the current row.
 *   2. preceding=0  => Window starts at 1 past the current row.
 *   3. preceding=-1 => Window starts at 2 past the current row. Etc.
 *
 * @param[in] group_keys The (pre-sorted) grouping columns
 * @param[in] input The input column (to be aggregated)
 * @param[in] preceding_window The static rolling window size in the backward direction (for
 * positive values), or forward direction (for negative values)
 * @param[in] following_window The static rolling window size in the forward direction (for positive
 * values), or backward direction (for negative values)
 * @param[in] min_periods Minimum number of valid (non-null) observations in window required to
 *                        produce a valid result, otherwise element `i` is null. Must be positive
 *                        (>= 1).
 * @param[in] aggr The rolling window aggregation type (SUM, MAX, MIN, etc.)
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns   A nullable output column containing the rolling window results
 */
std::unique_ptr<column> grouped_rolling_window(
  table_view const& group_keys,
  column_view const& input,
  size_type preceding_window,
  size_type following_window,
  size_type min_periods,
  rolling_aggregation const& aggr,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  @copybrief grouped_rolling_window
 * @details @copydetails grouped_rolling_window(
 *            table_view const& group_keys,
 *            column_view const& input,
 *            size_type preceding_window,
 *            size_type following_window,
 *            size_type min_periods,
 *            rolling_aggregation const& aggr,
 *            rmm::cuda_stream_view stream,
 *            rmm::device_async_resource_ref mr)
 */
std::unique_ptr<column> grouped_rolling_window(
  table_view const& group_keys,
  column_view const& input,
  window_bounds preceding_window,
  window_bounds following_window,
  size_type min_periods,
  rolling_aggregation const& aggr,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  @copybrief grouped_rolling_window
 * @details @copydetails grouped_rolling_window(
 *            table_view const& group_keys,
 *            column_view const& input,
 *            size_type preceding_window,
 *            size_type following_window,
 *            size_type min_periods,
 *            rolling_aggregation const& aggr,
 *            rmm::cuda_stream_view stream,,
 *            rmm::device_async_resource_ref mr)
 *
 * @param default_outputs A column of per-row default values to be returned instead
 *                        of nulls. Used for LEAD()/LAG(), if the row offset crosses
 *                        the boundaries of the column or group.
 */
std::unique_ptr<column> grouped_rolling_window(
  table_view const& group_keys,
  column_view const& input,
  column_view const& default_outputs,
  size_type preceding_window,
  size_type following_window,
  size_type min_periods,
  rolling_aggregation const& aggr,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  @copybrief grouped_rolling_window
 * @details @copydetails grouped_rolling_window(
 *            table_view const& group_keys,
 *            column_view const& input,
 *            column_view const& default_outputs,
 *            size_type preceding_window,
 *            size_type following_window,
 *            size_type min_periods,
 *            rolling_aggregation const& aggr,
 *            rmm::cuda_stream_view stream,
 *            rmm::device_async_resource_ref mr)
 */
std::unique_ptr<column> grouped_rolling_window(
  table_view const& group_keys,
  column_view const& input,
  column_view const& default_outputs,
  window_bounds preceding_window,
  window_bounds following_window,
  size_type min_periods,
  rolling_aggregation const& aggr,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Applies a grouping-aware, value range-based rolling window function to the values in a
 *         column.
 *
 * This function aggregates rows in a window around each element of a specified `input` column.
 * The window is determined based on the values of an ordered `orderby` column, and on the values
 * of a `preceding` and `following` scalar representing an inclusive range of orderby column values.
 *
 *   1. The elements of the `input` column are grouped into distinct groups (e.g. the result of a
 *      groupby), determined by the corresponding values of the columns under `group_keys`. The
 *      window-aggregation cannot cross the group boundaries.
 *   2. Within a group, with all rows sorted by the `orderby` column, the aggregation window
 *      for a row at index `i` is determined as follows:
 *      a) If `orderby` is ASCENDING, aggregation window for row `i` includes all `input` rows at
 *         index `j` such that:
 *         @code{.pseudo}
 *           (orderby[i] - preceding) <= orderby[j] <= orderby[i] + following
 *         @endcode
 *      b) If `orderby` is DESCENDING, aggregation window for row `i` includes all `input` rows at
 *         index `j` such that:
 *         @code{.pseudo}
 *           (orderby[i] + preceding) >= orderby[j] >= orderby[i] - following
 *         @endcode
 *
 * Note: This method requires that the rows are presorted by the group keys and orderby column
 * values.
 *
 * The window intervals are specified as scalar values appropriate for the orderby column.
 * Currently, only the following combinations of `orderby` column type and range types
 * are supported:
 *   1. If `orderby` column is a TIMESTAMP, the `preceding`/`following` windows are specified
 *      in terms of `DURATION` scalars of the same resolution.
 *      E.g. For `orderby` column of type `TIMESTAMP_SECONDS`, the intervals may only be
 *      `DURATION_SECONDS`. Durations of higher resolution (e.g. `DURATION_NANOSECONDS`)
 *      or lower (e.g. `DURATION_DAYS`) cannot be used.
 *   2. If the `orderby` column is an integral type (e.g. `INT32`), the `preceding`/`following`
 *      should be the exact same type (`INT32`).
 *
 * @code{.pseudo}
 * Example: Consider a motor-racing statistics dataset, containing the following columns:
 *   1. driver_name:   (STRING) Name of the car driver
 *   2. num_overtakes: (INT32)  Number of times the driver overtook another car in a lap
 *   3. lap_number:    (INT32)  The number of the lap
 *
 * The `group_range_rolling_window()` function allows one to calculate the total number of overtakes
 * each driver made within any 3 lap window of each entry:
 *   1. Group/partition the dataset by `driver_id` (This is the group_keys argument.)
 *   2. Sort each group by the `lap_number` (i.e. This is the orderby_column.)
 *   3. Calculate the SUM(num_overtakes) over a window (preceding=1, following=1)
 *
 * For the following input:
 *
 *  [ // driver_name,  num_overtakes,  lap_number
 *    {   "bottas",        1,            1        },
 *    {   "hamilton",      2,            1        },
 *    {   "bottas",        2,            2        },
 *    {   "bottas",        1,            3        },
 *    {   "hamilton",      3,            1        },
 *    {   "hamilton",      8,            2        },
 *    {   "bottas",        5,            7        },
 *    {   "bottas",        6,            8        },
 *    {   "hamilton",      4,            4        }
 *  ]
 *
 * Partitioning (grouping) by `driver_name`, and ordering by `lap_number` yields the following
 * `num_overtakes` vector (with 2 groups, one for each distinct `driver_name`):
 *
 * lap_number:      [ 1,  2,  3,  7,  8,   1,  1,   2,  4 ]
 * num_overtakes:   [ 1,  2,  1,  5,  6,   2,  3,   8,  4 ]
 *                    <-----bottas------>|<----hamilton--->
 *
 * The SUM aggregation is applied, with 1 preceding, and 1 following, with a minimum of 1
 * period. The aggregation window is thus 3 (laps) wide, yielding the following output column:
 *
 *  Results:        [ 3,  4,  3,  11, 11,  13, 13,  13,  4 ]
 *
 * @endcode
 *
 * Note: The number of rows participating in each window might vary, based on the index within the
 * group, datestamp, and `min_periods`. Apropos:
 *  1. results[0] considers 2 values, because it is at the beginning of its group, and has no
 *     preceding values.
 *  2. results[5] considers 3 values, despite being at the beginning of its group. It must include 2
 *     following values, based on its orderby_column value.
 *
 * Each aggregation operation cannot cross group boundaries.
 *
 * The type of the returned column depends on the input column type `T`, and the aggregation:
 *   1. COUNT   returns `INT32` columns
 *   2. MIN/MAX returns `T` columns
 *   3. SUM     returns the promoted type for T. Sum on `INT32` yields `INT64`.
 *   4. MEAN    returns FLOAT64 columns
 *   5. COLLECT returns columns of type `LIST<T>`.
 *
 * LEAD/LAG/ROW_NUMBER are undefined for range queries.
 *
 * @param[in] group_keys The (pre-sorted) grouping columns
 * @param[in] orderby_column The (pre-sorted) order-by column, for range comparisons
 * @param[in] order  The order (ASCENDING/DESCENDING) in which the order-by column is sorted
 * @param[in] input The input column (to be aggregated)
 * @param[in] preceding The interval value in the backward direction
 * @param[in] following The interval value in the forward direction
 * @param[in] min_periods Minimum number of valid (non-null) observations in window required to
 *                        produce a valid result, otherwise element `i` is null. Must be positive
 *                        (>= 1).
 * @param[in] aggr The rolling window aggregation type (SUM, MAX, MIN, etc.)
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns   A nullable output column containing the rolling window results
 */
std::unique_ptr<column> grouped_range_rolling_window(
  table_view const& group_keys,
  column_view const& orderby_column,
  cudf::order const& order,
  column_view const& input,
  range_window_bounds const& preceding,
  range_window_bounds const& following,
  size_type min_periods,
  rolling_aggregation const& aggr,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Apply a grouping-aware range-based rolling window function to a sequence of columns.
 *
 * @param group_keys Possibly empty table of sorted keys defining groups.
 * @param orderby Column defining window ranges. Must be sorted. If `group_keys` is non-empty, must
 * be sorted groupwise.
 * @param order Sort order of the `orderby` column.
 * @param null_order Null sort order in the sorted `orderby` column.
 * @param preceding Type of the preceding window.
 * @param following Type of the following window.
 * @param requests Columns to aggregate and the aggregation for each column.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return A table of results, one column per input request.
 */
std::unique_ptr<table> grouped_range_rolling_window(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  null_order null_order,
  range_window_type preceding,
  range_window_type following,
  std::span<rolling_request const> requests,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Apply a grouping-aware range-based rolling window function to a sequence of columns,
 * using one or more sorted order-by columns to identify peer rows.
 *
 * This overload is equivalent to the single-column `grouped_range_rolling_window` overload when
 * `orderby` contains one column. When `orderby` contains multiple columns, bounded scalar RANGE
 * endpoints are not supported because a scalar distance is only well-defined for one order-by
 * column. Multi-column order-by windows support peer-frame endpoints: `unbounded` and
 * `current_row`.
 *
 * @param group_keys Possibly empty table of sorted keys defining groups.
 * @param orderby Table defining sorted order-by keys. If `group_keys` is non-empty, must be sorted
 * groupwise.
 * @param orders Sort order for each order-by column.
 * @param null_orders Null sort order for each order-by column.
 * @param preceding Type of the preceding window.
 * @param following Type of the following window.
 * @param requests Columns to aggregate and the aggregation for each column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 * @return A table of results, one column per input request.
 */
std::unique_ptr<table> grouped_range_rolling_window(
  table_view const& group_keys,
  table_view const& orderby,
  host_span<order const> orders,
  host_span<null_order const> null_orders,
  range_window_type preceding,
  range_window_type following,
  host_span<rolling_request const> requests,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Applies a variable-size rolling window function to the values in a column.
 *
 * This function aggregates values in a window around each element i of the input column, and
 * invalidates the bit mask for element i if there are not enough observations. The window size is
 * dynamic (varying for each element). This matches Pandas' API for DataFrame.rolling with a few
 * notable differences:
 * - instead of the center flag it uses a two-part window to allow for more flexible windows.
 *   The total window size = `preceding_window + following_window`. Element `i` uses elements
 *   `[i-preceding_window+1, i+following_window]` to do the window computation.
 * - instead of storing NA/NaN for output rows that do not meet the minimum number of observations
 *   this function updates the valid bitmask of the column to indicate which elements are valid.
 * - support for dynamic rolling windows, i.e. window size can be specified for each element using
 *   an additional array.
 *
 * The returned column for count aggregation always has INT32 type. All other operators return a
 * column of the same type as the input. Therefore it is suggested to convert integer column types
 * (especially low-precision integers) to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 *
 * @note All entries in `preceding_window` and `following_window` must produce window extents that
 * are in-bounds for the `input`. That is, for all `i`, it is required that the set of rows defined
 * by the interval `[i - preceding_window[i] + 1, ..., i + following_window[i] + 1)` is a subset of
 * `[0, input.size())`.
 *
 * @throws cudf::logic_error if window column type is not INT32
 *
 * @param[in] input The input column
 * @param[in] preceding_window A non-nullable column of INT32 window sizes in the forward direction.
 *                             `preceding_window[i]` specifies preceding window size for
 *                             element `i`.
 * @param[in] following_window A non-nullable column of INT32 window sizes in the backward
 *                             direction. `following_window[i]` specifies following window size
 *                             for element `i`.
 * @param[in] min_periods Minimum number of valid (non-null) observations in window required to
 *                        produce a valid result. If the number of valid observations is less than
 *                        `min_periods`, the result for element `i` is null, except when
 *                        `min_periods=0`. When `min_periods=0`, aggregations return identity values
 *                        for windows with insufficient observations: SUM and COUNT return 0, MIN
 *                        returns the maximum value for the type, MAX returns the minimum value for
 *                        the type. Note: MEAN behavior is undefined for empty windows.
 * @param[in] agg The rolling window aggregation type (sum, max, min, etc.)
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns   A nullable output column containing the rolling window results
 */
std::unique_ptr<column> rolling_window(
  column_view const& input,
  column_view const& preceding_window,
  column_view const& following_window,
  size_type min_periods,
  rolling_aggregation const& agg,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Indicate if a rolling aggregation is supported for a source datatype.
 *
 * @param source Type of the column to perform the aggregation on.
 * @param kind The kind of the aggregation.
 * @returns true if the aggregation is supported.
 */
bool is_valid_rolling_aggregation(data_type source, aggregation::Kind kind);
/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
