/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/export.hpp>

/**
 * @file
 * @brief Class defining range-based window boundary sizes for rolling window operations
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup aggregation_rolling
 * @{
 */

/**
 * @brief Abstraction for window boundary sizes, to be used with
 *        `grouped_range_rolling_window()`.
 *
 * Similar to `window_bounds` in `grouped_rolling_window()`, `range_window_bounds`
 * represents window boundaries for use with `grouped_range_rolling_window()`.
 * A window may be specified as one of the following:
 *   1. A fixed-width numeric scalar value. E.g.
 *      a) A `DURATION_DAYS` scalar, for use with a `TIMESTAMP_DAYS` orderby column
 *      b) An `INT32` scalar, for use with an `INT32` orderby column
 *   2. "unbounded", indicating that the bounds stretch to the first/last
 *      row in the group.
 *   3. "current row", indicating that the bounds end at the first/last
 *      row in the group that match the value of the current row.
 */
struct range_window_bounds {
 public:
  /**
   * @brief The type of range_window_bounds.
   */
  enum class extent_type : int32_t {
    CURRENT_ROW = 0,  /// Bounds defined as the first/last row that matches the current row.
    BOUNDED,          /// Bounds defined as the first/last row that falls within
                      /// a specified range from the current row.
    UNBOUNDED         /// Bounds stretching to the first/last row in the entire group.
  };

  /**
   * @brief Factory method to construct a bounded window boundary.
   *
   * @param boundary Finite window boundary
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return A bounded window boundary object
   */
  static range_window_bounds get(scalar const& boundary,
                                 rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Factory method to construct a window boundary
   *  limited to the value of the current row
   *
   * @param type The datatype of the window boundary
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return  A "current row" window boundary object
   */
  static range_window_bounds current_row(data_type type,
                                         rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Whether or not the window is bounded to the current row
   *
   * @return true If window is bounded to the current row
   * @return false If window is not bounded to the current row
   */
  [[nodiscard]] bool is_current_row() const { return _extent == extent_type::CURRENT_ROW; }

  /**
   * @brief Factory method to construct an unbounded window boundary.
   *
   * @param type The datatype of the window boundary
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return  An unbounded window boundary object
   */
  static range_window_bounds unbounded(data_type type,
                                       rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Whether or not the window is unbounded
   *
   * @return true If window is unbounded
   * @return false If window is of finite bounds
   */
  [[nodiscard]] bool is_unbounded() const { return _extent == extent_type::UNBOUNDED; }

  /**
   * @brief Returns the underlying scalar value for the bounds
   *
   * @return  The underlying scalar value for the bounds
   */
  [[nodiscard]] scalar const& range_scalar() const { return *_range_scalar; }

  range_window_bounds(range_window_bounds const&) = default;  ///< Copy constructor
  range_window_bounds() = default;  // Required for use as return types from dispatch functors.

 private:
  extent_type _extent{extent_type::UNBOUNDED};
  std::shared_ptr<scalar> _range_scalar{nullptr};  // To enable copy construction/assignment.

  range_window_bounds(extent_type extent_,
                      std::unique_ptr<scalar> range_scalar_,
                      rmm::cuda_stream_view = cudf::get_default_stream());
};

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
