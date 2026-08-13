/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <utility>

/**
 * @file
 * @brief Class definition for distinct hash join, a hash join optimized for tables with unique
 * keys.
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

namespace detail {
/**
 * @brief Forward declaration for our distinct hash join
 */
class distinct_hash_join;
}  // namespace detail

/**
 * @brief Distinct hash join that builds a hash table with the right table on construction and
 * probes results in subsequent `*_join` member functions
 *
 * This class enables the distinct hash join scheme that builds with the right table once and
 * probes with many left tables (possibly in parallel).
 *
 * @note Behavior is undefined if the right table contains duplicates.
 * @note All NaNs are considered as equal
 */
class distinct_hash_join {
 public:
  distinct_hash_join() = delete;
  ~distinct_hash_join();
  distinct_hash_join(distinct_hash_join const&)            = delete;
  distinct_hash_join(distinct_hash_join&&)                 = delete;
  distinct_hash_join& operator=(distinct_hash_join const&) = delete;
  distinct_hash_join& operator=(distinct_hash_join&&)      = delete;

  /**
   * @brief Constructs a distinct hash join object for subsequent probe calls
   *
   * @throw cudf::logic_error if the right table has no columns
   * @throw std::invalid_argument if load_factor is not greater than 0 and less than or equal to 1
   *
   * @param right The right table that contains distinct elements
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param load_factor The desired ratio of filled slots to total slots in the hash table, must be
   * in range (0,1]. For example, 0.5 indicates a target of 50% occupancy. Note that the actual
   * occupancy achieved may be slightly lower than the specified value.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the internal hash table
   */
  distinct_hash_join(cudf::table_view const& right,
                     null_equality compare_nulls  = null_equality::EQUAL,
                     double load_factor           = 0.5,
                     rmm::cuda_stream_view stream = cudf::get_default_stream(),
                     cuda::mr::any_resource<cuda::mr::device_accessible> mr =
                       cudf::get_current_device_resource_ref());

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables. @see cudf::inner_join().
   *
   * @param left The left table, from which the keys are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices' device memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to
   * construct the result of performing an inner join between two tables
   * with `left` and `right` as the join keys.
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& left,
             rmm::cuda_stream_view stream      = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns the right table indices that can be used to construct the result of performing
   * a left join between two tables.
   *
   * @note For a given row index `i` of the left table, the resulting `right_indices[i]` contains
   * the row index of the matched row from the right table if there is a match. Otherwise, contains
   * `JoinNoMatch`.
   *
   * @param left The left table, from which the keys are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A `right_indices` column that can be used to construct the result of
   * performing a left join between two tables with `left` and `right` as the join
   * keys.
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> left_join(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  using impl_type = cudf::detail::distinct_hash_join;  ///< Implementation type

  std::unique_ptr<impl_type> _impl;  ///< Distinct hash join implementation
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
