/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

/**
 * @file
 * @brief Class definition for filtered hash join, which builds a hash table from a filter table
 * and probes it with left tables.
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

namespace detail {
/**
 * @brief Forward declaration for our filtered hash join
 */
class filtered_join;
}  // namespace detail

/**
 * @brief Filtered hash join that builds a hash table from the right (filter) table on creation
 * and probes results in subsequent `*_join` member functions.
 *
 * This class enables the filtered hash join scheme that builds a hash table once from the right
 * table, and probes as many times as needed (possibly in parallel) with different left tables.
 * The right table acts as the filter to be applied on left tables in subsequent `*_join`
 * operations. The underlying data structure is `cuco::static_set`.
 *
 * For use cases where the left table should be reused with multiple right tables, use
 * `cudf::mark_join` instead.
 *
 * @note All NaNs are considered as equal
 */
class filtered_join {
 public:
  filtered_join() = delete;
  ~filtered_join();
  filtered_join(filtered_join const&)            = delete;
  filtered_join(filtered_join&&)                 = delete;
  filtered_join& operator=(filtered_join const&) = delete;
  filtered_join& operator=(filtered_join&&)      = delete;

  /**
   * @brief Constructs a filtered hash join object for subsequent probe calls.
   *
   * The right table is used as the filter applied to multiple left tables in subsequent
   * `semi_join` or `anti_join` calls.
   *
   * @param right The right (filter) table used to build the hash table
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the internal hash table
   */
  filtered_join(cudf::table_view const& right,
                cudf::null_equality compare_nulls,
                rmm::cuda_stream_view stream,
                cuda::mr::any_resource<cuda::mr::device_accessible> mr =
                  cudf::get_current_device_resource_ref());

  /**
   * @brief Constructs a filtered hash join object for subsequent probe calls.
   *
   * The right table is used as the filter applied to multiple left tables in subsequent
   * `semi_join` or `anti_join` calls.
   *
   * @throws std::invalid_argument if `load_factor` is not in (0, 1]
   *
   * @param right The right (filter) table used to build the hash table
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param load_factor The desired ratio of filled slots to total slots in the hash table, must be
   * in range (0,1]. For example, 0.5 indicates a target of 50% occupancy. Note that the actual
   * occupancy achieved may be slightly lower than the specified value.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the internal hash table
   */
  filtered_join(cudf::table_view const& right,
                cudf::null_equality compare_nulls,
                double load_factor,
                rmm::cuda_stream_view stream,
                cuda::mr::any_resource<cuda::mr::device_accessible> mr =
                  cudf::get_current_device_resource_ref());

  /**
   * @brief Returns a vector of row indices corresponding to a semi-join
   * between the specified tables.
   *
   * The returned vector contains the row indices from the left table
   * for which there is a matching row in the right (filter) table.
   *
   * @code{.pseudo}
   * Right (filter):  {{1, 2, 3}}
   * Left:            {{0, 1, 2}}
   * Result: {1, 2}
   * @endcode
   *
   * @param left The left table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   *
   * @return A vector `left_indices` that can be used to construct
   * the result of performing a left semi join
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> semi_join(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns a vector of row indices corresponding to an anti-join
   * between the specified tables.
   *
   * The returned vector contains the row indices from the left table
   * for which there are no matching rows in the right (filter) table.
   *
   * @code{.pseudo}
   * Right (filter):  {{1, 2, 3}}
   * Left:            {{0, 1, 2}}
   * Result: {0}
   * @endcode
   *
   * @param left The left table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   *
   * @return A vector `left_indices` that can be used to construct
   * the result of performing a left anti join
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> anti_join(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  std::unique_ptr<cudf::detail::filtered_join> _impl;  ///< Filtered hash join implementation
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
