/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>

/**
 * @file
 * @brief Class implementing the sort-merge algorithm for table joins
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

// Forward declaration
namespace detail {
class sort_merge_join;
}

/**
 * @brief Class that implements sort-merge algorithm for table joins
 *
 * This class enables the sort-merge join scheme that builds a preprocessed right table once,
 * and probes as many times as needed. All join methods (`inner_join()`, `left_join()`,
 * `inner_join_match_context()`, and `partitioned_inner_join()`) are thread-safe and can be
 * called concurrently from multiple threads on the same instance.
 */
class sort_merge_join {
 public:
  sort_merge_join() = delete;
  ~sort_merge_join();
  sort_merge_join(sort_merge_join const&)            = delete;
  sort_merge_join(sort_merge_join&&)                 = delete;
  sort_merge_join& operator=(sort_merge_join const&) = delete;
  sort_merge_join& operator=(sort_merge_join&&)      = delete;

  /**
   * @brief Construct a sort-merge join object that pre-processes the right table
   * on creation, and can be used on subsequent join operations with multiple
   * left tables.
   *
   * @note The `sort_merge_join` object must not outlive the table viewed by `right`,
   * else behavior is undefined.
   *
   * @param right The right table
   * @param is_right_sorted Enum to indicate if right table is pre-sorted
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  sort_merge_join(table_view const& right,
                  sorted is_right_sorted,
                  null_equality compare_nulls  = null_equality::EQUAL,
                  rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * an inner join between the right table passed while creating the sort_merge_join object, and the
   * left table.
   * @see cudf::inner_join().
   *
   * This method is thread-safe and can be called concurrently from multiple threads.
   *
   * @param left The left table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device
   * memory.
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] that can be used to
   * construct the result of performing an inner join between two tables
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(table_view const& left,
             rmm::cuda_stream_view stream      = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * an inner join between the right table passed while creating the sort_merge_join object, and the
   * left table.
   *
   * @deprecated in release 26.10. The `is_left_sorted` parameter is ignored. Use the overload
   * without `is_left_sorted` instead.
   *
   * @param left The left table
   * @param is_left_sorted Ignored
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] that can be used to
   * construct the result of performing an inner join between two tables
   */
  [[deprecated(
    "The is_left_sorted parameter is ignored; use inner_join(left, stream, mr) "
    "instead.")]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(table_view const& left,
             sorted is_left_sorted,
             rmm::cuda_stream_view stream      = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * a left join between the right table passed while creating the sort_merge_join object, and the
   * left table.
   * @see cudf::left_join().
   *
   * This method is thread-safe and can be called concurrently from multiple threads.
   *
   * @param left The left table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] that can be used to
   * construct the result of performing a left join between two tables
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(table_view const& left,
            rmm::cuda_stream_view stream      = cudf::get_default_stream(),
            rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * a left join between the right table passed while creating the sort_merge_join object, and the
   * left table.
   *
   * @deprecated in release 26.10. The `is_left_sorted` parameter is ignored. Use the overload
   * without `is_left_sorted` instead.
   *
   * @param left The left table
   * @param is_left_sorted Ignored
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] that can be used to
   * construct the result of performing a left join between two tables
   */
  [[deprecated(
    "The is_left_sorted parameter is ignored; use left_join(left, stream, mr) "
    "instead.")]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                            std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(table_view const& left,
            sorted is_left_sorted,
            rmm::cuda_stream_view stream      = cudf::get_default_stream(),
            rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns context information about matches between the left and right tables.
   *
   * This method computes, for each row in the left table, how many matching rows exist in
   * the right table according to inner join semantics, and returns the number of matches through a
   * match_context object.
   *
   * This method is thread-safe and can be called concurrently from multiple threads.
   *
   * This is particularly useful for:
   * - Determining the total size of a potential join result without materializing it
   * - Planning partitioned join operations for large datasets
   *
   * The returned sort_merge_join_match_context can be used directly with partitioned_inner_join()
   * to process large joins in manageable chunks.
   *
   * @param left The left table to join with the pre-processed right table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A unique_ptr to join_match_context
   *         containing the left table view, match counts, and preprocessed left table state for
   *         partitioned joins
   */
  std::unique_ptr<join_match_context> inner_join_match_context(
    table_view const& left,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns context information about matches between the left and right tables.
   *
   * @deprecated in release 26.10. The `is_left_sorted` parameter is ignored. Use the overload
   * without `is_left_sorted` instead.
   *
   * @param left The left table to join with the pre-processed right table
   * @param is_left_sorted Ignored
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A unique_ptr to join_match_context
   */
  [[deprecated(
    "The is_left_sorted parameter is ignored; use inner_join_match_context(left, "
    "stream, mr) instead.")]] std::unique_ptr<join_match_context>
  inner_join_match_context(
    table_view const& left,
    sorted is_left_sorted,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Performs an inner join between a partition of the left table and the right table.
   *
   * This method executes an inner join operation between a specific partition of the left table
   * (defined by the join_partition_context) and the right table that was provided when constructing
   * the sort_merge_join object. The join_partition_context must have been previously created by
   * calling inner_join_match_context().
   *
   * This method is thread-safe and can be called concurrently from multiple threads.
   *
   * This partitioning approach enables processing large joins in smaller, memory-efficient chunks,
   * while maintaining consistent results as if the entire join was performed at once. This is
   * particularly useful for handling large datasets that would otherwise exceed available memory
   * resources.
   *
   * The returned indices can be used to construct the join result for this partition. The
   * left_indices are relative to the original complete left table (not just the partition), so they
   * can be used directly with the original left table to extract matching rows.
   *
   * @param context The partition context containing match information and partition bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] containing the row indices
   *         from both tables that satisfy the join condition for this partition. The left_indices
   *         are relative to the complete left table, not just the partition.
   *
   * @code{.cpp}
   * // Create join object with pre-processed right table
   * sort_merge_join join_obj(right_table, sorted::NO);
   *
   * // Get match context for the entire left table
   * auto match_ctx = join_obj.inner_join_match_context(left_table);
   *
   * // Create partition context
   * cudf::join_partition_context part_ctx{std::move(match_ctx), 0, 0};
   *
   * // Define partition boundaries (e.g., process 1000 rows at a time)
   * for (size_type start = 0; start < left_table.num_rows(); start += 1000) {
   *   size_type end = std::min(start + 1000, left_table.num_rows());
   *
   *   // Set partition boundaries
   *   part_ctx.left_start_idx = start;
   *   part_ctx.left_end_idx = end;
   *
   *   // Get join indices for this partition
   *   auto [left_indices, right_indices] = join_obj.partitioned_inner_join(part_ctx);
   *
   *   // Process the partition result...
   * }
   * @endcode
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_inner_join(
    cudf::join_partition_context const& context,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  using impl_type = cudf::detail::sort_merge_join;  ///< Implementation type
  std::unique_ptr<impl_type const> _impl;           ///< Pointer to implementation
};

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
