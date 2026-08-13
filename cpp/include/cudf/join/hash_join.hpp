/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/hashing.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <optional>
#include <utility>

/**
 * @file
 * @brief Class definition for hash join, which builds a hash table with the right table and
 * probes it with left tables.
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

// forward declaration
namespace hashing::detail {
/**
 * @brief Forward declaration for our Murmur Hash 3 implementation
 */
template <typename T>
class MurmurHash3_x86_32;
}  // namespace hashing::detail

namespace detail {
/**
 * @brief Forward declaration for our hash join
 */
template <typename T>
class hash_join;
}  // namespace detail

/**
 * @brief The enum class to specify if any of the input join tables (`right` table and any later
 * `left` table) has nulls.
 *
 * This is used upon hash_join object construction to specify the existence of nulls in all the
 * possible input tables. If such null existence is unknown, `YES` should be used as the default
 * option.
 */
enum class nullable_join : bool { YES, NO };

/**
 * @brief Hash join that builds a hash table with the right table on construction and probes
 * results in subsequent `*_join` member functions.
 *
 * This class enables the hash join scheme that builds with the right table once and probes
 * with many left tables (possibly in parallel).
 */
class hash_join {
 public:
  using impl_type = typename cudf::detail::hash_join<
    cudf::hashing::detail::MurmurHash3_x86_32<cudf::hash_value_type>>;  ///< Implementation type

  hash_join() = delete;
  ~hash_join();
  hash_join(hash_join const&)            = delete;
  hash_join(hash_join&&)                 = delete;
  hash_join& operator=(hash_join const&) = delete;
  hash_join& operator=(hash_join&&)      = delete;

  /**
   * @brief Construct a hash join object for subsequent probe calls.
   *
   * @note The `hash_join` object must not outlive the table viewed by `right`, else behavior is
   * undefined.
   *
   * @throws std::invalid_argument if the right table has no columns
   *
   * @param right The right table, from which the hash table is built
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the internal hash table
   */
  hash_join(cudf::table_view const& right,
            null_equality compare_nulls,
            rmm::cuda_stream_view stream = cudf::get_default_stream(),
            cuda::mr::any_resource<cuda::mr::device_accessible> mr =
              cudf::get_current_device_resource_ref());

  /**
   * @brief Construct a hash join object for subsequent probe calls.
   *
   * @note The `hash_join` object must not outlive the table viewed by `right`, else behavior is
   * undefined.
   *
   * @throws std::invalid_argument if the right table has no columns
   * @throws std::invalid_argument if load_factor is not greater than 0 and less than or equal to 1
   *
   * @param right The right table, from which the hash table is built
   * @param has_nulls Flag to indicate if there exists any nulls in the `right` table or
   *                  any `left` table that will be used later for join
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param load_factor The hash table occupancy ratio in (0,1]. A value of 0.5 means 50% desired
   * occupancy.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the internal hash table
   */
  hash_join(cudf::table_view const& right,
            nullable_join has_nulls,
            null_equality compare_nulls,
            double load_factor,
            rmm::cuda_stream_view stream = cudf::get_default_stream(),
            cuda::mr::any_resource<cuda::mr::device_accessible> mr =
              cudf::get_current_device_resource_ref());

  /**
   * Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables. @see cudf::inner_join(). Behavior is undefined if the
   * provided `output_size` is smaller than the actual output size.
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing an inner join between two tables with `left` and `right`
   * as the join keys .
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& left,
             std::optional<std::size_t> output_size = {},
             rmm::cuda_stream_view stream           = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * Returns the row indices that can be used to construct the result of performing
   * a left join between two tables. @see cudf::left_join(). Behavior is undefined if the
   * provided `output_size` is smaller than the actual output size.
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing a left join between two tables with `left` and `right`
   * as the join keys.
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(cudf::table_view const& left,
            std::optional<std::size_t> output_size = {},
            rmm::cuda_stream_view stream           = cudf::get_default_stream(),
            rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref()) const;

  /**
   * Returns the row indices that can be used to construct the result of performing
   * a full join between two tables. @see cudf::full_join(). Behavior is undefined if the
   * provided `output_size` is smaller than the actual output size.
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing a full join between two tables with `left` and `right`
   * as the join keys .
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  full_join(cudf::table_view const& left,
            std::optional<std::size_t> output_size = {},
            rmm::cuda_stream_view stream           = cudf::get_default_stream(),
            rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref()) const;

  /**
   * Returns the exact number of matches (rows) when performing an inner join with the specified
   * left table.
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return The exact number of output when performing an inner join between two tables with
   * `left` and `right` as the join keys .
   */
  [[nodiscard]] std::size_t inner_join_size(
    cudf::table_view const& left, rmm::cuda_stream_view stream = cudf::get_default_stream()) const;

  /**
   * Returns the exact number of matches (rows) when performing a left join with the specified left
   * table.
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return The exact number of output when performing a left join between two tables with `left`
   * and `right` as the join keys .
   */
  [[nodiscard]] std::size_t left_join_size(
    cudf::table_view const& left, rmm::cuda_stream_view stream = cudf::get_default_stream()) const;

  /**
   * Returns the exact number of matches (rows) when performing a full join with the specified left
   * table.
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the intermediate table and columns' device
   * memory.
   *
   * @return The exact number of output when performing a full join between two tables with `left`
   * and `right` as the join keys .
   */
  [[nodiscard]] std::size_t full_join_size(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns context information about matches between the left and right tables.
   *
   * This method computes, for each row in the left table, how many matching rows exist in
   * the right table according to inner join semantics, and returns the number of matches through a
   * join_match_context object.
   *
   * This is particularly useful for:
   * - Determining the total size of a potential join result without materializing it
   * - Planning partitioned join operations for large datasets
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table to join with the pre-processed right table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A join_match_context object containing the left table view and a device vector
   *         of match counts for each row in the left table
   */
  [[nodiscard]] cudf::join_match_context inner_join_match_context(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns context information about matches between the left and right tables.
   *
   * This method computes, for each row in the left table, how many matching rows exist in
   * the right table according to left join semantics, and returns the number of matches through a
   * join_match_context object.
   *
   * For left join, every row in the left table will have at least one match (either with a
   * matching row from the right table or with a null placeholder).
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table to join with the pre-processed right table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A join_match_context object containing the left table view and a device vector
   *         of match counts for each row in the left table
   */
  [[nodiscard]] cudf::join_match_context left_join_match_context(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns context information about matches between the left and right tables.
   *
   * This method computes, for each row in the left table, how many matching rows exist in
   * the right table according to full join semantics, and returns the number of matches through a
   * join_match_context object.
   *
   * For full join, this includes matches for left table rows, and the result may need to be
   * combined with unmatched rows from the right table to get the complete picture.
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table to join with the pre-processed right table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A join_match_context object containing the left table view and a device vector
   *         of match counts for each row in the left table
   */
  [[nodiscard]] cudf::join_match_context full_join_match_context(
    cudf::table_view const& left,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Performs an inner join on a partition of the probe table.
   *
   * This method executes an inner join between a specific partition of the probe table
   * (defined by the join_partition_context) and the build table. The context must have been
   * previously created by calling inner_join_match_context().
   *
   * The returned left_indices are relative to the original complete probe table, not just the
   * partition, so they can be used directly with the original probe table.
   *
   * @throw std::invalid_argument If `context.left_table_context` is null, if its
   * `_match_counts` is null, or if `[left_start_idx, left_end_idx)` is outside the bounds
   * of the left table.
   *
   * @param context The partition context containing match information and partition bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] for this partition
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_inner_join(
    cudf::join_partition_context const& context,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Performs a left join on a partition of the probe table.
   *
   * This method executes a left join between a specific partition of the probe table
   * (defined by the join_partition_context) and the build table. The context must have been
   * previously created by calling left_join_match_context().
   *
   * The returned left_indices are relative to the original complete probe table.
   *
   * @throw std::invalid_argument If `context.left_table_context` is null, if its
   * `_match_counts` is null, or if `[left_start_idx, left_end_idx)` is outside the bounds
   * of the left table.
   *
   * @param context The partition context containing match information and partition bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] for this partition
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_left_join(
    cudf::join_partition_context const& context,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Performs a full join probe on a partition of the probe table.
   *
   * This method executes the probe-side of a full join between a specific partition of the probe
   * table (defined by the join_partition_context) and the build table. The context must have been
   * previously created by calling full_join_match_context().
   *
   * @note This method does NOT include unmatched build rows (the complement).  After all
   * partitions have been processed, pass the collected results to
   * `finalize_partitioned_full_join()` to obtain the complete full join output.
   *
   * The returned left_indices are relative to the original complete probe table.
   *
   * @throw std::invalid_argument If `context.left_table_context` is null, if its
   * `_match_counts` is null, or if `[left_start_idx, left_end_idx)` is outside the bounds
   * of the left table.
   *
   * @param context The partition context containing match information and partition bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] for this partition
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_full_join(
    cudf::join_partition_context const& context,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Finalizes a partitioned full join by concatenating all per-partition results
   * and appending the unmatched right rows (the complement).
   *
   * Call this method after calling `partitioned_full_join()` for every partition.  It combines
   * the per-partition indices with the unmatched right row indices (a global property
   * across all partitions) and returns a single `(left_indices, right_indices)` pair equivalent
   * to the output of `full_join()`.
   *
   * @param left_partials Per-partition `left_indices` views produced by `partitioned_full_join()`
   * @param right_partials Per-partition `right_indices` views produced by `partitioned_full_join()`
   * @param left_table_num_rows Total number of rows in the original left table
   * @param right_table_num_rows Total number of rows in the right table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] representing the full
   *         join output.
   */
  [[nodiscard]] static std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                                 std::unique_ptr<rmm::device_uvector<size_type>>>
  finalize_partitioned_full_join(
    cudf::host_span<cudf::device_span<size_type const> const> left_partials,
    cudf::host_span<cudf::device_span<size_type const> const> right_partials,
    size_type left_table_num_rows,
    size_type right_table_num_rows,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

 private:
  std::unique_ptr<impl_type const> _impl;
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
