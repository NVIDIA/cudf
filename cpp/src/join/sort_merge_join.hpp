/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <memory>
#include <optional>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Implementation class for sort-merge join algorithm.
 *
 * This class enables the sort-merge join scheme that builds a preprocessed right table once,
 * and probes as many times as needed. All join methods are thread-safe and can be
 * called concurrently from multiple threads on the same instance.
 */
class sort_merge_join {
 public:
  sort_merge_join()                                  = delete;
  ~sort_merge_join()                                 = default;
  sort_merge_join(sort_merge_join const&)            = delete;
  sort_merge_join(sort_merge_join&&)                 = delete;
  sort_merge_join& operator=(sort_merge_join const&) = delete;
  sort_merge_join& operator=(sort_merge_join&&)      = delete;

  /**
   * @brief Construct a sort-merge join object that pre-processes the right table
   * on creation, and can be used on subsequent join operations with multiple
   * left tables.
   *
   * @param right The right table
   * @param is_right_sorted Enum to indicate if right table is pre-sorted
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  sort_merge_join(table_view const& right,
                  sorted is_right_sorted,
                  null_equality compare_nulls,
                  cuda::stream_ref stream);

  /**
   * @brief Returns the row indices for an inner join.
   *
   * @param left The left table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(table_view const& left,
             cuda::stream_ref stream,
             rmm::device_async_resource_ref mr) const;

  /**
   * @brief Returns the row indices for a left join.
   *
   * @param left The left table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(table_view const& left,
            cuda::stream_ref stream,
            rmm::device_async_resource_ref mr) const;

  /**
   * @brief Returns context information about matches between the left and right tables.
   *
   * @param left The left table to join with the pre-processed right table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   * @return A unique_ptr to join_match_context
   */
  std::unique_ptr<join_match_context> inner_join_match_context(
    table_view const& left, cuda::stream_ref stream, rmm::device_async_resource_ref mr) const;

  /**
   * @brief Performs an inner join between a partition of the left table and the right table.
   *
   * @param context The partition context containing match information and partition bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_inner_join(cudf::join_partition_context const& context,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr) const;

 private:
  /**
   * @brief Helper struct to pre-process tables before join operations
   */
  struct preprocessed_table {
    table_view _table_view;  ///< Unprocessed table view before pre-processing

    table_view
      _null_processed_table_view;  ///< Processed table view which is the null-free subset of the
                                   ///< rows of the unprocessed table view if null equality is set
                                   ///< to false, otherwise equal to the unprocessed table view

    std::optional<rmm::device_buffer> _validity_mask =
      std::nullopt;  ///< Optional validity mask for null_equality::UNEQUAL case
    std::optional<size_type> _num_nulls =
      std::nullopt;  ///< Optional count of nulls for null_equality::UNEQUAL case
    std::optional<std::unique_ptr<table>> _null_processed_table =
      std::nullopt;  ///< Optional filtered table for null_equality::UNEQUAL case

    std::optional<std::unique_ptr<column>> _null_processed_table_sorted_order =
      std::nullopt;  ///< Optional sort ordering for pre-sorted tables

    /**
     * @brief Factory method to create a preprocessed table
     *
     * @param table The table to preprocess
     * @param compare_nulls Controls whether null join-key values should match or not
     * @param is_sorted Enum to indicate if the table is pre-sorted
     * @param stream CUDA stream used for device memory operations and kernel launches
     * @return A preprocessed_table ready for join operations
     */
    static preprocessed_table create(table_view const& table,
                                     null_equality compare_nulls,
                                     sorted is_sorted,
                                     cuda::stream_ref stream);

    /**
     * @brief Mark rows in unprocessed table with nulls at root or child levels by populating the
     * _validity_mask
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     */
    void populate_nonnull_filter(cuda::stream_ref stream);

    /**
     * @brief Apply _validity_mask to the _table_view to create a null-free table
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     */
    void apply_nonnull_filter(cuda::stream_ref stream);

    /**
     * @brief Pre-process the unprocessed table when null equality is set to unequal
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     */
    void preprocess_unprocessed_table(cuda::stream_ref stream);

    /**
     * @brief Compute sorted ordering of the processed table
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     */
    void compute_sorted_order(cuda::stream_ref stream);

    /**
     * @brief Create mapping from processed table indices to unprocessed table indices
     *
     * @param stream CUDA stream used for device memory operations and kernel launches
     * @return A device vector containing the mapping from processed table indices to unprocessed
     * table indices
     */
    rmm::device_uvector<size_type> map_table_to_unprocessed(cuda::stream_ref stream) const;
  };

  /**
   * @brief Extended match context for sort-merge join with preprocessed table state.
   *
   * This struct derives from join_match_context and adds the preprocessed left table
   * state needed for thread-safe partitioned join operations. It is returned by
   * inner_join_match_context() and used with partitioned_inner_join().
   */
  struct sort_merge_join_match_context : public join_match_context {
    preprocessed_table preprocessed_left;  ///< Preprocessed left table state for partitioned joins

    /**
     * @brief Construct a sort_merge_join_match_context
     *
     * @param left_table The left table view
     * @param match_counts Device vector of match counts per row
     * @param preprocessed Preprocessed left table state
     */
    sort_merge_join_match_context(table_view left_table,
                                  std::unique_ptr<rmm::device_uvector<size_type>> match_counts,
                                  preprocessed_table preprocessed)
      : join_match_context{left_table, std::move(match_counts)},
        preprocessed_left{std::move(preprocessed)}
    {
    }
  };

  preprocessed_table preprocessed_right;  ///< Preprocessed right table
  std::unique_ptr<rmm::device_uvector<size_type>>
    right_run_rows;  ///< First row of each key run; the first `num_right_runs` entries are valid
  std::unique_ptr<rmm::device_uvector<size_type>>
    right_run_offsets;  ///< Start of each run in sorted order; the first `num_right_runs + 1`
                        ///< entries are valid, including the row-count sentinel
  size_type num_right_runs{};   ///< Number of distinct key runs in the processed right table
  null_equality compare_nulls;  ///< Null comparison mode

  /**
   * @brief Post-process left and right tables after the merge operation
   *
   * @param preprocessed_left The preprocessed left table
   * @param smaller_indices Indices for the smaller processed table used to construct the join
   * result
   * @param larger_indices Indices for the larger processed table used to construct the join result
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void postprocess_indices(preprocessed_table const& preprocessed_left,
                           device_span<size_type> smaller_indices,
                           device_span<size_type> larger_indices,
                           cuda::stream_ref stream) const;

  /**
   * @brief Core merge operation implementation for the sort-merge join algorithm.
   *
   * @tparam MergeOperation Functor type that implements the specific join operation
   * @param right_view The preprocessed right table view
   * @param left_view The preprocessed left table view
   * @param op The merge operation functor to execute during the merge
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return The result of the merge operation as defined by the MergeOperation functor
   */
  template <typename MergeOperation>
  auto invoke_merge(table_view right_view,
                    table_view left_view,
                    MergeOperation&& op,
                    cuda::stream_ref stream) const;
};

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
