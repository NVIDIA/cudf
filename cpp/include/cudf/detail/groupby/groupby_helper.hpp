/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf {
namespace groupby::detail {
namespace hash {
struct grouped_keys;
struct grouped_rows;
}  // namespace hash

/**
 * @brief Helper class for computing grouped operations
 *
 * This class builds and memoizes HashCSR groups and provides
 * building blocks for aggregations. It can provide:
 * 1. On-demand grouping or sorting of a value column based on `keys`
 *   which is provided at construction
 * 2. Group offsets: starting offsets of all groups in grouped key table
 * 3. Group rows: original row indices and scheduling metadata for direct reductions
 */
struct groupby_helper {
  using index_vector       = rmm::device_uvector<size_type>;
  using bitmask_vector     = rmm::device_uvector<bitmask_type>;
  using column_ptr         = std::unique_ptr<column>;
  using index_vector_ptr   = std::unique_ptr<index_vector>;
  using bitmask_vector_ptr = std::unique_ptr<bitmask_vector>;

  /**
   * @brief Construct a new helper object
   *
   * If `include_null_keys == NO`, then any row in `keys` containing a null
   * value will effectively be discarded. I.e., any values corresponding to
   * discarded rows in `keys` will not contribute to any aggregation.
   *
   * @param keys table to group by
   * @param include_null_keys Include rows in keys with nulls
   * @param keys_pre_sorted Indicate if the keys are already sorted. Enables
   *                        grouping by adjacent equality without hashing.
   */
  groupby_helper(table_view const& keys, null_policy include_null_keys, sorted keys_pre_sorted);

  ~groupby_helper();
  groupby_helper(groupby_helper const&)            = delete;
  groupby_helper& operator=(groupby_helper const&) = delete;
  groupby_helper(groupby_helper&&) noexcept;
  groupby_helper& operator=(groupby_helper&&) noexcept;

  /**
   * @brief Groups a column of values according to `keys` and sorts within each
   *  group.
   *
   * Groups the @p values where the groups are dictated by key table and each
   * group is sorted in ascending order, with NULL elements positioned at the
   * end of each group.
   *
   * @throw cudf::logic_error if `values.size() != keys.num_rows()`
   *
   * @param values The value column to group and sort
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   * @return the sorted and grouped column
   */
  std::unique_ptr<column> sorted_values(column_view const& values,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr);

  /**
   * @brief Groups a column of values according to `keys`
   *
   * The values within each group maintain their original order.
   *
   * @throw cudf::logic_error if `values.size() != keys.num_rows()`
   *
   * @param values The value column to group
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   * @return the grouped column
   */
  std::unique_ptr<column> grouped_values(column_view const& values,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

  /**
   * @brief Groups values without requiring their original order within each group.
   *
   * Groups have the same offsets and labels as `grouped_values`. The returned column owns its
   * gathered values, so subsequently requesting a stable grouped order does not change it.
   *
   * @param values The value column to group
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   * @return the grouped column with unspecified order within each group
   */
  std::unique_ptr<column> unordered_grouped_values(column_view const& values,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr);

  /**
   * @brief Get a table of unique keys
   *
   * @return a new table in which each row is a unique row in the grouped key table.
   */
  std::unique_ptr<table> unique_keys(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /**
   * @brief Get a table of grouped keys
   *
   * @return a new table containing the grouped keys.
   */
  std::unique_ptr<table> grouped_keys(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /**
   * @brief Get the number of groups in `keys`
   */
  size_type num_groups(cuda::stream_ref stream) { return group_offsets(stream).size() - 1; }

  /**
   * @brief Check whether grouped values can use the input order without filtering or gathering
   */
  bool is_presorted() const { return _is_presorted; }

  /**
   * @brief Return the effective number of keys
   *
   * When include_null_keys = YES, returned value is same as `keys.num_rows()`
   * When include_null_keys = NO, returned value is the number of rows in `keys`
   *  in which no element is null
   */
  size_type num_keys(cuda::stream_ref stream);

  /**
   * @brief Get the grouped order of `keys`, retaining input order within each group.
   *
   * Gathering `keys` by these indices produces contiguous groups.
   *
   * When ignore_null_keys = true, the result will not include indices
   * for null keys.
   *
   * Computes and stores a stable grouped order on first invocation, and returns
   * the stored order on subsequent calls.
   *
   * @return the grouped row indices for `keys`.
   */
  column_view grouped_order(cuda::stream_ref stream);

  /**
   * @brief Get grouped row indices without requiring input order within each group.
   *
   * The returned span is invalidated if a later request materializes stable row order.
   */
  device_span<size_type const> unordered_grouped_order(cuda::stream_ref stream);

  /**
   * @brief Get cached row indices, offsets, and scheduling arrays for direct reductions.
   *
   * The returned reference is invalidated if a later request materializes stable row order.
   */
  hash::grouped_rows const& reduction_groups(cuda::stream_ref stream);

  /**
   * @brief Get each group's offset into the grouped order of `keys`.
   *
   * Computes and stores the group offsets on first invocation and returns
   * the stored group offsets on subsequent calls.
   * This returns a vector of size `num_groups() + 1` such that the size of
   * group `i` is `group_offsets[i+1] - group_offsets[i]`
   *
   * @return vector of offsets of the starting point of each group in the grouped
   * key table
   */
  index_vector const& group_offsets(cuda::stream_ref stream);

  /**
   * @brief Get the group labels corresponding to the grouped order of `keys`.
   *
   * Each group is assigned a unique numerical "label" in
   * `[0, 1, 2, ... , num_groups() - 1, num_groups(stream))`.
   * For a row in grouped `keys`, its corresponding group label indicates which
   * group it belongs to.
   *
   * Computes and stores labels on first invocation and returns stored labels on
   * subsequent calls.
   *
   * @return vector of group labels for each row in the grouped key column
   */
  index_vector const& group_labels(cuda::stream_ref stream);

 private:
  /**
   * @brief Get the group labels for ungrouped keys
   *
   * Returns the group label for every row in the original `keys` table. For a
   * given unique key row, its group label is equivalent to what is returned by
   * `group_labels(stream)`. However, if a row contains a null value, and
   * `include_null_keys == NO`, then its label is NULL.
   *
   * Computes and stores unsorted labels on first invocation and returns stored
   * labels on subsequent calls.
   *
   * @return A nullable column of `INT32` containing group labels in the order
   *         of the ungrouped key table
   */
  column_view ungrouped_keys_labels(cuda::stream_ref stream);

  /// Materialize grouping metadata, optionally retaining input order within groups.
  void build_groups(cuda::stream_ref stream, bool stable_rows = false);

  /// Materialize a stable row permutation only when an ordered operation needs it.
  void make_stable(cuda::stream_ref stream);

  column_ptr _unsorted_keys_labels;             ///< Labels in input order, null for excluded rows
  table_view _keys;                             ///< Input grouping keys
  std::unique_ptr<hash::grouped_keys> _groups;  ///< HashCSR grouping metadata
  std::unique_ptr<hash::grouped_rows> _reduction_groups;  ///< Cached reduction scheduling
  index_vector_ptr _group_labels;                         ///< Labels in grouped order
  sorted _keys_pre_sorted;         ///< Whether key groups are already contiguous
  null_policy _include_null_keys;  ///< Whether to retain null key rows
  bool _is_presorted;              ///< Whether grouped values can use the input directly
  bool _stable{false};             ///< Whether rows within each group are in input order
};

}  // namespace groupby::detail
}  // namespace cudf
