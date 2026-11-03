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

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

namespace cudf {
namespace groupby::detail::sort {
/**
 * @brief Helper class for computing sort-based groupby
 *
 * This class serves the purpose of sorting the keys and values and provides
 * building blocks for aggregations. It can provide:
 * 1. On-demand grouping or sorting of a value column based on `keys`
 *   which is provided at construction
 * 2. Group offsets: starting offsets of all groups in sorted key table
 * 3. Group valid sizes: The number of valid values in each group in a sorted
 *   value column
 */
struct sort_groupby_helper {
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
   *                        optimizations to help skip re-sorting keys.
   * @param null_precedence Indicates the ordering of nulls in each column.
   *                        Default behavior for each column is
   *                        `null_order::AFTER`
   */
  sort_groupby_helper(table_view const& keys,
                      null_policy include_null_keys,
                      sorted keys_pre_sorted,
                      std::vector<null_order> const& null_precedence);

  ~sort_groupby_helper()                                     = default;
  sort_groupby_helper(sort_groupby_helper const&)            = delete;
  sort_groupby_helper& operator=(sort_groupby_helper const&) = delete;
  sort_groupby_helper(sort_groupby_helper&&)                 = default;
  sort_groupby_helper& operator=(sort_groupby_helper&&)      = default;

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
   * @brief Get a table of sorted unique keys
   *
   * @return a new table in which each row is a unique row in the sorted key table.
   */
  std::unique_ptr<table> unique_keys(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /**
   * @brief Get a table of sorted keys
   *
   * @return a new table containing the sorted keys.
   */
  std::unique_ptr<table> sorted_keys(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /**
   * @brief Get the number of groups in `keys`
   */
  size_type num_groups(cuda::stream_ref stream) { return group_offsets(stream).size() - 1; }

  /**
   * @brief check if the groupby keys are presorted
   */
  bool is_presorted() { return _keys_pre_sorted == sorted::YES; }

  /**
   * @brief Return the effective number of keys
   *
   * When include_null_keys = YES, returned value is same as `keys.num_rows()`
   * When include_null_keys = NO, returned value is the number of rows in `keys`
   *  in which no element is null
   */
  size_type num_keys(cuda::stream_ref stream);

  /**
   * @brief Get the sorted order of `keys`.
   *
   * Gathering `keys` by sort order indices will produce the sorted key table.
   *
   * When ignore_null_keys = true, the result will not include indices
   * for null keys.
   *
   * Computes and stores the key sorted order on first invocation, and returns
   * the stored order on subsequent calls.
   *
   * @return the sort order indices for `keys`.
   */
  column_view key_sort_order(cuda::stream_ref stream);

  /**
   * @brief Get each group's offset into the sorted order of `keys`.
   *
   * Computes and stores the group offsets on first invocation and returns
   * the stored group offsets on subsequent calls.
   * This returns a vector of size `num_groups() + 1` such that the size of
   * group `i` is `group_offsets[i+1] - group_offsets[i]`
   *
   * @return vector of offsets of the starting point of each group in the sorted
   * key table
   */
  index_vector const& group_offsets(cuda::stream_ref stream);

  /**
   * @brief Get the group labels corresponding to the sorted order of `keys`.
   *
   * Each group is assigned a unique numerical "label" in
   * `[0, 1, 2, ... , num_groups() - 1, num_groups(stream))`.
   * For a row in sorted `keys`, its corresponding group label indicates which
   * group it belongs to.
   *
   * Computes and stores labels on first invocation and returns stored labels on
   * subsequent calls.
   *
   * @return vector of group labels for each row in the sorted key column
   */
  index_vector const& group_labels(cuda::stream_ref stream);

 private:
  /**
   * @brief Get the group labels for unsorted keys
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
   *         of the unsorted key table
   */
  column_view unsorted_keys_labels(cuda::stream_ref stream);

  /**
   * @brief Get the column representing the row bitmask for the `keys`
   *
   * Computes a bitmask corresponding to the rows of `keys` where if bit `i` is
   * zero, then row `i` contains one or more null values. If bit `i` is one,
   * then row `i` does not contain null values. This bitmask is added as null
   * mask of a column of type `INT8` where all the data values are the same and
   * the elements differ only in validity.
   *
   * Computes and stores bitmask on first invocation and returns stored column
   * on subsequent calls.
   */
  column_view keys_bitmask_column(cuda::stream_ref stream);

  column_ptr _key_sorted_order;      ///< Indices to produce _keys in sorted order
  column_ptr _unsorted_keys_labels;  ///< Group labels for unsorted _keys
  column_ptr _keys_bitmask_column;   ///< Column representing rows with one or more nulls values
  table_view _keys;                  ///< Input keys to sort by

  index_vector_ptr
    _group_offsets;  ///< Indices into sorted _keys indicating starting index of each groups
  index_vector_ptr _group_labels;  ///< Group labels for sorted _keys

  size_type _num_keys;      ///< Number of effective rows in _keys (adjusted for _include_null_keys)
  sorted _keys_pre_sorted;  ///< Whether _keys are pre-sorted
  null_policy _include_null_keys;  ///< Whether to use rows with nulls in _keys for grouping
  std::vector<null_order> _null_precedence;  ///< How to sort NULLs
};

}  // namespace groupby::detail::sort
}  // namespace cudf
