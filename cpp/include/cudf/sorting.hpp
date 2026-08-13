/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <vector>

/**
 * @file
 * @brief Column APIs for sort and rank
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_sort
 * @{
 */

/**
 * @brief Computes the row indices that would produce `input` in a lexicographical sorted order.
 *
 * @param input The table to sort
 * @param column_order The desired sort order for each column. Size must be
 * equal to `input.num_columns()` or empty. If empty, all columns will be sorted
 * in ascending order.
 * @param null_precedence The desired order of null compared to other elements
 * for each column. Size must be equal to `input.num_columns()` or empty.
 * If empty, all columns will be sorted in `null_order::BEFORE`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A non-nullable column of elements containing the permuted row indices of
 * `input` if it were sorted
 */
std::unique_ptr<column> sorted_order(
  table_view const& input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the row indices that would produce `input` in a stable
 * lexicographical sorted order.
 *
 * The order of equivalent elements is guaranteed to be preserved.
 *
 * @copydoc cudf::sorted_order
 */
std::unique_ptr<column> stable_sorted_order(
  table_view const& input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Checks whether the rows of a `table` are sorted in a lexicographical
 *        order.
 *
 * @param table             Table whose rows need to be compared for ordering
 * @param column_order      The expected sort order for each column. Size
 *                          must be equal to `in.num_columns()` or empty. If
 *                          empty, it is expected all columns are in
 *                          ascending order.
 * @param null_precedence   The desired order of null compared to other
 *                          elements for each column. Size must be equal to
 *                          `input.num_columns()` or empty. If empty,
 *                          `null_order::BEFORE` is assumed for all columns.
 *
 * @param stream            CUDA stream used for device memory operations and kernel launches
 * @returns                 true if sorted as expected, false if not
 */
bool is_sorted(cudf::table_view const& table,
               std::vector<order> const& column_order,
               std::vector<null_order> const& null_precedence,
               rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Performs a lexicographic sort of the rows of a table
 *
 * @param input The table to sort
 * @param column_order The desired order for each column. Size must be
 * equal to `input.num_columns()` or empty. If empty, all columns are sorted in
 * ascending order.
 * @param null_precedence The desired order of a null element compared to other
 * elements for each column in `input`. Size must be equal to
 * `input.num_columns()` or empty. If empty, all columns will be sorted with
 * `null_order::BEFORE`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return New table containing the desired sorted order of `input`
 */
std::unique_ptr<table> sort(
  table_view const& input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a stable lexicographic sort of the rows of a table
 *
 * @copydoc cudf::sort
 */
std::unique_ptr<table> stable_sort(
  table_view const& input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a key-value sort.
 *
 * Creates a new table that reorders the rows of `values` according to the
 * lexicographic ordering of the rows of `keys`.
 *
 * @throws cudf::logic_error if `values.num_rows() != keys.num_rows()`.
 *
 * @param values The table to reorder
 * @param keys The table that determines the ordering
 * @param column_order The desired order for each column in `keys`. Size must be
 * equal to `keys.num_columns()` or empty. If empty, all columns are sorted in
 * ascending order.
 * @param null_precedence The desired order of a null element compared to other
 * elements for each column in `keys`. Size must be equal to
 * `keys.num_columns()` or empty. If empty, all columns will be sorted with
 * `null_order::BEFORE`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return The reordering of `values` determined by the lexicographic order of
 * the rows of `keys`.
 */
std::unique_ptr<table> sort_by_key(
  table_view const& values,
  table_view const& keys,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a key-value stable sort.
 *
 * @copydoc cudf::sort_by_key
 */
std::unique_ptr<table> stable_sort_by_key(
  table_view const& values,
  table_view const& keys,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the ranks of input column in sorted order.
 *
 * Rank indicate the position of each element in the sorted column and rank
 * value starts from 1.
 *
 * @code{.pseudo}
 * Using default order::ASCENDING
 * input = {3, 4, 5, 4, 1, 2}
 * Results for different rank_methods
 * FIRST    = {3, 4, 6, 5, 1, 2}
 * AVERAGE  = {3, 4.5, 6, 4.5, 1, 2}
 * MIN      = {3, 4, 6, 4, 1, 2}
 * MAX      = {3, 5, 6, 5, 1, 2}
 * DENSE    = {3, 4, 5, 4, 1, 2}
 *
 * For null_policy::INCLUDE, null_order::AFTER
 * input = {3, 4, null, 4, 1, 2}
 * The results are the same as above.
 *
 * For null_policy::INCLUDE, null_order::BEFORE
 * input = {3, 4, null, 4, 1, 2}
 * Results for different rank_methods
 * FIRST    = {4, 5, 1, 6, 2, 3}
 * AVERAGE  = {4, 5.5, 1, 5.5, 2, 3}
 * MIN      = {4, 5, 1, 5, 2, 3}
 * MAX      = {4, 6, 1, 6, 2, 3}
 * DENSE    = {4, 5, 1, 5, 2, 3}
 *
 * For null_policy::EXCLUDE (null_order::AFTER only)
 * input = {3, 4, null, 4, 1, 2}
 * Results for different rank_methods
 * FIRST    = {3, 4, null, 5, 1, 2}
 * AVERAGE  = {3, 4.5, null, 4.5, 1, 2}
 * MIN      = {3, 4, null, 4, 1, 2}
 * MAX      = {3, 5, null, 5, 1, 2}
 * DENSE    = {3, 4, null, 4, 1, 2}
 * @endcode
 *
 * For null_policy::EXCLUDE with null_order::BEFORE, using column_order::ASCENDING
 * will result in undefined behavior. Likewise for null_policy::EXCLUDE with
 * null_order::AFTER and column_order::DESCENDING.
 *
 * The output column type will be `double` when `method=rank_method::AVERAGE` or `percentage=True`
 * and `size_type` otherwise.
 *
 * @param input The column to rank
 * @param method The ranking method used for tie breaking (same values)
 * @param column_order The desired sort order for ranking
 * @param null_handling Flag to include nulls during ranking.
 *                      If nulls are excluded, the corresponding rank will be null.
 * @param null_precedence The desired order of null rows compared to other elements
 * @param percentage Flag to convert ranks to percentage in range (0,1]
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column of containing the rank of the each element of the column of `input`
 */
std::unique_ptr<column> rank(
  column_view const& input,
  rank_method method,
  order column_order,
  null_policy null_handling,
  null_order null_precedence,
  bool percentage,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns sorted order after sorting each segment in the table.
 *
 * If segment_offsets contains values larger than the number of rows, the behavior is undefined.
 * @throws cudf::logic_error if `segment_offsets` is not `size_type` column.
 *
 * @code{.pseudo}
 * Example:
 * keys = { {9, 8, 7, 6, 5, 4, 3, 2, 1, 0} }
 * offsets = {0, 3, 7, 10}
 * result = cudf::segmented_sorted_order(keys, offsets);
 * result is { 2,1,0, 6,5,4,3, 9,8,7 }
 * @endcode
 *
 * If segment_offsets is empty or contains a single index, no values are sorted
 * and the result is a sequence of integers from 0 to keys.size()-1.
 *
 * The segment_offsets are not required to include all indices. Any indices
 * outside the specified segments will not be sorted.
 *
 * @code{.pseudo}
 * Example: (offsets do not cover all indices)
 * keys = { {9, 8, 7, 6, 5, 4, 3, 2, 1, 0} }
 * offsets = {3, 7}
 * result = cudf::segmented_sorted_order(keys, offsets);
 * result is { 0,1,2, 6,5,4,3, 7,8,9 }
 * @endcode
 *
 * A zero-column `keys` table is treated as empty, producing an empty column even when `keys` has a
 * non-zero row count.
 *
 * @param keys The table that determines the ordering of elements in each segment
 * @param segment_offsets The column of `size_type` type containing start offset index for each
 * contiguous segment.
 * @param column_order The desired order for each column in `keys`. Size must be
 * equal to `keys.num_columns()` or empty. If empty, all columns are sorted in
 * ascending order.
 * @param null_precedence The desired order of a null element compared to other
 * elements for each column in `keys`. Size must be equal to
 * `keys.num_columns()` or empty. If empty, all columns will be sorted with
 * `null_order::BEFORE`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to allocate any returned objects
 *
 * @return sorted order of the segment sorted table
 *
 */
std::unique_ptr<column> segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Returns sorted order after stably sorting each segment in the table.
 *
 * @copydoc cudf::segmented_sorted_order
 */
std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a lexicographic segmented sort of a table
 *
 * If segment_offsets contains values larger than the number of rows, the behavior is undefined.
 * @throws cudf::logic_error if `values.num_rows() != keys.num_rows()`.
 * @throws cudf::logic_error if `segment_offsets` is not `size_type` column.
 *
 * @code{.pseudo}
 * Example:
 * keys = { {9, 8, 7, 6, 5, 4, 3, 2, 1, 0} }
 * values = { {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'} }
 * offsets = {0, 3, 7, 10}
 * result = cudf::segmented_sort_by_key(keys, values, offsets);
 * result is { 'c','b','a', 'g','f','e','d', 'j','i','h' }
 * @endcode
 *
 * If segment_offsets is empty or contains a single index, no values are sorted
 * and the result is a copy of the values.
 *
 * The segment_offsets are not required to include all indices. Any indices
 * outside the specified segments will not be sorted.
 *
 * @code{.pseudo}
 * Example: (offsets do not cover all indices)
 * keys = { {9, 8, 7, 6, 5, 4, 3, 2, 1, 0} }
 * values = { {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'} }
 * offsets = {3, 7}
 * result = cudf::segmented_sort_by_key(keys, values, offsets);
 * result is { 'a','b','c', 'g','f','e','d', 'h','i','j' }
 * @endcode
 *
 * @param values The table to reorder
 * @param keys The table that determines the ordering of elements in each segment
 * @param segment_offsets The column of `size_type` type containing start offset index for each
 * contiguous segment.
 * @param column_order The desired order for each column in `keys`. Size must be
 * equal to `keys.num_columns()` or empty. If empty, all columns are sorted in
 * ascending order.
 * @param null_precedence The desired order of a null element compared to other
 * elements for each column in `keys`. Size must be equal to
 * `keys.num_columns()` or empty. If empty, all columns will be sorted with
 * `null_order::BEFORE`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to allocate any returned objects
 * @return table with elements in each segment sorted
 *
 */
std::unique_ptr<table> segmented_sort_by_key(
  table_view const& values,
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a stably lexicographic segmented sort of a table
 *
 * @copydoc cudf::segmented_sort_by_key
 */
std::unique_ptr<table> stable_segmented_sort_by_key(
  table_view const& values,
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr              = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the top k values of a column
 *
 * This performs the equivalent of a sort and the slice of the resulting first k elements.
 * However, the returned column may or may not necessarily be sorted.
 *
 * @throw std::invalid_argument if k is greater than the number of rows in the column
 *
 * @param col Column to compute top k
 * @param k Number of values to return
 * @param topk_order The desired sort order for the top k values.
 *                   Default is high to low.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column with the top k values of the input column.
 */
std::unique_ptr<column> top_k(
  column_view const& col,
  size_type k,
  order topk_order                  = order::DESCENDING,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the indices of the top k values of a column
 *
 * The indices will represent the top k elements but may or may not represent
 * those elements as k sorted values.
 *
 * @throw std::invalid_argument if k is greater than the number of rows in the column
 *
 * @param col Column to compute top k
 * @param k Number of values to return
 * @param topk_order The desired sort order for the top k values.
 *                   Default is high to low.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Indices of the top k values of the input column
 */
std::unique_ptr<column> top_k_order(
  column_view const& col,
  size_type k,
  order topk_order                  = order::DESCENDING,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the top k values within each segment of a column
 *
 * Returns the top k values (largest or smallest) within each segment of the given column.
 * The values within each segment may not necessarily be sorted.
 * If a segment contain less than k elements then all values for that segment are returned.
 *
 * @code{.pseudo}
 * Example:
 * col = [ 3, 4, 5, 4, 1, 2, 3, 5, 6, 7, 8, 9, 10 ]
 * offsets = [0, 3, 7, 13]
 * result = cudf::segmented_top_k(col, offsets, 3);
 * result is [[5,4,3], [4,3,2], [10,8,9]] // each segment may not be sorted
 * @endcode
 *
 * The segment_offsets are not required to include all indices. Any indices outside the specified
 * segments are excluded from the result.
 *
 * @code{.pseudo}
 * Example: (offsets do not cover all indices)
 * col = [ 3, 4, 5, 4, 1, 2, 3, 5, 6, 7, 8, 9, 10 ]
 * offsets = [3, 10]
 * result = cudf::segmented_top_k(col, offsets, 3);
 * result is [[7,6,5]] // rows 0-2 and 10-12 are excluded
 * @endcode
 *
 * @throw std::invalid_argument if k less than or equal to zero
 * @throw cudf::data_type_error if segment_offsets is not size_type
 * @throw std::invalid_argument segments_offsets is empty or contains nulls
 *
 * @param col Column to compute top k
 * @param segment_offsets Start offset index for each contiguous segment
 * @param k Number of values to return for each segment
 * @param topk_order DESCENDING is the largest k values (default).
 *                   ASCENDING is the smallest k values.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column with the top k values of the input column.
 */
std::unique_ptr<column> segmented_top_k(
  column_view const& col,
  column_view const& segment_offsets,
  size_type k,
  order topk_order                  = order::DESCENDING,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the indices of the top k values within each segment of a column
 *
 * The indices will represent the top k elements within each segment but may not represent
 * those elements as k sorted values.
 * If a segment contain less than k elements then all values for that segment are returned.
 *
 * @code{.pseudo}
 * Example:
 * col = [ 3, 4, 5, 4, 1, 2, 3, 5, 6, 7, 8, 9, 10 ]
 * offsets = [0, 3, 7, 13]
 * result = cudf::segmented_top_k_order(col, offsets, 3);
 * result is [[2,1,0], [3,6,5], [12,10,11]] // each segment may not be sorted
 * @endcode
 *
 * The segment_offsets are not required to include all indices. Any indices outside the specified
 * segments are excluded from the result.
 *
 * @code{.pseudo}
 * Example: (offsets do not cover all indices)
 * col = [ 3, 4, 5, 4, 1, 2, 3, 5, 6, 7, 8, 9, 10 ]
 * offsets = [3, 10]
 * result = cudf::segmented_top_k_order(col, offsets, 3);
 * result is [[9,8,7]] // rows 0-2 and 10-12 are excluded
 * @endcode
 *
 * @throw std::invalid_argument if k less than or equal to zero
 * @throw cudf::data_type_error if segment_offsets is not size_type
 * @throw std::invalid_argument segments_offsets is empty or contains nulls
 *
 * @param col Column to compute top k
 * @param segment_offsets Start offset index for each contiguous segment
 * @param k Number of values to return for each segment
 * @param topk_order DESCENDING is the indices of the largest k values (default).
 *                   ASCENDING is the indices of the smallest k values.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Indices of the top k values of the input column
 */
std::unique_ptr<column> segmented_top_k_order(
  column_view const& col,
  column_view const& segment_offsets,
  size_type k,
  order topk_order                  = order::DESCENDING,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
