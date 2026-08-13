/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/replace.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <span>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Class definitions for grouping and aggregating values within groups of rows.
 */

namespace CUDF_EXPORT cudf {
//! `groupby` APIs
namespace groupby {
namespace detail {
namespace sort {
struct sort_groupby_helper;

}  // namespace sort
}  // namespace detail

/**
 * @addtogroup aggregation_groupby
 * @{
 */

/**
 * @brief Request for groupby aggregation(s) to perform on a column.
 *
 * The group membership of each `value[i]` is determined by the corresponding
 * row `i` in the original order of `keys` used to construct the
 * `groupby`. I.e., for each `aggregation`, `values[i]` is aggregated with all
 * other `values[j]` where rows `i` and `j` in `keys` are equivalent.
 *
 * `values.size()` column must equal `keys.num_rows()`.
 */
struct aggregation_request {
  column_view values;                                              ///< The elements to aggregate
  std::vector<std::unique_ptr<groupby_aggregation>> aggregations;  ///< Desired aggregations
};

/**
 * @brief Request for groupby aggregation(s) for scanning a column.
 *
 * The group membership of each `value[i]` is determined by the corresponding
 * row `i` in the original order of `keys` used to construct the
 * `groupby`. I.e., for each `aggregation`, `values[i]` is aggregated with all
 * other `values[j]` where rows `i` and `j` in `keys` are equivalent.
 *
 * `values.size()` column must equal `keys.num_rows()`.
 */
struct scan_request {
  column_view values;  ///< The elements to aggregate
  std::vector<std::unique_ptr<groupby_scan_aggregation>> aggregations;  ///< Desired aggregations
};

/**
 * @brief The result(s) of an `aggregation_request`
 *
 * For every `aggregation_request` given to `groupby::aggregate` an
 * `aggregation_result` will be returned. The `aggregation_result` holds the
 * resulting column(s) for each requested aggregation on the `request`s values.
 */
struct aggregation_result {
  /// Columns of results from an `aggregation_request`
  std::vector<std::unique_ptr<column>> results{};
};

/**
 * @brief Groups values by keys and computes aggregations on those groups.
 */
class groupby {
 public:
  groupby() = delete;
  ~groupby();
  groupby(groupby const&)            = delete;
  groupby(groupby&&)                 = delete;
  groupby& operator=(groupby const&) = delete;
  groupby& operator=(groupby&&)      = delete;

  /**
   * @brief Construct a groupby object with the specified `keys`
   *
   * If the `keys` are already sorted, better performance may be achieved by
   * passing `keys_are_sorted == true` and indicating the  ascending/descending
   * order of each column and null order in  `column_order` and
   * `null_precedence`, respectively.
   *
   * @note This object does *not* maintain the lifetime of `keys`. It is the
   * user's responsibility to ensure the `groupby` object does not outlive the
   * data viewed by the `keys` `table_view`.
   *
   * @param keys Table whose rows act as the groupby keys
   * @param null_handling Indicates whether rows in `keys` that contain
   * NULL values should be included
   * @param keys_are_sorted Indicates whether rows in `keys` are already sorted
   * @param column_order If `keys_are_sorted == YES`, indicates whether each
   * column is ascending/descending. If empty, assumes all  columns are
   * ascending. Ignored if `keys_are_sorted == false`.
   * @param null_precedence If `keys_are_sorted == YES`, indicates the ordering
   * of null values in each column. Else, ignored. If empty, assumes all columns
   * use `null_order::AFTER`. Ignored if `keys_are_sorted == false`.
   */
  explicit groupby(table_view const& keys,
                   null_policy null_handling                      = null_policy::EXCLUDE,
                   sorted keys_are_sorted                         = sorted::NO,
                   std::vector<order> const& column_order         = {},
                   std::vector<null_order> const& null_precedence = {});

  /**
   * @brief Performs grouped aggregations on the specified values.
   *
   * The values to aggregate and the aggregations to perform are specified in an
   * `aggregation_request`. Each request contains a `column_view` of values to
   * aggregate and a set of `aggregation`s to perform on those elements.
   *
   * For each `aggregation` in a request, `values[i]` is aggregated with
   * all other `values[j]` where rows `i` and `j` in `keys` are equivalent.
   *
   * The `size()` of the request column must equal `keys.num_rows()`.
   *
   * For every `aggregation_request` an `aggregation_result` will be returned.
   * The `aggregation_result` holds the resulting column(s) for each requested
   * aggregation on the `request`s values. The order of the columns in each
   * result is the same order as was specified in the request.
   *
   * The returned `table` contains the group labels for each group, i.e., the
   * distinct rows from `keys`. Element `i` across all aggregation results
   * belongs to the group at row `i` in the group labels table.
   *
   * The order of the rows in the group labels is arbitrary. Furthermore,
   * successive `groupby::aggregate` calls may return results in different
   * orders.
   *
   * @throws cudf::logic_error If `requests[i].values.size() !=
   * keys.num_rows()`.
   *
   * Example:
   * ```
   * Input:
   * keys:     {1 2 1 3 1}
   *           {1 2 1 4 1}
   * request:
   *   values: {3 1 4 9 2}
   *   aggregations: {{SUM}, {MIN}}
   *
   * result:
   *
   * keys:  {3 1 2}
   *        {4 1 2}
   * values:
   *   SUM: {9 9 1}
   *   MIN: {9 2 1}
   * ```
   *
   * @param requests The set of columns to aggregate and the aggregations to
   * perform
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   * @return Pair containing the table with each group's distinct key and
   * a vector of aggregation_results for each request in the same order as
   * specified in `requests`.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> aggregate(
    std::span<aggregation_request const> requests,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
  /**
   * @brief Performs grouped scans on the specified values.
   *
   * The values to aggregate and the aggregations to perform are specified in an
   * `aggregation_request`. Each request contains a `column_view` of values to
   * aggregate and a set of `aggregation`s to perform on those elements.
   *
   * For each `aggregation` in a request, `values[i]` is scan aggregated with
   * all previous `values[j]` where rows `i` and `j` in `keys` are equivalent.
   *
   * The `size()` of the request column must equal `keys.num_rows()`.
   *
   * For every `aggregation_request` an `aggregation_result` will be returned.
   * The `aggregation_result` holds the resulting column(s) for each requested
   * aggregation on the `request`s values. The order of the columns in each
   * result is the same order as was specified in the request.
   *
   * The returned `table` contains the group labels for each row, i.e., the
   * `keys` given to groupby object. Element `i` across all aggregation results
   * belongs to the group at row `i` in the group labels table.
   *
   * The order of the rows in the group labels is arbitrary. Furthermore,
   * successive `groupby::scan` calls may return results in different orders.
   *
   * @throws cudf::logic_error If `requests[i].values.size() !=
   * keys.num_rows()`.
   *
   * Example:
   * ```
   * Input:
   * keys:     {1 2 1 3 1}
   *           {1 2 1 4 1}
   * request:
   *   values: {3 1 4 9 2}
   *   aggregations: {{SUM}, {MIN}}
   *
   * result:
   *
   * keys:  {3 1 1 1 2}
   *        {4 1 1 1 2}
   * values:
   *   SUM: {9 3 7 9 1}
   *   MIN: {9 3 3 2 1}
   * ```
   *
   * @param requests The set of columns to scan and the scans to perform
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   * @return Pair containing the table with each group's key and
   * a vector of aggregation_results for each request in the same order as
   * specified in `requests`.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> scan(
    std::span<scan_request const> requests,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Performs grouped shifts for specified values.
   *
   * In `j`th column, for each group, `i`th element is determined by the `i - offsets[j]`th
   * element of the group. If `i - offsets[j] < 0 or >= group_size`, the value is determined by
   * @p fill_values[j].
   *
   * @note The first returned table stores the keys passed to the groupby object. Row `i` of the key
   * table corresponds to the group labels of row `i` in the shifted columns. The key order in
   * each group matches the input order. The order of each group is arbitrary. The group order
   * in successive calls to `groupby::shifts` may be different.
   *
   * Example:
   * @code{.pseudo}
   * keys:    {1 4 1 3 4 4 1}
   *          {1 2 1 3 2 2 1}
   * values:  {3 9 1 4 2 5 7}
   *          {"a" "c" "bb" "ee" "z" "x" "d"}
   * offset:  {2, -1}
   * fill_value: {@, @}
   * result (group order maybe different):
   *    keys:   {3 1 1 1 4 4 4}
   *            {3 1 1 1 2 2 2}
   *    values: {@ @ @ 3 @ @ 9}
   *            {@ "bb" "d" @ "z" "x" @}
   *
   * -------------------------------------------------
   * keys:    {1 4 1 3 4 4 1}
   *          {1 2 1 3 2 2 1}
   * values:  {3 9 1 4 2 5 7}
   *          {"a" "c" "bb" "ee" "z" "x" "d"}
   * offset:  {-2, 1}
   * fill_value: {-1, "42"}
   * result (group order maybe different):
   *    keys:   {3 1 1 1 4 4 4}
   *            {3 1 1 1 2 2 2}
   *    values: {-1 7 -1 -1 5 -1 -1}
   *            {"42" "42" "a" "bb" "42" "c" "z"}
   *
   * @endcode
   *
   * @param values Table whose columns to be shifted
   * @param offsets The offsets by which to shift the input
   * @param fill_values Fill values for indeterminable outputs
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   * @return Pair containing the tables with each group's key and the columns shifted
   *
   * @throws cudf::logic_error if @p fill_value[i] dtype does not match @p values[i] dtype for
   * `i`th column
   */
  std::pair<std::unique_ptr<table>, std::unique_ptr<table>> shift(
    table_view const& values,
    std::span<size_type const> offsets,
    std::vector<std::reference_wrapper<scalar const>> const& fill_values,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief The grouped data corresponding to a groupby operation on a set of values.
   *
   * A `groups` object holds two tables of identical number of rows:
   * a table of grouped keys and a table of grouped values. In addition, it holds
   * a vector of integer offsets into the rows of the tables, such that
   * `offsets[i+1] - offsets[i]` gives the size of group `i`.
   */
  struct groups {
    std::unique_ptr<table> keys;     ///< Table of grouped keys
    std::vector<size_type> offsets;  ///< Group Offsets
    std::unique_ptr<table> values;   ///< Table of grouped values
  };

  /**
   * @brief Get the grouped keys and values corresponding to a groupby operation on a set of values
   *
   * Returns a `groups` object representing the grouped keys and values.
   * If values is not provided, only a grouping of the keys is performed,
   * and the `values` of the `groups` object will be `nullptr`.
   *
   * @param values Table representing values on which a groupby operation is to be performed
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned tables's device memory in the
   * returned groups
   * @return A `groups` object representing grouped keys and values
   */
  groups get_groups(cudf::table_view values           = {},
                    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Performs grouped replace nulls on @p value
   *
   * For each `value[i] == NULL` in group `j`, `value[i]` is replaced with the first non-null value
   * in group `j` that precedes or follows `value[i]`. If a non-null value is not found in the
   * specified direction, `value[i]` is left NULL.
   *
   * The returned pair contains a column of the sorted keys and the result column. In result column,
   * values of the same group are in contiguous memory. In each group, the order of values maintain
   * their original order. The order of groups are not guaranteed.
   *
   * Example:
   * @code{.pseudo}
   *
   * //Inputs:
   * keys:    {3 3 1 3 1 3 4}
   *          {2 2 1 2 1 2 5}
   * values:  {3 4 7 @ @ @ @}
   *          {@ @ @ "x" "tt" @ @}
   * replace_policies:    {FORWARD, BACKWARD}
   *
   * //Outputs (group orders may be different):
   * keys:    {3 3 3 3 1 1 4}
   *          {2 2 2 2 1 1 5}
   * result:  {3 4 4 4 7 7 @}
   *          {"x" "x" "x" @ "tt" "tt" @}
   * @endcode
   *
   * @param[in] values A table whose column null values will be replaced
   * @param[in] replace_policies Specify the position of replacement values relative to null values,
   * one for each column
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   * @param[in] mr Device memory resource used to allocate device memory of the returned column
   *
   * @return Pair that contains a table with the sorted keys and the result column
   */
  std::pair<std::unique_ptr<table>, std::unique_ptr<table>> replace_nulls(
    table_view const& values,
    std::span<cudf::replace_policy const> replace_policies,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

 private:
  table_view _keys;                                      ///< Keys that determine grouping
  null_policy _include_null_keys{null_policy::EXCLUDE};  ///< Include rows in keys
                                                         ///< with NULLs
  sorted _keys_are_sorted{sorted::NO};                   ///< Whether or not the keys are sorted
  std::vector<order> _column_order{};                    ///< If keys are sorted, indicates
                                                         ///< the order of each column
  std::vector<null_order> _null_precedence{};            ///< If keys are sorted,
                                                         ///< indicates null order
                                                         ///< of each column
  std::unique_ptr<detail::sort::sort_groupby_helper>
    _helper;  ///< Helper object
              ///< used by sort based implementation

  /**
   * @brief Get the sort helper object
   *
   * The object is constructed on first invocation and subsequent invocations
   * of this function return the memoized object.
   */
  detail::sort::sort_groupby_helper& helper();

  /**
   * @brief Dispatches to the appropriate implementation to satisfy the
   * aggregation requests.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> dispatch_aggregation(
    std::span<aggregation_request const> requests,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  // Sort-based groupby
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> sort_aggregate(
    std::span<aggregation_request const> requests,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> sort_scan(
    std::span<scan_request const> requests,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
};

/**
 * @brief Request for a single streaming groupby aggregation on a column.
 *
 * Analogous to `aggregation_request` but identifies the value column by index rather than
 * by `column_view`, since data arrives in batches after construction, and carries exactly
 * one aggregation per request.
 *
 * `column_index` refers to the position of the value column in the `table_view` passed
 * to `streaming_groupby::aggregate()`.  Multiple aggregations on the same column are
 * expressed as separate requests (e.g., `[{col, sum}, {col, mean}]`).  Internal
 * deduplication ensures redundant computations are shared automatically.
 */
struct streaming_aggregation_request {
  size_type column_index;                            ///< Index of the value column
  std::unique_ptr<groupby_aggregation> aggregation;  ///< Desired aggregation
};

/**
 * @brief Stateful streaming groupby that accumulates partial aggregates across batches.
 *
 * `streaming_groupby` and the stateless `groupby` serve different use cases.  Use
 * the stateless `groupby` for single-shot aggregation when all input fits in memory
 * at once.  Use `streaming_groupby` when input arrives over multiple batches and
 * memory efficiency matters: peak memory does not scale with the number of *input*
 * rows, because only the distinct keys seen so far and one aggregation slot per
 * group are kept across batches.  Arbitrarily long high-duplicate streams therefore
 * accumulate without running out of memory.
 *
 * If memory is not a concern, concatenating all batches and calling the stateless
 * `groupby` once is also a valid choice.  Reach for `streaming_groupby` when
 * (a) the cumulative input does not fit in memory, or (b) partial-state aggregation
 * across distributed workers (`merge()`) is part of the workload.
 *
 * Per-batch cost is O(batch_size): each batch does direct hash table insertion
 * and in-place aggregation updates against the persistent state. Partial states
 * can be combined via `merge()`, and final results are produced via `finalize()`.
 *
 * The `max_distinct_keys` parameter sets the upper bound on the number of distinct key
 * combinations across the lifetime of this object.  The persistent state is sized to
 * `max_distinct_keys` (constant for the lifetime of the object); the stored distinct
 * keys grow with the number of distinct keys actually seen, so the incremental key
 * storage is O(`distinct_keys()` × key_size) and does not scale with cumulative input
 * rows.
 *
 * Cumulative input rows are not bounded — only cumulative distinct keys.  A single
 * batch may also not exceed `max_distinct_keys` rows; this is an implementation
 * limit because each in-flight batch row is encoded as `max_distinct_keys + row_idx`
 * inside the hash set, which must fit in `cudf::size_type`.
 *
 * All column types (including variable-width types such as strings, lists, and structs)
 * are supported for key columns.  Only hash-based aggregation kinds are supported; use
 * `is_streaming_groupby_supported()` to query a specific (value type, aggregation kind)
 * combination.
 *
 * Supported aggregation kinds:
 *   SUM, SUM_OF_SQUARES, PRODUCT, MIN, MAX, COUNT_VALID, COUNT_ALL,
 *   MEAN, M2, VARIANCE, STD
 *
 * @throws std::invalid_argument for unsupported aggregation kinds
 * @throws std::invalid_argument if a single batch exceeds `max_distinct_keys` rows
 * @throws cudf::logic_error if cumulative distinct keys exceed `max_distinct_keys`
 */
class streaming_groupby {
 public:
  streaming_groupby() = delete;
  ~streaming_groupby();
  streaming_groupby(streaming_groupby const&)            = delete;
  streaming_groupby& operator=(streaming_groupby const&) = delete;

  /** @brief Move constructor. */
  streaming_groupby(streaming_groupby&&) noexcept;

  /**
   * @brief Move assignment operator.
   * @return Reference to this object.
   */
  streaming_groupby& operator=(streaming_groupby&&) noexcept;

  /**
   * @brief Construct a streaming groupby object with a persistent hash table.
   *
   * @param key_indices Indices of columns in the data table that serve as groupby keys
   * @param requests The aggregations to perform and which columns to aggregate
   * @param max_distinct_keys Upper bound on distinct key combinations. The hash set,
   *        companion vectors, and aggregation results table are all sized to this
   *        capacity. Cumulative input rows are not bounded.
   * @param null_handling Indicates whether rows in keys that contain NULL values should be included
   * @param mr Device memory resource used to allocate the persistent hash table
   *
   * @throws std::invalid_argument if `max_distinct_keys <= 0`
   * @throws std::invalid_argument if any requested aggregation kind is unsupported
   */
  explicit streaming_groupby(host_span<size_type const> key_indices,
                             host_span<streaming_aggregation_request const> requests,
                             size_type max_distinct_keys,
                             null_policy null_handling = null_policy::EXCLUDE,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr =
                               cudf::get_current_device_resource_ref());

  /**
   * @brief Feed a batch of data into the streaming aggregation.
   *
   * Batch keys are inserted into the persistent hash set and aggregation results
   * are updated atomically. The input `data` table is not referenced after this
   * call returns.
   *
   * @param data Table containing both key and value columns
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @throws std::invalid_argument if `data.num_rows()` exceeds `max_distinct_keys`
   * @throws cudf::logic_error if cumulative distinct keys exceed `max_distinct_keys`
   */
  void aggregate(table_view const& data, rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Merge another streaming_groupby's accumulated partial state into this one.
   *
   * Extracts the other object's accumulated intermediate state and merges it into this
   * object's persistent hash table. The other object is not modified.
   * Both objects must have been constructed with compatible aggregation requests,
   * and this object must have had at least one `aggregate()` call.
   *
   * @param other The streaming_groupby whose partial state to merge
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @throws std::invalid_argument if the other object has more distinct keys than
   * `max_distinct_keys`
   * @throws cudf::logic_error if this object has not been initialized via `aggregate()`
   * @throws cudf::logic_error if distinct keys exceed `max_distinct_keys` after merge
   */
  void merge(streaming_groupby const& other,
             rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Finalize the accumulated partial aggregates into final results.
   *
   * For most aggregation kinds the partial state is the final result. For kinds like
   * MEAN, VARIANCE, or STD, a finalization step converts the internal partial representation
   * (e.g., sum+count) into the user-facing result.
   *
   * This does not modify the internal state; `aggregate()` may be called again afterward.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns
   * @return Pair of distinct keys table and a vector of aggregation_results (one per request)
   *
   * @throws cudf::logic_error if no data has been accumulated
   */
  [[nodiscard]] std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> finalize(
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns the number of distinct keys accumulated so far.
   *
   * Returns 0 before any successful `aggregate()` or `merge()` call.
   *
   * @return The current count of distinct keys in the persistent hash table
   */
  [[nodiscard]] size_type distinct_keys() const noexcept;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;

  void do_aggregate(table_view const& data, rmm::cuda_stream_view stream);
  void do_merge(streaming_groupby const& other, rmm::cuda_stream_view stream);
  [[nodiscard]] std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> do_finalize(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const;
};

/**
 * @brief Returns true if `streaming_groupby` supports the given value type and
 * aggregation kind combination.
 *
 * Use this to query support without constructing a `streaming_groupby`.  A `true`
 * return implies that an `aggregate()` call with a value column of `values_type` and
 * an aggregation of `kind` will not be rejected on type/kind grounds.
 *
 * @param values_type Type of the value column the aggregation would run on
 * @param kind Aggregation kind
 * @return True if the combination is supported, false otherwise
 */
[[nodiscard]] bool is_streaming_groupby_supported(data_type values_type, aggregation::Kind kind);

/** @} */
}  // namespace groupby
}  // namespace CUDF_EXPORT cudf
