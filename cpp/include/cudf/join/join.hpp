/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/limits>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

/**
 * @file
 * @brief Common types and utilities shared by cuDF's join APIs.
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

/**
 * @brief Specifies the type of join operation to perform.
 *
 * This enum is used to control the behavior of join operations, particularly
 * in functions like filter_join_indices() that need to apply different logic
 * based on the join semantics.
 */
enum class join_kind : int32_t {
  INNER_JOIN     = 0,  ///< Inner join: only matching rows from both tables
  LEFT_JOIN      = 1,  ///< Left join: all rows from left table plus matching rows from right
  FULL_JOIN      = 2,  ///< Full outer join: all rows from both tables
  LEFT_SEMI_JOIN = 3,  ///< Left semi join: left rows that have matches in right table
  LEFT_ANTI_JOIN = 4   ///< Left anti join: left rows that have no matches in right table
};

/**
 * @brief Specifies whether a join implementation should apply an optional prefilter that reduces
 *        candidate rows before probing the hash table.
 *
 * `NO` preserves the current direct probe behavior. `YES` enables implementation-defined
 * prefiltering, such as bloom-filter-based candidate reduction before probing a hash table.
 */
enum class join_prefilter : bool { NO = false, YES = true };

/**
 * @brief Sentinel value used to indicate an unmatched row index in join operations.
 *
 * This value is used in join result indices to represent rows that do not have a match
 * in the other table (e.g., in left joins, full joins, or when using filter_gather_map
 * with null indices from outer joins).
 *
 * The value is set to the minimum possible value for `size_type` to ensure it's easily
 * distinguishable from valid row indices, which are always non-negative.
 */
CUDF_HOST_DEVICE constexpr size_type JoinNoMatch = cuda::std::numeric_limits<size_type>::min();

/**
 * @brief Holds context information about matches between tables during a join operation.
 *
 * This structure stores the left table view and a device vector containing the count of
 * matching rows in the right table for each row in the left table. Used primarily by
 * inner_join_match_context() to track join match information.
 */
struct join_match_context {
  table_view _left_table;  ///< View of the left table involved in the join operation
  std::unique_ptr<rmm::device_uvector<size_type>>
    _match_counts;  ///< A device vector containing the count of matching rows in the right table
                    ///< for each row in left table

  /**
   * @brief Construct a join_match_context
   *
   * @param left_table View of the left table involved in the join operation
   * @param match_counts Device vector containing the count of matching rows in the right table
   *                     for each row in the left table
   */
  join_match_context(table_view const& left_table,  // NOLINT(modernize-pass-by-value)
                     std::unique_ptr<rmm::device_uvector<size_type>> match_counts)
    : _left_table{left_table}, _match_counts{std::move(match_counts)}
  {
  }
  join_match_context(join_match_context const&)            = delete;
  join_match_context& operator=(join_match_context const&) = delete;
  join_match_context(join_match_context&&)                 = default;  ///< Move constructor
  /**
   * @brief Move assignment operator
   * @return Reference to this object
   */
  join_match_context& operator=(join_match_context&&) = default;
  virtual ~join_match_context() = default;  ///< Virtual destructor for proper polymorphic deletion
};

/**
 * @brief Stores context information for partitioned join operations.
 *
 * This structure maintains context for partitioned join operations, containing the match
 * context from a previous join operation along with the start and end indices that define
 * the current partition of the left table being processed.
 *
 * Used with partitioned_inner_join() to perform large joins in smaller chunks while
 * preserving the context from the initial match operation.
 */
struct join_partition_context {
  std::unique_ptr<join_match_context>
    left_table_context;      ///< The match context from a previous inner_join_match_context call
  size_type left_start_idx;  ///< The starting row index of the current left table partition
  size_type left_end_idx;  ///< The ending row index (exclusive) of the current left table partition
};

/**
 * @brief Returns a pair of row index vectors corresponding to an
 * inner join between the specified tables.
 *
 * The first returned vector contains the row indices from the left
 * table that have a match in the right table (in unspecified order).
 * The corresponding values in the second returned vector are
 * the matched row indices from the right table.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{1, 2}, {0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{1}, {0}}
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing an inner join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(cudf::table_view const& left_keys,
           cudf::table_view const& right_keys,
           null_equality compare_nulls       = null_equality::EQUAL,
           cuda::stream_ref stream           = cudf::get_default_stream(),
           rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to a
 * left join between the specified tables.
 *
 * The first returned vector contains all the row indices from the left
 * table (in unspecified order). The corresponding value in the
 * second returned vector is either (1) the row index of the matched row
 * from the right table, if there is a match  or  (2) `JoinNoMatch`.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{0, 1, 2}, {None, 0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{0, 1, 2}, {None, 0, None}}
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a left join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
left_join(cudf::table_view const& left_keys,
          cudf::table_view const& right_keys,
          null_equality compare_nulls       = null_equality::EQUAL,
          cuda::stream_ref stream           = cudf::get_default_stream(),
          rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to a
 * full join between the specified tables.
 *
 * Taken pairwise, the values from the returned vectors are one of:
 * (1) row indices corresponding to matching rows from the left and
 * right tables, (2) a row index and `JoinNoMatch`,
 * representing a row from one table without a match in the other.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{0, 1, 2, None}, {None, 0, 1, 2}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{0, 1, 2, None, None}, {None, 0, None, 1, 2}}
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a full join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
full_join(cudf::table_view const& left_keys,
          cudf::table_view const& right_keys,
          null_equality compare_nulls       = null_equality::EQUAL,
          cuda::stream_ref stream           = cudf::get_default_stream(),
          rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a cross join on two tables (`left`, `right`)
 *
 * The cross join returns the cartesian product of rows from each table.
 *
 * The result has `left.num_rows() * right.num_rows()` rows and
 * `left.num_columns() + right.num_columns()` columns. Either operand may have
 * zero columns and still contribute its row count.
 *
 * @note Warning: This function can easily cause out-of-memory errors. The size of the output is
 * equal to `left.num_rows() * right.num_rows()`. Use with caution.
 *
 * @code{.pseudo}
 * Left a: {0, 1, 2}
 * Right b: {3, 4, 5}
 * Result: { a: {0, 0, 0, 1, 1, 1, 2, 2, 2}, b: {3, 4, 5, 3, 4, 5, 3, 4, 5} }
 * @endcode
 *
 * @throw std::overflow_error if `left.num_rows() * right.num_rows()` exceeds the maximum
 * number of rows a column can hold.
 *
 * @param left  The left table
 * @param right The right table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr    Device memory resource used to allocate the returned table's device memory
 *
 * @return     Result of cross joining `left` and `right` tables
 */
std::unique_ptr<cudf::table> cross_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Filters join result indices based on a conditional predicate and join type.
 *
 * This function takes the result indices from a hash/sort join operation and applies
 * a conditional predicate to filter the pairs. It enables implementing mixed joins
 * as a two-step process: equality-based join followed by conditional filtering.
 *
 * The behavior depends on the join type:
 * - INNER_JOIN: Only pairs that satisfy the predicate and have valid indices are kept.
 * - LEFT_JOIN: All left rows are preserved. Failed predicates nullify right indices.
 * - FULL_JOIN: All rows from both sides are preserved. Failed predicates create separate pairs.
 *
 * Note on JoinNoMatch pairs: If an input pair already contains `JoinNoMatch` in either
 * position, the predicate cannot be evaluated and the pair passes through unchanged. The
 * "separate pairs" splitting only occurs when both indices are valid but the predicate fails.
 * For example, a FULL_JOIN pair `(5, 10)` that fails the predicate becomes two pairs:
 * `(5, JoinNoMatch)` and `(JoinNoMatch, 10)`, ensuring both rows appear in the output.
 *
 * ## Usage Pattern
 *
 * Typical usage involves performing an equality-based hash join first, then filtering
 * the results with a conditional predicate:
 *
 * @code{.cpp}
 * // Step 1: Perform equality-based hash join
 * auto hash_joiner = cudf::hash_join(right_equality_table, null_equality::EQUAL);
 * auto [left_indices, right_indices] = hash_joiner.inner_join(left_equality_table);
 *
 * // Step 2: Apply conditional filter on conditional columns
 * auto [filtered_left, filtered_right] = cudf::filter_join_indices(
 *   left_conditional_table,   // Table with columns referenced by predicate
 *   right_conditional_table,  // Table with columns referenced by predicate
 *   *left_indices,           // Indices from hash join
 *   *right_indices,          // Indices from hash join
 *   predicate,               // AST expression: e.g., left.col0 > right.col0
 *   cudf::join_kind::INNER_JOIN);
 * @endcode
 *
 * ## Example
 * @code{.pseudo}
 * Left equality:    {id: [1, 2, 3]}
 * Right equality:   {id: [1, 2, 3]}
 * Left conditional: {val: [10, 20, 30]}
 * Right conditional:{val: [15, 15, 25]}
 *
 * Hash join (id == id): left_indices = {0, 1, 2}, right_indices = {0, 1, 2}
 * Predicate: left.val > right.val
 *
 * INNER_JOIN result: left_indices = {1, 2}, right_indices = {1, 2}  // 20>15, 30>25
 * LEFT_JOIN result:  left_indices = {0, 1, 2}, right_indices = {JoinNoMatch, 1, 2}
 * @endcode
 *
 *
 * @throw std::invalid_argument if join_kind is not INNER_JOIN, LEFT_JOIN, or FULL_JOIN.
 * @throw std::invalid_argument if left_indices and right_indices have different sizes.
 * @throw std::invalid_argument if predicate does not produce a Boolean output.
 *
 * @param left The left table for predicate evaluation (conditional columns only).
 * @param right The right table for predicate evaluation (conditional columns only).
 * @param left_indices Device span of row indices in the left table from hash join.
 * @param right_indices Device span of row indices in the right table from hash join.
 * @param predicate An AST expression that returns a boolean for each pair of rows.
 * @param join_kind The type of join operation. Must be INNER_JOIN, LEFT_JOIN, or FULL_JOIN.
 * @param output_size Optional precomputed number of output rows. When provided, skips the internal
 *        size-counting pass. Behavior is undefined if it differs from the size the function would
 *        otherwise produce for the same inputs.
 * @param stream CUDA stream used for kernel launches and memory operations.
 * @param mr Device memory resource used to allocate output indices.
 *
 * @return A pair of device vectors [filtered_left_indices, filtered_right_indices]
 *         corresponding to rows that satisfy the join semantics and predicate.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices(cudf::table_view const& left,
                    cudf::table_view const& right,
                    cudf::device_span<size_type const> left_indices,
                    cudf::device_span<size_type const> right_indices,
                    cudf::ast::expression const& predicate,
                    cudf::join_kind join_kind,
                    std::optional<std::size_t> output_size = std::nullopt,
                    cuda::stream_ref stream                = cudf::get_default_stream(),
                    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact output size of `filter_join_indices` without materializing
 *        the filtered index vectors.
 *
 * Runs the same predicate evaluation as `filter_join_indices` but skips the index
 * materialization step, returning the total number of pairs that would be emitted along with the
 * per-output contribution counts whose sum is that total. The counts are laid out per `join_kind`
 * so that each entry records how many output rows the corresponding input contributes:
 * - INNER_JOIN: indexed per input pair; entry `i` is `1` if the predicate passes and `0` otherwise.
 * - FULL_JOIN: indexed per input pair; entry `i` is `1` for a preserved pair (predicate passes or
 *   the pair already contains a `JoinNoMatch`) and `2` for a failed valid pair (which splits into
 *   `(left, JoinNoMatch)` and `(JoinNoMatch, right)`).
 * - LEFT_JOIN: indexed per left row; each entry holds the number of passing pairs for that left
 *   row, floored to `1` to account for the synthetic `(left, JoinNoMatch)` entry.
 *
 * The returned size and contribution counts may be passed as a precomputed hint to APIs that
 * compose `filter_join_indices` (for example, the mixed join APIs). The layout above is an
 * implementation detail that callers should treat as opaque rather than rely upon.
 *
 * @throw std::invalid_argument if `join_kind` is not INNER_JOIN, LEFT_JOIN, or FULL_JOIN.
 * @throw std::invalid_argument if `left_indices` and `right_indices` have different sizes.
 * @throw std::invalid_argument if `predicate` does not produce a Boolean output.
 *
 * @param left The left table for predicate evaluation (conditional columns only).
 * @param right The right table for predicate evaluation (conditional columns only).
 * @param left_indices Device span of row indices in the left table.
 * @param right_indices Device span of row indices in the right table.
 * @param predicate An AST expression that returns a boolean for each pair of rows.
 * @param join_kind The type of join operation. Must be INNER_JOIN, LEFT_JOIN, or FULL_JOIN.
 * @param stream CUDA stream used for kernel launches and memory operations.
 * @param mr Device memory resource used to allocate the returned contribution counts.
 *
 * @return A pair containing the exact number of pairs that `filter_join_indices` would produce
 *         and the per-output contribution counts that sum to that number.
 */
[[nodiscard]] std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_output_size(
  cudf::table_view const& left,
  cudf::table_view const& right,
  cudf::device_span<size_type const> left_indices,
  cudf::device_span<size_type const> right_indices,
  cudf::ast::expression const& predicate,
  cudf::join_kind join_kind,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief JIT-based filtering of join result indices using string predicate.
 *
 * This function provides a JIT-compiled alternative to filter_join_indices(),
 * taking a string-based predicate that gets compiled to optimized GPU code.
 *
 * The behavior depends on the join type (same as filter_join_indices):
 * - INNER_JOIN: Only pairs that satisfy the predicate and have valid indices are kept.
 * - LEFT_JOIN: All left rows are preserved. Failed predicates nullify right indices.
 * - FULL_JOIN: All rows from both sides are preserved. Failed predicates create separate pairs.
 *
 * ## Usage Pattern
 *
 * Similar to filter_join_indices but uses JIT compilation for better performance:
 *
 * @code{.cpp}
 * // Step 1: Perform equality-based hash join (same as before)
 * auto hash_joiner = cudf::hash_join(right_equality_table, null_equality::EQUAL);
 * auto [left_indices, right_indices] = hash_joiner.inner_join(left_equality_table);
 *
 * // Step 2: Apply JIT-compiled conditional filter
 * std::string predicate_code = R"(
 *   __device__ void predicate(bool* output, double left_val, double right_val) {
 *     *output = left_val > right_val;
 *   }
 * )";
 * auto [filtered_left, filtered_right] = cudf::filter_join_indices_jit(
 *   left_conditional_table,   // Table with columns referenced by predicate
 *   right_conditional_table,  // Table with columns referenced by predicate
 *   *left_indices,           // Indices from hash join
 *   *right_indices,          // Indices from hash join
 *   predicate_code,          // JIT-compiled predicate function
 *   cudf::join_kind::INNER_JOIN);
 * @endcode
 *
 * ## Predicate Function Requirements
 *
 * The predicate_code must define a device function with signature:
 * ```cpp
 * __device__ void predicate(bool* output, T1 left_col0, T2 left_col1, ..., T1 right_col0, T2
 * right_col1, ...)
 * ```
 * The first parameter is a pointer to a bool that the function must set to `true` or `false`.
 * The remaining parameters correspond to columns in the left table followed by the right table.
 *
 * @throw std::invalid_argument if join_kind is not INNER_JOIN, LEFT_JOIN, or FULL_JOIN.
 * @throw std::invalid_argument if left_indices and right_indices have different sizes.
 * @throw cudf::jit_compilation_error if predicate_code fails to compile.
 *
 * @param left The left table for predicate evaluation (conditional columns only).
 * @param right The right table for predicate evaluation (conditional columns only).
 * @param left_indices Device span of row indices in the left table from hash join.
 * @param right_indices Device span of row indices in the right table from hash join.
 * @param predicate_code String containing CUDA device code for predicate function.
 * @param join_kind The type of join operation. Must be INNER_JOIN, LEFT_JOIN, or FULL_JOIN.
 * @param is_ptx Whether predicate_code contains PTX assembly instead of CUDA C++.
 * @param stream CUDA stream used for kernel launches and memory operations.
 * @param mr Device memory resource used to allocate output indices.
 *
 * @return A pair of device vectors [filtered_left_indices, filtered_right_indices]
 *         corresponding to rows that satisfy the join semantics and predicate.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(
  cudf::table_view const& left,
  cudf::table_view const& right,
  cudf::device_span<size_type const> left_indices,
  cudf::device_span<size_type const> right_indices,
  std::string const& predicate_code,
  cudf::join_kind join_kind,
  bool is_ptx                       = false,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Filters join indices using a JIT-compiled predicate from an AST expression.
 *
 * This overload converts an AST expression referencing columns from both left and right
 * tables into JIT-compiled CUDA code and uses it to filter the join index pairs.
 *
 * @param left The left table for predicate evaluation
 * @param right The right table for predicate evaluation
 * @param left_indices Device span of row indices in left table from join
 * @param right_indices Device span of row indices in right table from join
 * @param predicate An AST expression that returns a boolean for each pair of rows
 * @param join_kind The type of join operation (INNER_JOIN, LEFT_JOIN, or FULL_JOIN)
 * @param stream CUDA stream for operations
 * @param mr Device memory resource
 * @return A pair of device vectors [filtered_left_indices, filtered_right_indices]
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_join_indices_jit(
  cudf::table_view const& left,
  cudf::table_view const& right,
  cudf::device_span<size_type const> left_indices,
  cudf::device_span<size_type const> right_indices,
  cudf::ast::expression const& predicate,
  cudf::join_kind join_kind,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
