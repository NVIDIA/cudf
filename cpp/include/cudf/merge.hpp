/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <vector>

/**
 * @file
 * @brief APIs for merging a set of sorted tables into a single sorted table.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup column_merge
 * @{
 */

/**
 * @brief Merge a set of sorted tables.
 *
 * Merges sorted tables into one sorted table
 * containing data from all tables. The key columns
 * of each table must be sorted according to the
 * parameters (cudf::column_order and cudf::null_order)
 * specified for that column.
 *
 * ```
 * Example 1:
 * input:
 * table 1 => col 1 {0, 1, 2, 3}
 *            col 2 {4, 5, 6, 7}
 * table 2 => col 1 {1, 2}
 *            col 2 {8, 9}
 * table 3 => col 1 {2, 4}
 *            col 2 {8, 9}
 * output:
 * table => col 1 {0, 1, 1, 2, 2, 2, 3, 4}
 *          col 2 {4, 5, 8, 6, 8, 9, 7, 9}
 * ```
 * ```
 * Example 2:
 * input:
 * table 1 => col 0 {1, 0}
 *            col 1 {'c', 'b'}
 *            col 2 {RED, GREEN}
 *
 *
 * table 2 => col 0 {1}
 *            col 1 {'a'}
 *            col 2 {NULL}
 *
 *  with key_cols[] = {0,1}
 *  and  asc_desc[] = {ASC, ASC};
 *
 *  Lex-sorting is on columns {0,1}; hence, lex-sorting of ((L0 x L1) V (R0 x R1)) is:
 *  (0,'b', GREEN), (1,'a', NULL), (1,'c', RED)
 *
 *  (third column, the "color", just "goes along for the ride";
 *   meaning it is permuted according to the data movements dictated
 *   by lexicographic ordering of columns 0 and 1)
 *
 *   with result columns:
 *
 *   Res0 = {0,1,1}
 *   Res1 = {'b', 'a', 'c'}
 *   Res2 = {GREEN, NULL, RED}
 * ```
 *
 * @throws cudf::logic_error if tables in `tables_to_merge` have different
 * number of columns
 * @throws cudf::logic_error if tables in `tables_to_merge` have columns with
 * mismatched types
 * @throws cudf::logic_error if `key_cols` is empty
 * @throws cudf::logic_error if `key_cols` size is larger than the number of
 * columns in `tables_to_merge` tables
 * @throws cudf::logic_error if `key_cols` size and `column_order` size mismatches
 *
 * @param[in] tables_to_merge Non-empty list of tables to be merged
 * @param[in] key_cols Indices of left_cols and right_cols to be used
 *                     for comparison criteria
 * @param[in] column_order Sort order types of columns indexed by key_cols
 * @param[in] null_precedence Array indicating the order of nulls with respect
 * to non-nulls for the indexing columns (key_cols)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 *
 * @returns A table containing sorted data from all input tables
 */
std::unique_ptr<cudf::table> merge(
  std::vector<table_view> const& tables_to_merge,
  std::vector<cudf::size_type> const& key_cols,
  std::vector<cudf::order> const& column_order,
  std::vector<cudf::null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                         = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr                    = cudf::get_current_device_resource_ref());
/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
