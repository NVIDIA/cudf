/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <vector>

/**
 * @file
 * @brief Column APIs for lower_bound, upper_bound, and contains
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup column_search
 * @{
 */

/**
 * @brief Find smallest indices in a sorted table where values should be inserted to maintain order.
 *
 * For each row in `needles`, find the first index in `haystack` where inserting the row still
 * maintains its sort order.
 *
 * @code{.pseudo}
 * Example:
 *
 *  Single column:
 *      idx        0   1   2   3   4
 *   haystack = { 10, 20, 20, 30, 50 }
 *   needles  = { 20 }
 *   result   = {  1 }
 *
 *  Multi Column:
 *      idx          0    1    2    3    4
 *   haystack = {{  10,  20,  20,  20,  20 },
 *               { 5.0,  .5,  .5,  .7,  .7 },
 *               {  90,  77,  78,  61,  61 }}
 *   needles  = {{ 20 },
 *               { .7 },
 *               { 61 }}
 *   result   = {   3 }
 * @endcode
 *
 * @param haystack The table containing search space
 * @param needles Values for which to find the insert locations in the search space
 * @param column_order Vector of column sort order
 * @param null_precedence Vector of null_precedence enums needles
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A non-nullable column of elements containing the insertion points
 */
std::unique_ptr<column> lower_bound(
  table_view const& haystack,
  table_view const& needles,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Find largest indices in a sorted table where values should be inserted to maintain order.
 *
 * For each row in `needles`, find the last index in `haystack` where inserting the row still
 * maintains its sort order.
 *
 * @code{.pseudo}
 * Example:
 *
 *  Single Column:
 *      idx        0   1   2   3   4
 *   haystack = { 10, 20, 20, 30, 50 }
 *   needles  = { 20 }
 *   result   = {  3 }
 *
 *  Multi Column:
 *      idx          0    1    2    3    4
 *   haystack = {{  10,  20,  20,  20,  20 },
 *               { 5.0,  .5,  .5,  .7,  .7 },
 *               {  90,  77,  78,  61,  61 }}
 *   needles  = {{ 20 },
 *               { .7 },
 *               { 61 }}
 *   result =     { 5 }
 * @endcode
 *
 * @param haystack The table containing search space
 * @param needles Values for which to find the insert locations in the search space
 * @param column_order Vector of column sort order
 * @param null_precedence Vector of null_precedence enums needles
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A non-nullable column of elements containing the insertion points
 */
std::unique_ptr<column> upper_bound(
  table_view const& haystack,
  table_view const& needles,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Check if the given `needle` value exists in the `haystack` column.
 *
 * @throws cudf::logic_error If `haystack.type() != needle.type()`.
 *
 * @code{.pseudo}
 *  Single Column:
 *   idx           0   1   2   3   4
 *   haystack = { 10, 20, 20, 30, 50 }
 *   needle   = { 20 }
 *   result   = true
 * @endcode
 *
 * @param haystack The column containing search space
 * @param needle A scalar value to check for existence in the search space
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return true if the given `needle` value exists in the `haystack` column
 */
bool contains(column_view const& haystack,
              scalar const& needle,
              rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Check if the given `needles` values exists in the `haystack` column.
 *
 * The new column will have type BOOL and have the same size and null mask as the input `needles`
 * column. That is, any null row in the `needles` column will result in a null row in the output
 * column.
 *
 * @throws cudf::logic_error If `haystack.type() != needles.type()`
 *
 * @code{.pseudo}
 *   haystack = { 10, 20, 30, 40, 50 }
 *   needles  = { 20, 40, 60, 80 }
 *   result   = { true, true, false, false }
 * @endcode
 *
 * @param haystack The column containing search space
 * @param needles A column of values to check for existence in the search space
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A BOOL column indicating if each element in `needles` exists in the search space
 */
std::unique_ptr<column> contains(
  column_view const& haystack,
  column_view const& needles,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
