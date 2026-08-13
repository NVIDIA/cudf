/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief APIs for checking whether a value is contained in each row of a list column
 */

namespace CUDF_EXPORT cudf {
namespace lists {
/**
 * @addtogroup lists_contains
 * @{
 */

/**
 * @brief Create a column of `bool` values indicating whether the specified scalar
 * is an element of each row of a list column.
 *
 * The output column has as many elements as the input `lists` column.
 * Output `column[i]` is set to true if the lists row `lists[i]` contains the value
 * specified in `search_key`. Otherwise, it is set to false.
 *
 * Output `column[i]` is set to null if one or more of the following are true:
 *   1. The search key `search_key` is null
 *   2. The list row `lists[i]` is null
 *
 * @param lists Lists column whose `n` rows are to be searched
 * @param search_key The scalar key to be looked up in each list row
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return BOOL8 column of `n` rows with the result of the lookup
 */
std::unique_ptr<column> contains(
  cudf::lists_column_view const& lists,
  cudf::scalar const& search_key,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a column of `bool` values indicating whether the list rows of the first
 * column contain the corresponding values in the second column
 *
 * The output column has as many elements as the input `lists` column.
 * Output `column[i]` is set to true if the lists row `lists[i]` contains the value
 * in `search_keys[i]`. Otherwise, it is set to false.
 *
 * Output `column[i]` is set to null if one or more of the following are true:
 *   1. The row `search_keys[i]` is null
 *   2. The list row `lists[i]` is null
 *
 * @param lists Lists column whose `n` rows are to be searched
 * @param search_keys Column of elements to be looked up in each list row.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return BOOL8 column of `n` rows with the result of the lookup
 */
std::unique_ptr<column> contains(
  cudf::lists_column_view const& lists,
  cudf::column_view const& search_keys,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a column of `bool` values indicating whether each row in the `lists` column
 * contains at least one null element.
 *
 * The output column has as many elements as the input `lists` column.
 * Output `column[i]` is set to null if the row `lists[i]` is null.
 * Otherwise, `column[i]` is set to a non-null boolean value, depending on whether that list
 * contains a null element.
 *
 * A row with an empty list will always return false.
 * Nulls inside non-null nested elements (such as lists or structs) are not considered.
 *
 * @param lists Lists column whose `n` rows are to be searched.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return BOOL8 column of `n` rows with the result of the lookup
 */
std::unique_ptr<column> contains_nulls(
  cudf::lists_column_view const& lists,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Option to choose whether `index_of()` returns the first or last match
 * of a search key in a list row
 */
enum class duplicate_find_option : int32_t {
  FIND_FIRST = 0,  ///< Finds first instance of a search key in a list row.
  FIND_LAST        ///< Finds last instance of a search key in a list row.
};

/**
 * @brief Create a column of values indicating the position of a search key
 * within each list row in the `lists` column
 *
 * The output column has as many elements as there are rows in the input `lists` column.
 * Output `column[i]` contains a 0-based index indicating the position of the search key
 * in each list, counting from the beginning of the list.
 * Note:
 *   1. If the `search_key` is null, all output rows are set to null.
 *   2. If the row `lists[i]` is null, `output[i]` is also null.
 *   3. If the row `lists[i]` does not contain the `search_key`, `output[i]` is set to `-1`.
 *   4. In all other cases, `output[i]` is set to a non-negative `size_type` index.
 *
 * If the `find_option` is set to `FIND_FIRST`, the position of the first match for
 * `search_key` is returned.
 * If `find_option == FIND_LAST`, the position of the last match in the list row is
 * returned.
 *
 * @throw cudf::data_type_error If `search_keys` type does not match the element type in `lists`
 *
 * @param lists Lists column whose `n` rows are to be searched
 * @param search_key The scalar key to be looked up in each list row
 * @param find_option Whether to return the position of the first match (`FIND_FIRST`) or
 * last (`FIND_LAST`)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return column of `n` rows with the location of the `search_key`
 */
std::unique_ptr<column> index_of(
  cudf::lists_column_view const& lists,
  cudf::scalar const& search_key,
  duplicate_find_option find_option = duplicate_find_option::FIND_FIRST,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a column of values indicating the position of a search key
 * row within the corresponding list row in the `lists` column
 *
 * The output column has as many elements as there are rows in the input `lists` column.
 * Output `column[i]` contains a 0-based index indicating the position of each search key
 * row in its corresponding list row, counting from the beginning of the list.
 * Note:
 *   1. If `search_keys[i]` is null, `output[i]` is also null.
 *   2. If the row `lists[i]` is null, `output[i]` is also null.
 *   3. If the row `lists[i]` does not contain `search_key[i]`, `output[i]` is set to `-1`.
 *   4. In all other cases, `output[i]` is set to a non-negative `size_type` index.
 *
 * If the `find_option` is set to `FIND_FIRST`, the position of the first match for
 * `search_key` is returned.
 * If `find_option == FIND_LAST`, the position of the last match in the list row is
 * returned.
 *
 * @throw cudf::logic_error If `search_keys` does not match `lists` in its number of rows
 * @throw cudf::data_type_error If `search_keys` type does not match the element type in `lists`
 *
 * @param lists Lists column whose `n` rows are to be searched
 * @param search_keys A column of search keys to be looked up in each corresponding row of
 * `lists`
 * @param find_option Whether to return the position of the first match (`FIND_FIRST`) or
 * last (`FIND_LAST`)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return column of `n` rows with the location of the `search_key`
 */
std::unique_ptr<column> index_of(
  cudf::lists_column_view const& lists,
  cudf::column_view const& search_keys,
  duplicate_find_option find_option = duplicate_find_option::FIND_FIRST,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace lists
}  // namespace CUDF_EXPORT cudf
