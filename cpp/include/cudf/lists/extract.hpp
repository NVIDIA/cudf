/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief APIs for extracting an element at a given position from each row of a list column
 */

namespace CUDF_EXPORT cudf {
namespace lists {
/**
 * @addtogroup lists_extract
 * @{
 */

/**
 * @brief Create a column where each row is the element at position `index` from the corresponding
 * sublist in the input `lists_column`.
 *
 * Output `column[i]` is set from element `lists_column[i][index]`.
 * If `index` is larger than the size of the sublist at `lists_column[i]`
 * then output `column[i] = null`.
 *
 * @code{.pseudo}
 * l = { {1, 2, 3}, {4}, {5, 6} }
 * r = extract_list_element(l, 1)
 * r is now {2, null, 6}
 * @endcode
 *
 * The `index` may also be negative in which case the row retrieved is offset
 * from the end of each sublist.
 *
 * @code{.pseudo}
 * l = { {"a"}, {"b", "c"}, {"d", "e", "f"} }
 * r = extract_list_element(l, -1)
 * r is now {"a", "c", "f"}
 * @endcode
 *
 * Any input where `lists_column[i] == null` will produce
 * output `column[i] = null`. Also, any element where
 * `lists_column[i][index] == null` will produce
 * output `column[i] = null`.
 *
 * @param lists_column Column to extract elements from.
 * @param index The row within each sublist to retrieve.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Column of extracted elements.
 */
std::unique_ptr<column> extract_list_element(
  lists_column_view const& lists_column,
  size_type index,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a column where each row is a single element from the corresponding sublist
 * in the input `lists_column`, selected using indices from the `indices` column.
 *
 * Output `column[i]` is set from element `lists_column[i][indices[i]]`.
 * If `indices[i]` is larger than the size of the sublist at `lists_column[i]`
 * then output `column[i] = null`.
 * Similarly, if `indices[i]` is `null`, then `column[i] = null`.
 *
 * @code{.pseudo}
 * l = { {1, 2, 3}, {4}, {5, 6} }
 * r = extract_list_element(l, {0, null, 2})
 * r is now {1, null, null}
 * @endcode
 *
 * `indices[i]` may also be negative, in which case the row retrieved is offset
 * from the end of each sublist.
 *
 * @code{.pseudo}
 * l = { {"a"}, {"b", "c"}, {"d", "e", "f"} }
 * r = extract_list_element(l, {-1, -2, -4})
 * r is now {"a", "b", null}
 * @endcode
 *
 * Any input where `lists_column[i] == null` produces output `column[i] = null`.
 * Any input where `lists_column[i][indices[i]] == null` produces output `column[i] = null`.
 *
 * @param lists_column Column to extract elements from.
 * @param indices The column whose rows indicate the element index to be retrieved from each list
 * row.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Column of extracted elements.
 * @throws cudf::logic_error If the sizes of `lists_column` and `indices` do not match.
 */
std::unique_ptr<column> extract_list_element(
  lists_column_view const& lists_column,
  column_view const& indices,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace lists
}  // namespace CUDF_EXPORT cudf
