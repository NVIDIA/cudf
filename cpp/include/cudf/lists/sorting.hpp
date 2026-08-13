/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief API for segmented sorting of elements within each row of a list column
 */

namespace CUDF_EXPORT cudf {
namespace lists {
/**
 * @addtogroup lists_sort
 * @{
 */

/**
 * @brief Segmented sort of the elements within a list in each row of a list column.
 *
 * The `source_column` child may be of any type whose leaf elements are relationally comparable,
 * including arbitrarily nested LIST and STRUCT children.
 *
 * @code{.pseudo}
 * source_column            : [{4, 2, 3, 1}, {1, 2, NULL, 4}, {-10, 10, 0}]
 *
 * Ascending,  Null After   : [{1, 2, 3, 4}, {1, 2, 4, NULL}, {-10, 0, 10}]
 * Ascending,  Null Before  : [{1, 2, 3, 4}, {NULL, 1, 2, 4}, {-10, 0, 10}]
 * Descending, Null After   : [{4, 3, 2, 1}, {NULL, 4, 2, 1}, {10, 0, -10}]
 * Descending, Null Before  : [{4, 3, 2, 1}, {4, 2, 1, NULL}, {10, 0, -10}]
 * @endcode
 *
 * @param source_column View of the list column to sort
 * @param column_order The desired sort order
 * @param null_precedence The desired order of null compared to other elements in the list
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to allocate any returned objects
 * @return list column with elements in each list sorted.
 *
 */
std::unique_ptr<column> sort_lists(
  lists_column_view const& source_column,
  order column_order,
  null_order null_precedence,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Segmented sort of the elements within a list in each row of a list column using stable
 * sort.
 *
 * @copydoc cudf::lists::sort_lists
 */
std::unique_ptr<column> stable_sort_lists(
  lists_column_view const& source_column,
  order column_order,
  null_order null_precedence,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace lists
}  // namespace CUDF_EXPORT cudf
