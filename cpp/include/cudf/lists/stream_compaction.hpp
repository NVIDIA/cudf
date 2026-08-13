/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief APIs for filtering elements from each row of a list column using a boolean mask
 */

namespace CUDF_EXPORT cudf {
namespace lists {

/**
 * @addtogroup lists_filtering
 * @{
 */

/**
 * @brief Filters elements in each row of `input` LIST column using `boolean_mask`
 * LIST of booleans as a mask.
 *
 * Given an input `LIST` column and a list-of-bools column, the function produces
 * a new `LIST` column of the same type as `input`, where each element is copied
 * from the input row *only* if the corresponding `boolean_mask` is non-null and `true`.
 *
 * E.g.
 * @code{.pseudo}
 * input        = { {0,1,2}, {3,4}, {5,6,7}, {8,9} };
 * boolean_mask = { {0,1,1}, {1,0}, {1,1,1}, {0,0} };
 * results      = { {1,2},   {3},   {5,6,7}, {} };
 * @endcode
 *
 * `input` and `boolean_mask` must have the same number of rows.
 * The output column has the same number of rows as the input column.
 * An element is copied to an output row *only* if the corresponding boolean_mask element is `true`.
 * An output row is invalid only if the input row is invalid.
 *
 * @throws cudf::logic_error if `boolean_mask` is not a "lists of bools" column
 * @throws cudf::logic_error if `input` and `boolean_mask` have different number of rows
 *
 * @param input The input list column view to be filtered
 * @param boolean_mask A nullable list of bools column used to filter `input` elements
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return List column of the same type as `input`, containing filtered list rows
 */
std::unique_ptr<column> apply_boolean_mask(
  lists_column_view const& input,
  lists_column_view const& boolean_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Filters elements in each row of `input` LIST column using `deletion_mask`
 * LIST of booleans as a mask.
 *
 * Given an input `LIST` column and a list-of-bools column, the function produces
 * a new `LIST` column of the same type as `input`, where each element is deleted
 * from the input row if the corresponding `deletion_mask` is non-null and `true`.
 *
 * E.g.
 * @code{.pseudo}
 * input          = { {0,1,2}, {3,4}, {5,6,7}, {8,9} };
 * deletion_mask  = { {0,1,1}, {1,0}, {1,1,1}, {0,0} };
 * results        = { {0},     {4},   {},      {8,9} };
 * @endcode
 *
 * `input` and `deletion_mask` must have the same number of rows.
 * The output column has the same number of rows as the input column.
 * An element is copied to an output row if the corresponding deletion_mask element is
 * non-null and `false`.
 * An output row is invalid only if the input row is invalid.
 *
 * @throws cudf::logic_error if `deletion_mask` is not a "lists of bools" column
 * @throws cudf::logic_error if `input` and `deletion_mask` have different number of rows
 *
 * @param input The input list column view to be filtered
 * @param deletion_mask A nullable list of bools column used to filter `input` elements
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return List column of the same type as `input`, containing list rows with deleted elements
 * removed
 */
std::unique_ptr<column> apply_deletion_mask(
  lists_column_view const& input,
  lists_column_view const& deletion_mask,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a new list column without duplicate elements in each list.
 *
 * Given a lists column `input`, distinct elements of each list are copied to the corresponding
 * output list. The order of lists is preserved while the order of elements within each list is not
 * guaranteed.
 *
 * Example:
 * @code{.pseudo}
 * input  = { {0, 1, 2, 3, 2}, {3, 1, 2}, null, {4, null, null, 5} }
 * result = { {0, 1, 2, 3}, {3, 1, 2}, null, {4, null, 5} }
 * @endcode
 *
 * @param input The input lists column
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether floating-point NaNs should be considered as equal
 * @param keep_option Flag to specify which element to keep (first, last, any)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return The resulting lists column containing lists without duplicates
 */
std::unique_ptr<column> distinct(
  lists_column_view const& input,
  null_equality nulls_equal         = null_equality::EQUAL,
  nan_equality nans_equal           = nan_equality::ALL_EQUAL,
  duplicate_keep_option keep_option = duplicate_keep_option::KEEP_ANY,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace lists
}  // namespace CUDF_EXPORT cudf
