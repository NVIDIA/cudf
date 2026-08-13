/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file strings/attributes.hpp
 * @brief Read attributes of strings column
 */

namespace CUDF_EXPORT cudf {

//! Strings column APIs
namespace strings {
/**
 * @addtogroup strings_apis
 * @{
 */

/**
 * @brief Returns a column containing character lengths
 * of each string in the given column
 *
 * The output column will have the same number of rows as the
 * specified strings column. Each row value will be the number of
 * characters in the corresponding string.
 *
 * Any null string will result in a null entry for that row in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with lengths for each string
 */
std::unique_ptr<column> count_characters(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column containing byte lengths
 * of each string in the given column
 *
 * The output column will have the same number of rows as the
 * specified strings column. Each row value will be the number of
 * bytes in the corresponding string.
 *
 * Any null string will result in a null entry for that row in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with the number of bytes for each string
 */
std::unique_ptr<column> count_bytes(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a numeric column with code point values (integers) for each
 * character of each string
 *
 * A code point is the integer value representation of a character.
 * For example, the code point value for the character 'A' in UTF-8 is 65.
 *
 * The size of the output column will be the total number of characters in the
 * strings column.
 *
 * Any null string is ignored. No null entries will appear in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New INT32 column with code point integer values for each character
 */
std::unique_ptr<column> code_points(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of strings_apis group

}  // namespace strings
}  // namespace CUDF_EXPORT cudf
