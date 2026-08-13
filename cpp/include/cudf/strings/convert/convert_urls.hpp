/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Strings column APIs for URL encoding and decoding.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 */

/**
 * @brief Encodes each string using URL encoding.
 *
 * Converts mostly non-ascii characters and control characters into UTF-8 hex code-points
 * prefixed with '%'. For example, the space character must be converted to characters '%20' where
 * the '20' indicates the hex value for space in UTF-8. Likewise, multi-byte characters are
 * converted to multiple hex characters. For example, the é character is converted to characters
 * '%C3%A9' where 'C3A9' is the UTF-8 bytes 0xC3A9 for this character.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> url_encode(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Decodes each string using URL encoding.
 *
 * Converts all character sequences starting with '%' into character code-points
 * interpreting the 2 following characters as hex values to create the code-point.
 * For example, the sequence '%20' is converted into byte (0x20) which is a single
 * space character. Another example converts '%C3%A9' into 2 sequential bytes
 * (0xc3 and 0xa9 respectively) which is the é character. Overall, 3 characters
 * are converted into one char byte whenever a '%%' (single percent) character
 * is encountered in the string.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> url_decode(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
