/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <optional>

/**
 * @file
 * @brief Strings column APIs for casting short strings to/from an integer bit representation.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 */

/**
 * @brief Configures whether casting also byte swaps
 */
enum class endian : bool { BIG, LITTLE };

/**
 * @brief Returns a new integer numeric column encoded to represent the
 * strings in the input column
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * If the input column contains only strings less than or equal to 8 bytes,
 * the column can be converted to an appropriate integer type INT8, INT16, INT32, or INT64.
 * This is useful in libcudf functions that only require relational algebra operations
 * like groupby, join, or sort.
 *
 * @code{.pseudo}
 * Example:
 * s = ['a', 'b', '', 'c', 'd', 'ab']
 * b = cast_to_integer(s, INT16)
 * b is [24832, 25088, 0, 25344, 25600, 24930]
 *
 * in hex b is [0x6100, 0x6200, 0x0, 0x6300, 0x6400, 0x6162]
 * Note that with endian::LITTLE (default) the first byte of the string is
 * stored in the most significant byte of the integer and shorter strings
 * are padded with zero bytes on the right.
 * @endcode
 *
 * Multi-byte UTF-8 characters are encoded as multiple bytes in the integer result.
 *
 * If the input column contains strings greater than the number of bytes supported
 * by the output type, only the leading bytes that fit are encoded.
 *
 * If only equals logic is needed, `swap==BIG` is sufficient. Otherwise
 * use `swap==LITTLE` to ensure comparison operations work correctly.
 * With `swap==LITTLE`, the integers order the same as the strings' leading
 * bytes when the output type is unsigned or when no string starts with a byte
 * greater than 0x7F. Strings with embedded null bytes result in undefined behavior.
 *
 * @throw cudf::logic_error if output_type is not integral type.
 *
 * @param input Strings instance for this operation
 * @param output_type Type of integer numeric column to return
 * @param swap Whether to swap bytes on the output. Default is to swap.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with encoded integers converted from strings
 */
std::unique_ptr<column> cast_to_integer(
  strings_column_view const& input,
  data_type output_type,
  endian swap                       = endian::LITTLE,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column converting the encoded integer values from the
 * provided column into strings.
 *
 * This performs the inverse of `cast_to_integer`. The individual bytes of the integer
 * are reconverted into UTF-8 strings.
 *
 * @code{.pseudo}
 * Example:
 * b is INT16 [24832, 25088, 0, 25344, 25600, 24930]
 * s = cast_from_integer(b)
 * s is ['a', 'b', '', 'c', 'd', 'ab']
 *
 * in hex b is [0x6100, 0x6200, 0x0, 0x6300, 0x6400, 0x6162]
 * @endcode
 *
 * Trailing zero bytes (endian::LITTLE) or leading zero bytes (endian::BIG)
 * are not included in the output strings.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * Ensure the `swap` parameter matches the same value used in the `cast_to_integer` API.
 *
 * @throw cudf::logic_error if integers column is not integral type.
 *
 * @param integers Encoded integer column to convert
 * @param swap Whether the input is stored as big or little endian
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with integers as strings
 */
std::unique_ptr<column> cast_from_integer(
  column_view const& integers,
  endian swap                       = endian::LITTLE,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the minimum integer type required to encode the input column.
 *
 * Use this type with `cast_to_integer` to encode the input column.
 * An unsigned type is returned if any string starts with a byte greater than 0x7F
 * so that the encoded integers maintain the sort order of the strings.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The minimum integer type required to encode the input column, or std::nullopt
 * if the input column is not castable to an integer.
 */
std::optional<cudf::data_type> integer_cast_type(
  strings_column_view const& input, cuda::stream_ref stream = cudf::get_default_stream());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
