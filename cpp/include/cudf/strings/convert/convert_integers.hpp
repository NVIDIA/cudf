/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Strings column APIs for converting to/from integer numeric types.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 */

/**
 * @brief Returns a new integer numeric column parsing integer values from the
 * provided strings column.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * Only characters [0-9] plus a prefix '-' and '+' are recognized.
 * When any other character is encountered, the parsing ends for that string
 * and the current digits are converted into an integer.
 *
 * Overflow of the resulting integer type is not checked.
 * Each string is converted using an int64 type and then cast to the
 * target integer type before storing it into the output column.
 * If the resulting integer type is too small to hold the value,
 * the stored value will be undefined.
 *
 * @throw cudf::logic_error if output_type is not integral type.
 *
 * @param input Strings instance for this operation
 * @param output_type Type of integer numeric column to return
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with integers converted from strings
 */
std::unique_ptr<column> to_integers(
  strings_column_view const& input,
  data_type output_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column converting the integer values from the
 * provided column into strings.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * For each integer, a string is created in base-10 decimal.
 * Negative numbers will include a '-' prefix.
 *
 * @throw cudf::logic_error if integers column is not integral type.
 *
 * @param integers Numeric column to convert
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with integers as strings
 */
std::unique_ptr<column> from_integers(
  column_view const& integers,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to integers.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * have all characters in [-+0-9]. The optional sign character must only be in the first
 * position. Notice that the integer value is not checked to be within its storage limits.
 * For strict integer type check, use the other `is_integer()` API which accepts `data_type`
 * argument.
 *
 * @code{.pseudo}
 * Example:
 * s = ['123', '-456', '', 'A', '+7']
 * b = s.is_integer(s)
 * b is [true, true, false, false, true]
 * @endcode
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of boolean results for each string
 */
std::unique_ptr<column> is_integer(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to integers.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * has all characters in [-+0-9]. The optional sign character must only be in the first
 * position. Also, the integer component must fit within the size limits of the underlying
 * storage type, which is provided by the int_type parameter.
 *
 * @code{.pseudo}
 * Example:
 * s = ['123456', '-456', '', 'A', '+7']
 *
 * output1 = s.is_integer(s, data_type{type_id::INT32})
 * output1 is [true, true, false, false, true]
 *
 * output2 = s.is_integer(s, data_type{type_id::INT8})
 * output2 is [false, false, false, false, true]
 * @endcode
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param input Strings instance for this operation
 * @param int_type Integer type used for checking underflow and overflow
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of boolean results for each string
 */
std::unique_ptr<column> is_integer(
  strings_column_view const& input,
  data_type int_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new integer numeric column parsing hexadecimal values from the
 * provided strings column.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * Only characters [0-9] and [A-F] are recognized.
 * When any other character is encountered, the parsing ends for that string.
 * No interpretation is made on the sign of the integer.
 *
 * Overflow of the resulting integer type is not checked.
 * Each string is converted using an int64 type and then cast to the
 * target integer type before storing it into the output column.
 * If the resulting integer type is too small to hold the value,
 * the stored value will be undefined.
 *
 * @throw cudf::logic_error if output_type is not integral type.
 *
 * @param input Strings instance for this operation
 * @param output_type Type of integer numeric column to return
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with integers converted from strings
 */
std::unique_ptr<column> hex_to_integers(
  strings_column_view const& input,
  data_type output_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to integers from hex.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * has at least one character in [0-9A-Za-z]. Also, the string may start
 * with '0x'.
 *
 * @code{.pseudo}
 * Example:
 * s = ['123', '-456', '', 'AGE', '+17EA', '0x9EF' '123ABC']
 * b = is_hex(s)
 * b is [true, false, false, false, false, true, true]
 * @endcode
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of boolean results for each string
 */
std::unique_ptr<column> is_hex(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column converting integer columns to hexadecimal
 * characters.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * The output character set is '0'-'9' and 'A'-'F'. The output string width will
 * be a multiple of 2 depending on the size of the integer type. A single leading
 * zero is applied to the first non-zero output byte if it less than 0x10.
 *
 * @code{.pseudo}
 * Example:
 * input = [1234, -1, 0, 27, 342718233] // int32 type input column
 * s = integers_to_hex(input)
 * s is [ '04D2', 'FFFFFFFF', '00', '1B', '146D7719']
 * @endcode
 *
 * The example above shows an `INT32` type column where each integer is 4 bytes.
 * Leading zeros are suppressed unless filling out a complete byte as in
 * `1234 -> '04D2'` instead of `000004D2` or `4D2`.
 *
 * @throw cudf::logic_error if the input column is not integral type.
 *
 * @param input Integer column to convert to hex
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with hexadecimal characters
 */
std::unique_ptr<column> integers_to_hex(
  column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
