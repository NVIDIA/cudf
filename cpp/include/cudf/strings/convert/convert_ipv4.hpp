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
 * @brief Strings column APIs for converting IPv4 addresses to/from integers.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 */

/**
 * @brief Converts IPv4 addresses into integers.
 *
 * The IPv4 format is 1-3 character digits [0-9] between 3 dots
 * (e.g. 123.45.67.890). Each section can have a value between [0-255].
 *
 * The four sets of digits are converted to integers and placed in 8-bit fields inside
 * the resulting integer.
 * ```
 *   i0.i1.i2.i3 -> (i0 << 24) | (i1 << 16) | (i2 << 8) | (i3)
 * ```
 *
 * No checking is done on the format. If a string is not in IPv4 format, the resulting
 * integer is undefined.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @param input Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New UINT32 column converted from strings
 */
std::unique_ptr<column> ipv4_to_integers(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Converts integers into IPv4 addresses as strings.
 *
 * The IPv4 format is 1-3 character digits [0-9] between 3 dots
 * (e.g. 123.45.67.890). Each section can have a value between [0-255].
 *
 * Each input integer is dissected into four integers by dividing the input into 8-bit sections.
 * These sub-integers are then converted into [0-9] characters and placed between '.' characters.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @throw cudf::logic_error if the input column is not UINT32 type.
 *
 * @param integers Integer (UINT32) column to convert
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> integers_to_ipv4(
  column_view const& integers,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to integers from IPv4 format.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * has the following format `xxx.xxx.xxx.xxx` where `xxx` is integer digits
 * between 0-255.
 *
 * @code{.pseudo}
 * Example:
 * s = ['123.255.0.7', '127.0.0.1', '', '1.2.34' '123.456.789.10']
 * b = s.is_ipv4(s)
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
std::unique_ptr<column> is_ipv4(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
