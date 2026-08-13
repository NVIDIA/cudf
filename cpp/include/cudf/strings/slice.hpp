/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <optional>

/**
 * @file
 * @brief Strings column APIs for extracting substrings by character position.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_slice
 * @{
 */

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column.
 *
 * @deprecated in 26.08. To be removed in a future version. Use the overload that take std::optional
 * parameters instead.
 *
 * The character positions to retrieve in each string are `[start,stop)`.
 * If the start position is outside a string's length, an empty
 * string is returned for that entry. If the stop position is past the
 * end of a string's length, the end of the string is used for
 * stop position for that string.
 *
 * Null string entries will return null output string entries.
 *
 * The scalar parameters are expected to be created using the same stream as the one passed to this
 * function. Otherwise, that stream must be synchronized before calling this function to ensure that
 * the scalar values are available on the device.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * r = slice_strings(s,2,6)
 * r is now ["llo","odby"]
 * r2 = slice_strings(s,2,5,2)
 * r2 is now ["lo","ob"]
 * @endcode
 *
 * @param input Strings column for this operation
 * @param start First character position to begin the substring
 * @param stop Last character position (exclusive) to end the substring
 * @param step Distance between input characters retrieved
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with sorted elements of this instance
 */
[[deprecated(
  "Use the overload that takes std::optional parameters instead")]] std::unique_ptr<column>
slice_strings(strings_column_view const& input,
              numeric_scalar<size_type> const& start = numeric_scalar<size_type>(0, false),
              numeric_scalar<size_type> const& stop  = numeric_scalar<size_type>(0, false),
              numeric_scalar<size_type> const& step  = numeric_scalar<size_type>(1),
              rmm::cuda_stream_view stream           = cudf::get_default_stream(),
              rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column.
 *
 * The character positions to retrieve in each string are `[start,stop)`.
 * If the start position is outside a string's length, an empty
 * string is returned for that entry. If the stop position is past the
 * end of a string's length, the end of the string is used for
 * stop position for that string.
 *
 * Null string entries will return null output string entries.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * r = slice_strings(s,2,6)
 * r is now ["llo","odby"]
 * r2 = slice_strings(s,2,5,2)
 * r2 is now ["lo","ob"]
 * @endcode
 *
 * @param input Strings column for this operation
 * @param start First character position to begin the substring
 * @param stop Last character position (exclusive) to end the substring
 * @param step Distance between input characters retrieved
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with sorted elements of this instance
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& input,
  std::optional<size_type> start,
  std::optional<size_type> stop     = std::nullopt,
  std::optional<size_type> step     = std::nullopt,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column using unique ranges for each string.
 *
 * The character positions to retrieve in each string are specified in
 * the `starts` and `stops` integer columns.
 * If a start position is outside a string's length, an empty
 * string is returned for that entry. If a stop position is past the
 * end of a string's length, the end of the string is used for
 * stop position for that string. Any stop position value set to -1 will
 * indicate to use the end of the string as the stop position for that
 * string.
 *
 * Null string entries will return null output string entries.
 *
 * The starts and stops column must both be the same integer type and
 * must be the same size as the strings column.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * starts = [ 1, 2 ]
 * stops = [ 5, 4 ]
 * r = slice_strings(s,starts,stops)
 * r is now ["ello","od"]
 * @endcode
 *
 * @throw cudf::logic_error if starts or stops is a different size than the strings column.
 * @throw cudf::logic_error if starts and stops are not same integer type.
 * @throw cudf::logic_error if starts or stops contains nulls.
 *
 * @param input Strings column for this operation
 * @param starts First character positions to begin the substring
 * @param stops Last character (exclusive) positions to end the substring
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with sorted elements of this instance
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& input,
  column_view const& starts,
  column_view const& stops,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
