/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/regex/flags.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Strings column APIs for extracting matched regex groups from each string.
 */

namespace CUDF_EXPORT cudf {
namespace strings {

struct regex_program;

/**
 * @addtogroup strings_extract
 * @{
 */

/**
 * @brief Returns a table of strings columns where each column corresponds to the matching
 * group specified in the given regex_program object
 *
 * All the strings for the first group will go in the first output column; the second group
 * go in the second column and so on. Null entries are added to the columns in row `i` if
 * the string at row `i` does not match.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a1", "b2", "c3"]
 * p = regex_program::create("([ab])(\\d)")
 * r = extract(s, p)
 * r is now [ ["a", "b", null],
 *            ["1", "2", null] ]
 * @endcode
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Columns of strings extracted from the input column
 */
std::unique_ptr<table> extract(
  strings_column_view const& input,
  regex_program const& prog,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a lists column of strings where each string column row corresponds to the
 * matching group specified in the given regex_program object
 *
 * All the matching groups for the first row will go in the first row output column; the second
 * row results will go into the second row output column and so on.
 *
 * A null output row will result if the corresponding input string row does not match or
 * that input row is null.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a1 b4", "b2", "c3 a5", "b", null]
 * p = regex_program::create("([ab])(\\d)")
 * r = extract_all_record(s, p)
 * r is now [ ["a", "1", "b", "4"],
 *            ["b", "2"],
 *            ["a", "5"],
 *            null,
 *            null ]
 * @endcode
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate any returned device memory
 * @return Lists column containing strings extracted from the input column
 */
std::unique_ptr<column> extract_all_record(
  strings_column_view const& input,
  regex_program const& prog,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a strings column where each column corresponds to the specified
 * group in the given regex_program object
 *
 * Any null string entries return corresponding null output for that row.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a1", "b2", "c3"]
 * p = regex_program::create("([ab])(\\d)")
 * r = extract(s, p, 1)
 * r is now [ "1", "2", "3" ]
 * @endcode
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param group Index of the group number to extract
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Columns of strings extracted from the input column
 */
std::unique_ptr<column> extract_single(
  strings_column_view const& input,
  regex_program const& prog,
  size_type group,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
