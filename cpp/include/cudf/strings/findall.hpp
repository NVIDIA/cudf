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
 * @brief Strings column APIs for finding all regex matches within each string.
 */

namespace CUDF_EXPORT cudf {
namespace strings {

struct regex_program;

/**
 * @addtogroup strings_contains
 * @{
 */

/**
 * @brief Returns a lists column of strings for each matching occurrence using
 * the regex_program pattern within each string
 *
 * Each output row includes all the substrings within the corresponding input row
 * that match the given pattern. If no matches are found, the output row is empty.
 *
 * @code{.pseudo}
 * Example:
 * s = ["bunny", "rabbit", "hare", "dog"]
 * p = regex_program::create("[ab]")
 * r = findall(s, p)
 * r is now a lists column like:
 *  [ ["b"]
 *    ["a","b","b"]
 *    ["a"]
 *    [] ]
 * @endcode
 *
 * A null output row occurs if the corresponding input row is null.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column of strings
 */
std::unique_ptr<column> findall(
  strings_column_view const& input,
  regex_program const& prog,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the starting character index of the first match for the given pattern
 * in each row of the input column
 *
 * @code{.pseudo}
 * Example:
 * s = ["bunny", "rabbit", "hare", "dog"]
 * p = regex_program::create("[be]")
 * r = find_re(s, p)
 * r is now [0, 2, 3, -1]
 * @endcode
 *
 * A null output row occurs if the corresponding input row is null.
 * A -1 is returned for rows that do not contain a match.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of integers
 */
std::unique_ptr<column> find_re(
  strings_column_view const& input,
  regex_program const& prog,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
