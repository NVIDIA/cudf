/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/regex/flags.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file strings/contains.hpp
 * @brief Strings APIs for regex contains, count, matches, like
 */

namespace CUDF_EXPORT cudf {
namespace strings {

struct regex_program;

/**
 * @addtogroup strings_contains
 * @{
 */

/**
 * @brief Returns a boolean column identifying rows which
 * match the given regex_program object
 *
 * @code{.pseudo}
 * Example:
 * s = ["abc", "123", "def456"]
 * p = regex_program::create("\\d+")
 * r = contains_re(s, p)
 * r is now [false, true, true]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of boolean results for each string
 */
std::unique_ptr<column> contains_re(
  strings_column_view const& input,
  regex_program const& prog,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a boolean column identifying rows which
 * matching the given regex_program object but only at the beginning the string.
 *
 * @code{.pseudo}
 * Example:
 * s = ["abc", "123", "def456"]
 * p = regex_program::create("\\d+")
 * r = matches_re(s, p)
 * r is now [false, true, false]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of boolean results for each string
 */
std::unique_ptr<column> matches_re(
  strings_column_view const& input,
  regex_program const& prog,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the number of times the given regex_program's pattern
 * matches in each string
 *
 * @code{.pseudo}
 * Example:
 * s = ["abc", "123", "def45"]
 * p = regex_program::create("\\d")
 * r = count_re(s, p)
 * r is now [0, 3, 2]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of match counts for each string
 */
std::unique_ptr<column> count_re(
  strings_column_view const& input,
  regex_program const& prog,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a boolean column identifying rows which
 * match the given like pattern.
 *
 * The like pattern expects only 2 wildcard special characters:
 * - `%` zero or more of any character
 * - `_` any single character
 *
 * @code{.pseudo}
 * Example:
 * s = ["azaa", "ababaabba", "aaxa"]
 * r = like(s, "%a_aa%")
 * r is now [1, 1, 0]
 * r = like(s, "a__a")
 * r is now [1, 0, 1]
 * @endcode
 *
 * Specify an escape character to include either `%` or `_` in the search.
 * The `escape_character` is expected to be either 0 or 1 characters and
 * is expected to be an ASCII character.
 *
 * @code{.pseudo}
 * Example:
 * s = ["abc_def", "abc1def", "abc_"]
 * r = like(s, "abc/_d%", "/")
 * r is now [1, 0, 0]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * The `pattern` and `escape_character` parameter must be valid until a
 * synchronize is performed on the given `stream` parameter.
 *
 * @throw std::invalid_argument if `escape_character` contains more than on byte
 *
 * @param input Strings instance for this operation
 * @param pattern Like pattern to match within each string
 * @param escape_character Optional character specifies the escape prefix.
 *                         Default is no escape character.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New boolean column
 */
std::unique_ptr<column> like(
  strings_column_view const& input,
  std::string_view const& pattern,
  std::string_view const& escape_character = "",
  rmm::cuda_stream_view stream             = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr        = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a boolean column identifying rows which
 * match the corresponding like pattern in the given patterns
 *
 * The like pattern expects only 2 wildcard special characters:
 * - `%` zero or more of any character
 * - `_` any single character
 *
 * @code{.pseudo}
 * Example:
 * s = ["azaa", "ababaabba", "aaxa"]
 * p = ["%a", "b%", "__x_"]
 * r = like(s, p)
 * r is now [1, 0, 1]
 * @endcode
 *
 * Specify an escape character to include either `%` or `_` in the search.
 * The `escape_character` is expected to be either 0 or 1 characters and
 * is expected to be an ASCII character.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @throw std::invalid_argument if `patterns` contains nulls or `escape_character` is invalid
 * @throw std::invalid_argument if `patterns.size() != input.size()`
 *
 * @param input Strings instance for this operation
 * @param patterns Like patterns to match within each corresponding string
 * @param escape_character Optional character specifies the escape prefix.
 *                         Default is no escape character.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New boolean column
 */
std::unique_ptr<column> like(
  strings_column_view const& input,
  strings_column_view const& patterns,
  string_scalar const& escape_character = string_scalar(""),
  rmm::cuda_stream_view stream          = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr     = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
