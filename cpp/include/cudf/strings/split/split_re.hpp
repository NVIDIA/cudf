/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Strings column APIs for splitting strings using a regex pattern.
 */

namespace CUDF_EXPORT cudf {
namespace strings {

struct regex_program;

/**
 * @addtogroup strings_split
 * @{
 */

/**
 * @brief Splits strings elements into a table of strings columns
 * using a regex_program's pattern to delimit each string
 *
 * Each element generates a vector of strings that are stored in corresponding
 * rows in the output table -- `table[col,row] = token[col] of strings[row]`
 * where `token` is a substring between delimiters.
 *
 * The number of rows in the output table will be the same as the number of
 * elements in the input column. The resulting number of columns will be the
 * maximum number of tokens found in any input row.
 *
 * The `pattern` is used to identify the delimiters within a string
 * and splitting stops when either `maxsplit` or the end of the string is reached.
 *
 * An empty input string will produce a corresponding empty string in the
 * corresponding row of the first column.
 * A null row will produce corresponding null rows in the output table.
 *
 * The regex_program's regex_flags are ignored.
 *
 * @code{.pseudo}
 * s = ["a_bc def_g", "a__bc", "_ab cd", "ab_cd "]
 * p1 = regex_program::create("[_ ]")
 * s1 = split_re(s, p1)
 * s1 is a table of strings columns:
 *     [ ["a", "a", "", "ab"],
 *       ["bc", "", "ab", "cd"],
 *       ["def", "bc", "cd", ""],
 *       ["g", null, null, null] ]
 * p2 = regex_program::create("[ _]")
 * s2 = split_re(s, p2, 1)
 * s2 is a table of strings columns:
 *     [ ["a", "a", "", "ab"],
 *       ["bc def_g", "_bc", "ab cd", "cd "] ]
 * @endcode
 *
 * @throw cudf::logic_error if `pattern` is empty.
 *
 * @param input A column of string elements to be split
 * @param prog Regex program instance
 * @param maxsplit Maximum number of splits to perform.
 *        Default of -1 indicates all possible splits on each string.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned result's device memory
 * @return A table of columns of strings
 */
std::unique_ptr<table> split_re(
  strings_column_view const& input,
  regex_program const& prog,
  size_type maxsplit                = -1,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Splits strings elements into a table of strings columns using a
 * regex_program's pattern to delimit each string starting from the end of the string
 *
 * Each element generates a vector of strings that are stored in corresponding
 * rows in the output table -- `table[col,row] = token[col] of string[row]`
 * where `token` is the substring between each delimiter.
 *
 * The number of rows in the output table will be the same as the number of
 * elements in the input column. The resulting number of columns will be the
 * maximum number of tokens found in any input row.
 *
 * Splitting occurs by traversing starting from the end of the input string.
 * The `pattern` is used to identify the delimiters within a string
 * and splitting stops when either `maxsplit` or the beginning of the string
 * is reached.
 *
 * An empty input string will produce a corresponding empty string in the
 * corresponding row of the first column.
 * A null row will produce corresponding null rows in the output table.
 *
 * The regex_program's regex_flags are ignored.
 *
 * @code{.pseudo}
 * s = ["a_bc def_g", "a__bc", "_ab cd", "ab_cd "]
 * p1 = regex_program::create("[_ ]")
 * s1 = rsplit_re(s, p1)
 * s1 is a table of strings columns:
 *     [ ["a", "a", "", "ab"],
 *       ["bc", "", "ab", "cd"],
 *       ["def", "bc", "cd", ""],
 *       ["g", null, null, null] ]
 * p2 = regex_program::create("[ _]")
 * s2 = rsplit_re(s, p2, 1)
 * s2 is a table of strings columns:
 *     [ ["a_bc def", "a_", "_ab", "ab"],
 *       ["g", "bc", "cd", "cd "] ]
 * @endcode
 *
 * @throw cudf::logic_error if `pattern` is empty.
 *
 * @param input A column of string elements to be split
 * @param prog Regex program instance
 * @param maxsplit Maximum number of splits to perform.
 *        Default of -1 indicates all possible splits on each string.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned result's device memory
 * @return A table of columns of strings
 */
std::unique_ptr<table> rsplit_re(
  strings_column_view const& input,
  regex_program const& prog,
  size_type maxsplit                = -1,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Splits strings elements into a list column of strings
 * using the given regex_program to delimit each string
 *
 * Each element generates an array of strings that are stored in an output
 * lists column -- `list[row] = [token1, token2, ...] found in input[row]`
 * where `token` is a substring between delimiters.
 *
 * The number of elements in the output column will be the same as the number of
 * elements in the input column. Each individual list item will contain the
 * new strings for that row. The resulting number of strings in each row can vary
 * from 0 to `maxsplit + 1`.
 *
 * The `pattern` is used to identify the delimiters within a string
 * and splitting stops when either `maxsplit` or the end of the string is reached.
 *
 * An empty input string will produce a corresponding empty list item output row.
 * A null row will produce a corresponding null output row.
 *
 * The regex_program's regex_flags are ignored.
 *
 * @code{.pseudo}
 * s = ["a_bc def_g", "a__bc", "_ab cd", "ab_cd "]
 * p1 = regex_program::create("[_ ]")
 * s1 = split_record_re(s, p1)
 * s1 is a lists column of strings:
 *     [ ["a", "bc", "def", "g"],
 *       ["a", "", "bc"],
 *       ["", "ab", "cd"],
 *       ["ab", "cd", ""] ]
 * p2 = regex_program::create("[ _]")
 * s2 = split_record_re(s, p2, 1)
 * s2 is a lists column of strings:
 *     [ ["a", "bc def_g"],
 *       ["a", "_bc"],
 *       ["", "ab cd"],
 *       ["ab", "cd "] ]
 * @endcode
 *
 * @throw cudf::logic_error if `pattern` is empty.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input A column of string elements to be split
 * @param prog Regex program instance
 * @param maxsplit Maximum number of splits to perform.
 *        Default of -1 indicates all possible splits on each string.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned result's device memory
 * @return Lists column of strings
 */
std::unique_ptr<column> split_record_re(
  strings_column_view const& input,
  regex_program const& prog,
  size_type maxsplit                = -1,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Splits strings elements into a list column of strings using the given
 * regex_program to delimit each string starting from the end of the string
 *
 * Each element generates a vector of strings that are stored in an output
 * lists column -- `list[row] = [token1, token2, ...] found in input[row]`
 * where `token` is a substring between delimiters.
 *
 * The number of elements in the output column will be the same as the number of
 * elements in the input column. Each individual list item will contain the
 * new strings for that row. The resulting number of strings in each row can vary
 * from 0 to `maxsplit + 1`.
 *
 * Splitting occurs by traversing starting from the end of the input string.
 * The `pattern` is used to identify the separation points within a string
 * and splitting stops when either `maxsplit` or the beginning of the string
 * is reached.
 *
 * An empty input string will produce a corresponding empty list item output row.
 * A null row will produce a corresponding null output row.
 *
 * The regex_program's regex_flags are ignored.
 *
 * @code{.pseudo}
 * s = ["a_bc def_g", "a__bc", "_ab cd", "ab_cd "]
 * p1 = regex_program::create("[_ ]")
 * s1 = rsplit_record_re(s, p1)
 * s1 is a lists column of strings:
 *     [ ["a", "bc", "def", "g"],
 *       ["a", "", "bc"],
 *       ["", "ab", "cd"],
 *       ["ab", "cd", ""] ]
 * p2 = regex_program::create("[ _]")
 * s2 = rsplit_record_re(s, p2, 1)
 * s2 is a lists column of strings:
 *     [ ["a_bc def", "g"],
 *       ["a_", "bc"],
 *       ["_ab", "cd"],
 *       ["ab_cd", ""] ]
 * @endcode
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @throw cudf::logic_error if `pattern` is empty.
 *
 * @param input A column of string elements to be split
 * @param prog Regex program instance
 * @param maxsplit Maximum number of splits to perform.
 *        Default of -1 indicates all possible splits on each string.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned result's device memory
 * @return Lists column of strings
 */
std::unique_ptr<column> rsplit_record_re(
  strings_column_view const& input,
  regex_program const& prog,
  size_type maxsplit                = -1,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
