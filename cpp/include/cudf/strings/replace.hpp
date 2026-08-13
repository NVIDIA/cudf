/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Strings column APIs for replacing target strings or slices with replacement strings.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_replace
 * @{
 */

/**
 * @brief Replaces target string within each string with the specified
 * replacement string.
 *
 * This function searches each string in the column for the target string.
 * If found, the target string is replaced by the repl string within the
 * input string. If not found, the output entry is just a copy of the
 * corresponding input string.
 *
 * Specifying an empty string for repl will essentially remove the target
 * string if found in each string.
 *
 * Null string entries will return null output string entries.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * r1 = replace(s,"o","OOO")
 * r1 is now ["hellOOO","gOOOOOOdbye"]
 * r2 = replace(s,"oo","")
 * r2 is now ["hello","gdbye"]
 * @endcode
 *
 * @throw cudf::logic_error if target is an empty string.
 *
 * @param input Strings column for this operation
 * @param target String to search for within each string
 * @param repl Replacement string if target is found
 * @param maxrepl Maximum times to replace if target appears multiple times in the input string.
 *        Default of -1 specifies replace all occurrences of target in each string.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> replace(
  strings_column_view const& input,
  string_scalar const& target,
  string_scalar const& repl,
  cudf::size_type maxrepl           = -1,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief This function replaces each string in the column with the provided
 * repl string within the [start,stop) character position range.
 *
 * Null string entries will return null output string entries.
 *
 * Position values are 0-based meaning position 0 is the first character
 * of each string.
 *
 * This function can be used to insert a string into specific position
 * by specifying the same position value for start and stop. The repl
 * string can be appended to each string by specifying -1 for both
 * start and stop.
 *
 * @code{.pseudo}
 * Example:
 * s = ["abcdefghij","0123456789"]
 * r = s.replace_slice(s,2,5,"z")
 * r is now ["abzfghij", "01z56789"]
 * @endcode
 *
 * @throw cudf::logic_error if start is greater than stop.
 *
 * @param input Strings column for this operation.
 * @param repl Replacement string for specified positions found.
 *        Default is empty string.
 * @param start Start position where repl will be added.
 *        Default is 0, first character position.
 * @param stop End position (exclusive) to use for replacement.
 *        Default of -1 specifies the end of each string.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> replace_slice(
  strings_column_view const& input,
  string_scalar const& repl         = string_scalar(""),
  size_type start                   = 0,
  size_type stop                    = -1,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Replaces, per row, all occurrences of `targets[i]` within `input[i]` with `repls[i]`.
 *
 * For each row `i`, this function searches `input[i]` for the substring `targets[i]`
 * and replaces every occurrence with `repls[i]`. If `targets[i]` is an empty string,
 * `input[i]` is copied unchanged. If `targets[i]` is not found,
 * the output entry is a copy of `input[i]`.
 *
 * Output row `i` is null if any of `input[i]`, `targets[i]`, or `repls[i]` is null.
 *
 * @code{.pseudo}
 * Example:
 * input   = ["hello world", "foo bar", "aaa"]
 * targets = ["o", "bar", "a"]
 * repls   = ["0", "BAR", "X"]
 * result  = replace(input, targets, repls)
 * result is now ["hell0 w0rld", "foo BAR", "XXX"]
 * @endcode
 *
 * @throw std::invalid_argument if `targets.size() != input.size()`
 * @throw std::invalid_argument if `repls.size() != input.size()`
 *
 * @param input Strings column for this operation
 * @param targets Per-row strings to search for within each input string
 * @param repls Per-row replacement strings used when the corresponding target is found
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> replace(
  strings_column_view const& input,
  strings_column_view const& targets,
  strings_column_view const& repls,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Replaces substrings matching a list of targets with the corresponding
 * replacement strings.
 *
 * For each string in strings, the list of targets is searched within that string.
 * If a target string is found, it is replaced by the corresponding entry in the repls column.
 * All occurrences found in each string are replaced.
 *
 * This does not use regex to match targets in the string. Empty string targets are ignored.
 *
 * Null string entries will return null output string entries.
 *
 * The repls argument can optionally contain a single string. In this case, all
 * matching target substrings will be replaced by that single string.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * tgts = ["e","o"]
 * repls = ["EE","OO"]
 * r1 = replace(s,tgts,repls)
 * r1 is now ["hEEllO", "gOOOOdbyEE"]
 * tgts = ["e","oo"]
 * repls = ["33",""]
 * r2 = replace(s,tgts,repls)
 * r2 is now ["h33llo", "gdby33"]
 * @endcode
 *
 * @throw cudf::logic_error if targets and repls are different sizes except
 * if repls is a single string.
 * @throw cudf::logic_error if targets or repls contain null entries.
 *
 * @param input Strings column for this operation
 * @param targets Strings to search for in each string
 * @param repls Corresponding replacement strings for target strings
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> replace_multiple(
  strings_column_view const& input,
  strings_column_view const& targets,
  strings_column_view const& repls,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
