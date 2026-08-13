/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <vector>

/**
 * @file
 * @brief Strings column APIs for translating or filtering individual characters.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_modify
 * @{
 */

/**
 * @brief Translates individual characters within each string.
 *
 * This can also be used to remove a character by specifying 0 for the corresponding table entry.
 *
 * Null string entries result in null entries in the output column.
 *
 * @code{.pseudo}
 * Example:
 * s = ["aa","bbb","cccc","abcd"]
 * t = [['a','A'],['b',''],['d':'Q']]
 * r = translate(s,t)
 * r is now ["AA", "", "cccc", "AcQ"]
 * @endcode
 *
 * @param input Strings instance for this operation
 * @param chars_table Table of UTF-8 character mappings
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with padded strings
 */
std::unique_ptr<column> translate(
  strings_column_view const& input,
  std::vector<std::pair<char_utf8, char_utf8>> const& chars_table,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Removes or keeps the specified character ranges in cudf::strings::filter_characters
 */
enum class filter_type : bool {
  KEEP,   ///< All characters but those specified are removed
  REMOVE  ///< Only the specified characters are removed
};

/**
 * @brief Removes ranges of characters from each string in a strings column.
 *
 * This can also be used to keep only the specified character ranges
 * and remove all others from each string.
 *
 * @code{.pseudo}
 * Example:
 * s = ["aeiou", "AEIOU", "0123456789", "bcdOPQ5"]
 * f = [{'M','Z'}, {'a','l'}, {'4','6'}]
 * r1 = filter_characters(s, f)
 * r1 is now ["aei", "OU", "456", "bcdOPQ5"]
 * r2 = filter_characters(s, f, REMOVE)
 * r2 is now ["ou", "AEI", "0123789", ""]
 * r3 = filter_characters(s, f, KEEP, "*")
 * r3 is now ["aei**", "***OU", "****456***", "bcdOPQ5"]
 * @endcode
 *
 * Null string entries result in null entries in the output column.
 *
 * @throw cudf::logic_error if `replacement` is invalid
 *
 * @param input Strings instance for this operation
 * @param characters_to_filter Table of character ranges to filter on
 * @param keep_characters If true, the `characters_to_filter` are retained and all other characters
 * are removed
 * @param replacement Optional replacement string for each character removed
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with filtered strings
 */
std::unique_ptr<column> filter_characters(
  strings_column_view const& input,
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> characters_to_filter,
  filter_type keep_characters       = filter_type::KEEP,
  string_scalar const& replacement  = string_scalar(""),
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
