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

#include <optional>

/**
 * @file
 * @brief Strings column APIs for replacing substrings matching a regex pattern.
 */

namespace CUDF_EXPORT cudf {
namespace strings {

struct regex_program;

/**
 * @addtogroup strings_replace
 * @{
 */

/**
 * @brief For each string, replaces any character sequence matching the given regex
 * with the provided replacement string.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param replacement The string used to replace the matched sequence in each string.
 *        Default is an empty string.
 * @param max_replace_count The maximum number of times to replace the matched pattern
 *        within each string. Default replaces every substring that is matched.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> replace_re(
  strings_column_view const& input,
  regex_program const& prog,
  string_scalar const& replacement           = string_scalar(""),
  std::optional<size_type> max_replace_count = std::nullopt,
  rmm::cuda_stream_view stream               = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr          = cudf::get_current_device_resource_ref());

/**
 * @brief For each string, replaces any character sequence matching the given regex
 * using the replacement template for back-references.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * See the @ref md_regex "Regex Features" page for details on patterns supported by this API.
 *
 * @throw cudf::logic_error if capture index values in `replacement` are not in range 0-99, and also
 * if the index exceeds the group count specified in the pattern
 *
 * @param input Strings instance for this operation
 * @param prog Regex program instance
 * @param replacement The replacement template for creating the output string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> replace_with_backrefs(
  strings_column_view const& input,
  regex_program const& prog,
  std::string_view replacement,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace strings
}  // namespace CUDF_EXPORT cudf
