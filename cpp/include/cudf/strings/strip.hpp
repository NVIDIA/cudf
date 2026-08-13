/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/side_type.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Strings column APIs for removing specified characters from the ends of each string.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_modify
 * @{
 */

/**
 * @brief Removes the specified characters from the beginning or end
 * (or both) of each string.
 *
 * The to_strip parameter can contain one or more characters.
 * All characters in `to_strip` are removed from the input strings.
 *
 * If `to_strip` is the empty string, whitespace characters are removed.
 * Whitespace is considered the space character plus control characters
 * like tab and line feed.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @code{.pseudo}
 * Example:
 * s = [" aaa ", "_bbbb ", "__cccc  ", "ddd", " ee _ff gg_"]
 * r = strip(s,both," _")
 * r is now ["aaa", "bbbb", "cccc", "ddd", "ee _ff gg"]
 * @endcode
 *
 * @throw cudf::logic_error if `to_strip` is invalid.
 *
 * @param input Strings column for this operation
 * @param side Indicates characters are to be stripped from the beginning, end, or both of each
 *        string; Default is both
 * @param to_strip UTF-8 encoded characters to strip from each string;
 *        Default is empty string which indicates strip whitespace characters
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> strip(
  strings_column_view const& input,
  side_type side                    = side_type::BOTH,
  string_scalar const& to_strip     = string_scalar(""),
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
