/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Strings column API for wrapping long strings onto multiple lines.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_modify
 * @{
 */

/**
 * @brief Wraps strings onto multiple lines shorter than `width` by replacing appropriate white
 * space with new-line characters (ASCII 0x0A).
 *
 * For each string row in the input column longer than `width`, the corresponding output string row
 * will have newline characters inserted so that each line is no more than `width characters`.
 * Attempts to use existing white space locations to split the strings, but may split
 * non-white-space sequences if necessary.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * Example 1:
 * ```
 * width = 3
 * input_string_tbl = [ "12345", "thesé", nullptr, "ARE THE", "tést strings", "" ];
 *
 * wrapped_string_tbl = wrap(input_string_tbl, width)
 * wrapped_string_tbl = [ "12345", "thesé", nullptr, "ARE\nTHE", "tést\nstrings", "" ]
 * ```
 *
 * Example 2:
 * ```
 * width = 12;
 * input_string_tbl = ["the quick brown fox jumped over the lazy brown dog", "hello, world"]
 *
 * wrapped_string_tbl = wrap(input_string_tbl, width)
 * wrapped_string_tbl = ["the quick\nbrown fox\njumped over\nthe lazy\nbrown dog", "hello, world"]
 * ```
 *
 * @param input String column
 * @param width Maximum character width of a line within each string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of wrapped strings
 */
std::unique_ptr<column> wrap(
  strings_column_view const& input,
  size_type width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
