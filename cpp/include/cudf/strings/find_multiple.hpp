/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Strings column APIs for searching each string for multiple target strings at once.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_find
 * @{
 */

/**
 * @brief Searches for the given target strings within each string in the provided column
 *
 * Each column in the result table corresponds to the result for the target string at the same
 * ordinal. i.e. 0th column is the BOOL8 column result for the 0th target string, 1st for 1st,
 * etc.
 *
 * If the target is not found for a string, false is returned for that entry in the output column.
 * If the target is an empty string, true is returned for all non-null entries in the output column.
 *
 * Any null input strings return corresponding null entries in the output columns.
 *
 * @code{.pseudo}
 * input = ["a", "b", "c"]
 * targets = ["a", "c"]
 * output is a table with two boolean columns:
 *   column 0: [true, false, false]
 *   column 1: [false, false, true]
 * @endcode
 *
 * @throw std::invalid_argument if `targets` is empty or contains nulls
 *
 * @param input Strings instance for this operation
 * @param targets UTF-8 encoded strings to search for in each string in `input`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Table of BOOL8 columns
 */
std::unique_ptr<table> contains_multiple(
  strings_column_view const& input,
  strings_column_view const& targets,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Searches for the given target strings within each string in the provided column
 * and returns the position the targets were found
 *
 * The size of the output column is `input.size()`.
 * Each row of the output column is of size `targets.size()`.
 *
 * `output[i,j]` contains the position of `targets[j]` in `input[i]`
 *
 * @code{.pseudo}
 * Example:
 * s = ["abc", "def"]
 * t = ["a", "c", "e"]
 * r = find_multiple(s, t)
 * r is now {[ 0, 2,-1],   // for "abc": "a" at pos 0, "c" at pos 2, "e" not found
 *           [-1,-1, 1 ]}  // for "def": "a" and "b" not found, "e" at  pos 1
 * @endcode
 *
 * @throw std::invalid_argument if `targets` is empty or contains nulls
 *
 * @param input Strings instance for this operation
 * @param targets Strings to search for in each string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Lists column with character position values
 */
std::unique_ptr<column> find_multiple(
  strings_column_view const& input,
  strings_column_view const& targets,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
