/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

//! NVText APIs
namespace CUDF_EXPORT nvtext {
/**
 * @addtogroup nvtext_edit_distance
 * @{
 * @file
 * @brief APIs for computing the Levenshtein edit distance between strings.
 */

/**
 * @brief Compute the edit distance between individual strings in two strings columns.
 *
 * The `output[i]` is the edit distance between `input[i]` and `targets[i]`.
 * This edit distance calculation uses the Levenshtein algorithm as documented here:
 * https://en.wikipedia.org/wiki/Levenshtein_distance
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "", "world"]
 * t = ["hallo", "goodbye", "world"]
 * d = edit_distance(s, t)
 * d is now [1, 7, 0]
 * @endcode
 *
 * Any null entries for either `input` or `targets` is ignored and the edit distance
 * is computed as though the null entry is an empty string.
 *
 * The `targets.size()` must equal `input.size()` unless `targets.size()==1`.
 * In this case, all `input` will be computed against the single `targets[0]` string.
 *
 * @throw std::invalid_argument if `targets.size() != input.size()` and if `targets.size() != 1`
 * @throw std::invalid_argument if `targets.size() == 1` and `targets[0].is_null()`
 *
 * @param input Strings column of input strings
 * @param targets Strings to compute edit distance against `input`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column of edit distance values
 */
std::unique_ptr<cudf::column> edit_distance(
  cudf::strings_column_view const& input,
  cudf::strings_column_view const& targets,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
