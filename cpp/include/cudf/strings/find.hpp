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
 * @brief Strings column APIs for locating and testing substrings within each string.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_find
 * @{
 */

/**
 * @brief Returns a column of character position values where the target
 * string is first found in each string of the provided column.
 *
 * If `target` is not found, -1 is returned for that row entry in the output column.
 *
 * The target string is searched within each string in the character
 * position range [start,stop). If the stop parameter is -1, then the
 * end of each string becomes the final position to include in the search.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @throw cudf::logic_error if start position is greater than stop position.
 *
 * @param input Strings instance for this operation
 * @param target UTF-8 encoded string to search for in each string
 * @param start First character position to include in the search
 * @param stop Last position (exclusive) to include in the search.
 *             Default of -1 will search to the end of the string.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New integer column with character position values
 */
std::unique_ptr<column> find(
  strings_column_view const& input,
  string_scalar const& target,
  size_type start                   = 0,
  size_type stop                    = -1,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column of character position values where the target
 * string is first found searching from the end of each string.
 *
 * If `target` is not found, -1 is returned for that entry.
 *
 * The target string is searched within each string in the character
 * position range [start,stop). If the stop parameter is -1, then the
 * end of each string becomes the final position to include in the search.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @throw cudf::logic_error if start position is greater than stop position.
 *
 * @param input Strings instance for this operation
 * @param target UTF-8 encoded string to search for in each string
 * @param start First position to include in the search
 * @param stop Last position (exclusive) to include in the search.
 *             Default of -1 will search starting at the end of the string.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New integer column with character position values
 */
std::unique_ptr<column> rfind(
  strings_column_view const& input,
  string_scalar const& target,
  size_type start                   = 0,
  size_type stop                    = -1,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column of character position values where the target
 * string is first found in the corresponding string of the provided column
 *
 * The output of row `i` is the character position of the target string for row `i`
 * within input string of row `i` starting at the character position `start`.
 * If the target is not found within the input string, -1 is returned for that
 * row entry in the output column.
 *
 * Any null input or target entries return corresponding null output column entries.
 *
 * @throw cudf::logic_error if `input.size() != target.size()`
 *
 * @param input Strings to search against
 * @param target Strings to search for in `input`
 * @param start First character position to include in the search
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New integer column with character position values
 */
std::unique_ptr<column> find(
  strings_column_view const& input,
  strings_column_view const& target,
  size_type start                   = 0,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column of character position values where the index-th target
 * string is found in each string of the input
 *
 * The output of row `i` is the character position of the index-th target string for row `i`
 * within input string of row `i` starting at the character position `start`.
 * If the index-th target is not found within the input string, -1 is returned for that
 * row entry in the output column.
 *
 * @code{.pseudo}
 * Example:
 * s = [ 'aaaaa', 'aabbccbbaa', 'bbcc', 'bbaagg' ]
 * r = find_instance(s, 'aa', 1)
 * r is [ 1, 8, -1, -1 ]
 * @endcode
 *
 * Any null input rows return corresponding null output column rows.
 * This API produces the same output as `find()` when `instance == 0`.
 *
 * @param input Strings for this operation
 * @param target UTF-8 encoded string to search for in each string
 * @param instance The instance of the target string to locate (0-based index)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New integer column with character position values
 */
std::unique_ptr<column> find_instance(
  strings_column_view const& input,
  string_scalar const& target,
  size_type instance                = 0,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found within that string in the provided column.
 *
 * If the `target` is not found for a string, false is returned for that entry in the output column.
 * If `target` is an empty string, true is returned for all non-null entries in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param input Strings instance for this operation
 * @param target UTF-8 encoded string to search for in each string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New BOOL8 column
 */
std::unique_ptr<column> contains(
  strings_column_view const& input,
  string_scalar const& target,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the corresponding target string was found within that string in the provided column.
 *
 * `output[i] = true` if string `targets[i]` is found inside `input[i]` otherwise
 * `output[i] = false`.
 * If `target[i]` is an empty string, true is returned for `output[i]`.
 * If `target[i]` is null, false is returned for `output[i]`.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @throw cudf::logic_error if `strings.size() != targets.size()`.
 *
 * @param input Strings instance for this operation
 * @param targets Strings column of targets to check row-wise in `strings`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New BOOL8 column
 */
std::unique_ptr<column> contains(
  strings_column_view const& input,
  strings_column_view const& targets,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found at the beginning of that string in the provided column.
 *
 * If `target` is not found at the beginning of a string, false is set for
 * that row entry in the output column.
 * If `target` is an empty string, true is returned for all non-null entries in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param input Strings instance for this operation
 * @param target UTF-8 encoded string to search for in each string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New type_id::BOOL8 column.
 */
std::unique_ptr<column> starts_with(
  strings_column_view const& input,
  string_scalar const& target,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * corresponding string in target column was found at the beginning of that string in
 * the provided column.
 *
 * If `targets[i]` is not found at the beginning of a string in `strings[i]`, false is set for
 * that row entry in the output column.
 * If `targets[i]` is an empty string, true is returned for corresponding entry in the
 * output column.
 *
 * Any null string entries in `targets` return corresponding null entries in the output columns.
 *
 * @throw cudf::logic_error if `strings.size() != targets.size()`.
 *
 * @param input Strings instance for this operation
 * @param targets Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New BOOL8 column
 */
std::unique_ptr<column> starts_with(
  strings_column_view const& input,
  strings_column_view const& targets,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found at the end of that string in the provided column.
 *
 * If `target` is not found at the end of a string, false is set for
 * that row entry in the output column.
 * If `target` is an empty string, true is returned for all non-null entries in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param input Strings instance for this operation
 * @param target UTF-8 encoded string to search for in each string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New BOOL8 column
 */
std::unique_ptr<column> ends_with(
  strings_column_view const& input,
  string_scalar const& target,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * corresponding string in target column was found at the end of that string in
 * the provided column.
 *
 * If `targets[i]` is not found at the end of a string in `strings[i]`, false is set for
 * that row entry in the output column.
 * If `targets[i]` is an empty string, true is returned for the corresponding entry in the
 * output column.
 *
 * Any null string entries in `targets` return corresponding null entries in the output columns.
 *
 * @throw cudf::logic_error if `strings.size() != targets.size()`.
 *
 * @param input Strings instance for this operation
 * @param targets Strings instance for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New BOOL8 column
 */
std::unique_ptr<column> ends_with(
  strings_column_view const& input,
  strings_column_view const& targets,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the number of times the given target string
 * matches in each string
 *
 * Counting proceeds left to right within the row and does not include
 * overlapping matches.
 *
 * @code{.pseudo}
 * Example:
 * s = ["abababa", "bab", "aba"]
 * r = count(s, "aba")
 * r is now [2, 0, 1]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @param input Strings instance for this operation
 * @param target String to search for in each row of the input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with counts for each row
 */
std::unique_ptr<column> count(
  strings_column_view const& input,
  string_scalar const& target,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
