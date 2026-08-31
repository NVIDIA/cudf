/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace strings::detail {

/**
 * @copydoc cudf::strings::replace(strings_column_view const&, string_scalar const&,
 * string_scalar const&, int32_t, cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> replace(strings_column_view const& strings,
                                string_scalar const& target,
                                string_scalar const& repl,
                                int32_t maxrepl,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::strings::replace_multiple(strings_column_view const&, strings_column_view const&,
 * strings_column_view const&, cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> replace_mutiple(strings_column_view const& strings,
                                        strings_column_view const& targets,
                                        strings_column_view const& repls,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr);

/**
 * @brief Replaces any null string entries with the given string.
 *
 * This returns a strings column with no null entries.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", nullptr, "goodbye"]
 * r = replace_nulls(s,"**")
 * r is now ["hello", "**", "goodbye"]
 * @endcode
 *
 * @param strings Strings column for this operation.
 * @param repl Replacement string for null entries. Default is empty string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> replace_nulls(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::strings::replace_slice(strings_column_view const&, string_scalar const&,
 * size_type, size_type, cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> replace_slice(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      size_type start,
                                      size_type stop,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Return a copy of `input` replacing any `values_to_replace[i]`
 * found with `replacement_values[i]`
 *
 * @param input The column to find and replace values
 * @param values_to_replace The values to find
 * @param replacement_values The corresponding replacement values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Copy of `input` with specified values replaced
 */
std::unique_ptr<cudf::column> find_and_replace_all(
  cudf::strings_column_view const& input,
  cudf::strings_column_view const& values_to_replace,
  cudf::strings_column_view const& replacement_values,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace cudf
