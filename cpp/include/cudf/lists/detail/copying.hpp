/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace lists::detail {

/**
 * @brief Returns a new lists column created from a subset of the
 * lists column. The subset of lists selected is between start (inclusive)
 * and end (exclusive).
 *
 * @code{.pseudo}
 * Example:
 * s1 = {{1, 2, 3}, {4, 5}, {6, 7}, {}, {8, 9}}
 * s2 = slice( s1, 1, 4)
 * s2 is {{4, 5}, {6, 7}, {}}
 * @endcode
 *
 * @param lists Lists instance for this operation.
 * @param start Index to first list to select in the column
 * @param end One past the index to last list to select in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New lists column of size (end - start)
 */
std::unique_ptr<cudf::column> copy_slice(lists_column_view const& lists,
                                         size_type start,
                                         size_type end,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace cudf
