/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace strings::detail {

/**
 * @brief Internal API to copy a range of string elements out-of-place from
 * a source column to a target column
 *
 * Creates a new column as if an in-place copy was performed into `target`.
 * The elements indicated by the indices `source_begin`, `source_end`)
 * replace with the elements in the target column starting at `target_begin`.
 * Elements outside the range are copied from `target` into the new target
 * column to return.
 *
 * @throws cudf::logic_error for invalid range (if `target_begin < 0`,
 * or `target_begin >= target.size()`,
 * or `target_begin + (source_end-source_begin)` > target.size()`).
 *
 * @param source The strings column to copy from inside the `target_begin` range
 * @param target The strings column to copy from outside the range
 * @param source_begin Index of the first element in the source range
 * @param source_end The index of the last element in the source range (exclusive)
 * @param target_begin The starting index of the target range (inclusive)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return The result target column
 */
std::unique_ptr<column> copy_range(strings_column_view const& source,
                                   strings_column_view const& target,
                                   size_type source_begin,
                                   size_type source_end,
                                   size_type target_begin,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace cudf
