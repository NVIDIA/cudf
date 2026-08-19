/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {

class column_device_view;

namespace strings {

class regex_program;

namespace detail {

/**
 * @brief Returns a column of regex match counts for each string in the given column.
 *
 * A null entry will result in a zero count for that output row.
 *
 * This overload evaluates against an already-built device regex program. Callers that
 * also need the device program for other work (e.g. extraction) should build it once
 * and pass it here to avoid a redundant device program build.
 *
 * @tparam ProgDevice The regex prog device instance used for this API
 * @param d_strings Device view of the input strings column
 * @param d_prog Device regex program to evaluate on each string
 * @param strings_count Number of strings (and rows in the output column)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Integer column of match counts
 */
template <typename ProgDevice>
std::unique_ptr<column> count_matches(column_device_view const& d_strings,
                                      ProgDevice& d_prog,
                                      size_type strings_count,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Returns a column of regex match counts for each string in the given column.
 *
 * A null entry will result in a zero count for that output row.
 *
 * This overload builds its own device regex program. Prefer the overload above when
 * the device program is also needed for other work on the same call site.
 *
 * @param d_strings Device view of the input strings column.
 * @param prog Regex program to evaluate on each string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Integer column of match counts
 */
std::unique_ptr<column> count_matches(column_device_view const& d_strings,
                                      regex_program const& prog,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
