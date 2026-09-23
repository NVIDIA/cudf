/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf {
namespace lists::detail {

/**
 * @brief Returns a single column by concatenating the given vector of
 * lists columns.
 *
 * @code{.pseudo}
 * s1 = [{0, 1}, {2, 3, 4, 5, 6}]
 * s2 = [{7, 8, 9}, {10, 11}]
 * r = concatenate(s1, s2)
 * r is now [{0, 1}, {2, 3, 4, 5, 6}, {7, 8, 9}, {10, 11}]
 * @endcode
 *
 * @param columns Vector of lists columns to concatenate.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> concatenate(host_span<column_view const> columns,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace cudf
