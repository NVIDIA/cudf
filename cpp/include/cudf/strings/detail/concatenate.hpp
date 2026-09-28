/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf {
namespace strings::detail {
/**
 * @brief Returns a single column by vertically concatenating the given vector of
 * strings columns.
 *
 * ```
 * s1 = ['aa', 'bb', 'cc']
 * s2 = ['dd', 'ee']
 * r = concatenate_vertically([s1,s2])
 * r is now ['aa', 'bb', 'cc', 'dd', 'ee']
 * ```
 *
 * @param columns List of string columns to concatenate.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> concatenate(host_span<column_view const> columns,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace cudf
