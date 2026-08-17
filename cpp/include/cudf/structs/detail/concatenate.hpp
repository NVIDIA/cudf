/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

namespace cudf {
namespace structs::detail {

/**
 * @brief Returns a single column by concatenating the given vector of structs columns.
 *
 * @code{.pseudo}
 * s1 = [ col0 : {0, 1}
 *        col1 : {2, 3, 4, 5, 6}
 *        col2 : {"abc", "def", "ghi"} ]
 *
 * s2 = [ col0 : {7, 8}
 *        col1 : {-4, -5, -6}
 *        col2 : {"uvw", "xyz"} ]
 *
 * r = concatenate({s1, s2})
 *
 * r is now [ col0: {0, 1, 7, 8}
 *            col1: {2, 3, 4, 5, 6, -4, -5, -6}
 *            col2: {"abc", "def", "ghi", "uvw", "xyz"} ]
 * @endcode
 *
 * @param columns Vector of structs columns to concatenate.
 * @param stream  CUDA stream used for device memory operations and kernel launches.
 * @param mr      Device memory resource used to allocate the returned column's device memory.
 * @return        New column with concatenated results.
 */
std::unique_ptr<column> concatenate(host_span<column_view const> columns,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

}  // namespace structs::detail
}  // namespace cudf
