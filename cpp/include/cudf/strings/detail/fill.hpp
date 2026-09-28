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

#include <cuda/stream>

namespace cudf {
namespace strings::detail {
/**
 * @brief Returns a strings column replacing a range of rows
 * with the specified string.
 *
 * If the value parameter is invalid, the specified rows are filled with
 * null entries.
 *
 * @throw cudf::logic_error if [begin,end) is outside the range of the input column.
 *
 * @param strings Strings column to fill.
 * @param begin First row index to include the new string.
 * @param end Last row index (exclusive).
 * @param value String to use when filling the range.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> fill(strings_column_view const& strings,
                             size_type begin,
                             size_type end,
                             string_scalar const& value,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace cudf
