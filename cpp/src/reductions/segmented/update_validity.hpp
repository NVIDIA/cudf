/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <optional>

namespace cudf {
namespace reduction {
namespace detail {

/**
 * @brief Compute the validity mask and set it on the result column
 *
 * If `null_handling == null_policy::INCLUDE`, all elements in a segment must be valid for the
 * reduced value to be valid.
 * If `null_handling == null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @param result Result of segmented reduce to update the null mask
 * @param col Input column before reduce
 * @param offsets Indices to segment boundaries
 * @param null_handling How null entries are processed within each segment
 * @param init Optional initial value
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
void segmented_update_validity(column& result,
                               column_view const& col,
                               device_span<size_type const> offsets,
                               null_policy null_handling,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
