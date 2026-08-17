/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

namespace cudf {
class column_device_view;

namespace reduction {
namespace detail {

/**
 * @brief Compute the number of elements per segment
 *
 * If `null_handling == null_policy::EXCLUDE`, the count for each
 * segment omits any null entries. Otherwise, this returns the number
 * of elements in each segment.
 *
 * @param null_mask Null values over which the segment offsets apply
 * @param has_nulls True if d_col contains any nulls
 * @param offsets Indices to segment boundaries
 * @param null_handling How null entries are processed within each segment
 * @param stream Used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The number of elements in each segment
 */
rmm::device_uvector<size_type> segmented_counts(bitmask_type const* null_mask,
                                                bool has_nulls,
                                                device_span<size_type const> offsets,
                                                null_policy null_handling,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
