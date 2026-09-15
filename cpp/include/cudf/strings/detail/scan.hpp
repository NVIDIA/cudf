/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace strings::detail {
/**
 * @brief Scan function for strings
 *
 * Called by cudf::scan() with only min and max aggregates.
 *
 * @tparam Op Either DeviceMin or DeviceMax operations
 *
 * @param input Input strings column
 * @param mask Mask for scan
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
template <typename Op>
std::unique_ptr<column> scan_inclusive(column_view const& input,
                                       bitmask_type const* mask,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace cudf
