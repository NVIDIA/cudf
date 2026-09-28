/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf {
namespace io {
namespace detail {
namespace csv {

std::unique_ptr<column> pandas_format_durations(column_view const& durations,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf
