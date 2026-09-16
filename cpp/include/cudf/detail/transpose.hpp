/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::transpose
 */
std::pair<std::unique_ptr<column>, table_view> transpose(table_view const& input,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
