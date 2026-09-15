/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/lists/reverse.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace lists::detail {

/**
 * @copydoc cudf::lists::reverse
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<column> reverse(lists_column_view const& input,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace cudf
