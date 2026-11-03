/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/interop.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::from_dlpack
 */
std::unique_ptr<table> from_dlpack(DLManagedTensor const* managed_tensor,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::to_dlpack
 */
DLManagedTensor* to_dlpack(table_view const& input,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
