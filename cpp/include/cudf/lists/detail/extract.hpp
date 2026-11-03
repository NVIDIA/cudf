/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/lists/extract.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace lists::detail {

/**
 * @copydoc cudf::lists::extract_list_element(lists_column_view, size_type,
 * rmm::device_async_resource_ref)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             size_type const index,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::extract_list_element(lists_column_view, column_view const&,
 * rmm::device_async_resource_ref)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             column_view const& indices,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace cudf
