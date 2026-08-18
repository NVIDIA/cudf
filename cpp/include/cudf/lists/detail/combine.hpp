/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace lists::detail {
/**
 * @copydoc cudf::lists::concatenate_rows
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate_rows(table_view const& input,
                                         concatenate_null_policy null_policy,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::concatenate_list_elements
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate_list_elements(column_view const& input,
                                                  concatenate_null_policy null_policy,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace cudf
