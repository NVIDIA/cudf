/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace lists::detail {

/**
 * @copydoc cudf::lists::sort_lists
 */
std::unique_ptr<column> sort_lists(lists_column_view const& input,
                                   order column_order,
                                   null_order null_precedence,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::stable_sort_lists
 */
std::unique_ptr<column> stable_sort_lists(lists_column_view const& input,
                                          order column_order,
                                          null_order null_precedence,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace cudf
