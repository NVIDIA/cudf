/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace lists::detail {

/**
 * @copydoc cudf::list::have_overlap
 */
std::unique_ptr<column> have_overlap(lists_column_view const& lhs,
                                     lists_column_view const& rhs,
                                     null_equality nulls_equal,
                                     nan_equality nans_equal,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::list::intersect_distinct
 */
std::unique_ptr<column> intersect_distinct(lists_column_view const& lhs,
                                           lists_column_view const& rhs,
                                           null_equality nulls_equal,
                                           nan_equality nans_equal,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::list::union_distinct
 */
std::unique_ptr<column> union_distinct(lists_column_view const& lhs,
                                       lists_column_view const& rhs,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::list::difference_distinct
 */
std::unique_ptr<column> difference_distinct(lists_column_view const& lhs,
                                            lists_column_view const& rhs,
                                            null_equality nulls_equal,
                                            nan_equality nans_equal,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

/** @} */  // end of group
}  // namespace lists::detail
}  // namespace cudf
