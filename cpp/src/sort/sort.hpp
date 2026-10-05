/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @brief The enum specifying which sorting method to use (stable or unstable).
 */
enum class sort_method : bool { STABLE, UNSTABLE };

/**
 * @brief Sort table row indices using a shared owner for stable and unstable instantiations.
 *
 * Keeping both instantiations in one translation unit avoids duplicating their common kernels.
 */
std::unique_ptr<column> sorted_order_impl(table_view const& input,
                                          std::vector<order> const& column_order,
                                          std::vector<null_order> const& null_precedence,
                                          sort_method method,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @brief Sort indices of a single column.
 *
 * This API offers fast sorting for primitive types. It cannot handle nested types and will not
 * consider `NaN` as equivalent to other `NaN`.
 *
 * @tparam method Whether to use stable sort
 * @param input Column to sort. The column data is not modified.
 * @param column_order Ascending or descending sort order
 * @param null_precedence How null rows are to be ordered
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Sorted indices for the input column.
 */
template <sort_method method>
std::unique_ptr<column> sorted_order(column_view const& input,
                                     order column_order,
                                     null_order null_precedence,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
