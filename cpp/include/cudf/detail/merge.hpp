/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/utility>

namespace cudf {
namespace detail {

/**
 * @brief Source table identifier to copy data from.
 */
enum class side : bool { LEFT, RIGHT };

/**
 * @brief Tagged index type: `cuda::std::get<0>` indicates left/right side,
 * `cuda::std::get<1>` indicates the row index
 */
using index_type = cuda::std::pair<side, cudf::size_type>;

/**
 * @brief Vector of `index_type` values.
 */
using index_vector = rmm::device_uvector<index_type>;

/**
 * @copydoc std::unique_ptr<cudf::table> merge(
 *            std::vector<table_view> const& tables_to_merge,
 *            std::vector<cudf::size_type> const& key_cols,
 *            std::vector<cudf::order> const& column_order,
 *            std::vector<cudf::null_order> const& null_precedence,
 *            cuda::stream_ref stream,
 *            rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::table> merge(std::vector<table_view> const& tables_to_merge,
                                   std::vector<cudf::size_type> const& key_cols,
                                   std::vector<cudf::order> const& column_order,
                                   std::vector<cudf::null_order> const& null_precedence,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
