/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/detail/merge.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

namespace cudf {
namespace strings::detail {
/**
 * @brief Merges two strings columns
 *
 * @param lhs First column
 * @param rhs Second column
 * @param row_order Indices for each column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> merge(strings_column_view const& lhs,
                              strings_column_view const& rhs,
                              cudf::detail::index_vector const& row_order,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace cudf
