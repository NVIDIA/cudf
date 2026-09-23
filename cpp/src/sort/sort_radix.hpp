/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @brief Check if radix sort is available for the given column
 *
 * @param column The column to check
 * @return true if radix sort is available, false otherwise
 */
bool is_radix_sortable(column_view const& column);

/**
 * @brief Sort a column using radix sort
 *
 * This should only be used for fixed-width columns with no nulls.
 *
 * @param input The column to sort
 * @param ascending The sort order
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 * @return The sorted column
 */
std::unique_ptr<column> sort_radix(column_view const& input,
                                   bool ascending,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @brief Sort a column using radix sort
 *
 * This should only be used for fixed-width columns with no nulls.
 *
 * @param input The column to sort
 * @param indices The indices to return
 * @param ascending The sort order
 * @param stream The CUDA stream to use
 */
void sorted_order_radix(column_view const& input,
                        mutable_column_view& indices,
                        bool ascending,
                        cuda::stream_ref stream);
}  // namespace detail
}  // namespace cudf
