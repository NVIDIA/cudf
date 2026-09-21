/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sort.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @brief Check if the strings fast-path sort is available for the given column
 *
 * @param column The column to check
 * @return true if the strings fast-path sort is available, false otherwise
 */
bool is_strings_sortable(column_view const& column);

/**
 * @brief Sort indices of a single strings column
 *
 * This should only be used for non-empty strings columns with no nulls.
 *
 * @tparam method Whether to use stable sort
 * @param input The strings column to sort
 * @param indices The output sorted indices
 * @param ascending The sort order
 * @param stream The CUDA stream to use
 */
template <sort_method method>
void sorted_order_strings(column_view const& input,
                          mutable_column_view& indices,
                          bool ascending,
                          cuda::stream_ref stream);

}  // namespace detail
}  // namespace cudf
