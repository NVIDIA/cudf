/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::distinct_count(column_view const&, null_policy, nan_policy, cuda::stream_ref)
 */
cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               cuda::stream_ref stream);

/**
 * @copydoc cudf::distinct_count(table_view const&, null_equality, cuda::stream_ref)
 */
cudf::size_type distinct_count(table_view const& input,
                               null_equality nulls_equal,
                               cuda::stream_ref stream);

}  // namespace detail
}  // namespace cudf
