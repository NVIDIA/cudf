/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

namespace cudf {

class rolling_aggregation;
class table_view;

namespace detail {
/**
 * @brief Checks if it is possible to optimize fully UNBOUNDED window function.
 *
 * @return true if the window aggregation can optimized, i.e. if it is unbounded-preceding,
 * unbounded-following, if it has a supported aggregation type, and if min_periods is 1.
 * @return false if the window aggregation cannot be optimized.
 */
bool can_optimize_unbounded_window(bool unbounded_preceding,
                                   bool unbounded_following,
                                   size_type min_periods,
                                   rolling_aggregation const& agg);

/**
 * @brief Optimized bypass for fully UNBOUNDED window functions.
 *
 * @return the result column from running the unbounded window aggregation,
 * via the optimized aggregation/reduction path.
 */
std::unique_ptr<column> optimized_unbounded_window(table_view const& group_keys,
                                                   column_view const& input,
                                                   rolling_aggregation const& aggr,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr);
}  // namespace detail
}  // namespace cudf
