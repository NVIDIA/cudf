/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/replace.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

namespace cudf {
namespace groupby {
namespace detail {

/**
 * @brief Internal API to replace nulls with preceding/following non-null values in @p values
 *
 * The result is in grouped order: row `i` of the result is row `grouped_order[i]` of @p values
 * with its null replaced by the nearest non-null value of the same group.
 *
 * @param values The ungrouped column whose null values will be replaced.
 * @param grouped_order Row of @p values at each grouped position, in input order within groups.
 * @param group_labels Group label of each grouped position.
 * @param replace_policy Specify the position of replacement values relative to null values.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate device memory of the returned column.
 */
std::unique_ptr<column> group_replace_nulls(cudf::column_view const& values,
                                            device_span<size_type const> grouped_order,
                                            device_span<size_type const> group_labels,
                                            cudf::replace_policy replace_policy,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
