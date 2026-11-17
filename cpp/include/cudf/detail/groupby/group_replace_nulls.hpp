/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/replace.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

namespace cudf {
namespace groupby {
namespace detail {

/**
 * @brief Internal API to replace nulls with preceding/following non-null values in @p value
 *
 * The result is in grouped order: row `i` of the result is row `grouped_order[i]` of @p value
 * with its null replaced by the nearest non-null value of the same group.
 *
 * @param value An ungrouped column whose null values will be replaced.
 * @param grouped_order Row of @p value at each grouped position, in input order within groups.
 * @param group_labels Group labels for @p grouped_order, corresponding to group keys.
 * @param replace_policy Specify the position of replacement values relative to null values.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resources for the returned column and temporary storage.
 */
std::unique_ptr<column> group_replace_nulls(cudf::column_view const& value,
                                            device_span<size_type const> grouped_order,
                                            device_span<size_type const> group_labels,
                                            cudf::replace_policy replace_policy,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr);

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
