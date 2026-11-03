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
 * @brief Internal API to replace nulls with preceding/following non-null values in @p value
 *
 * @param grouped_value A column whose null values will be replaced.
 * @param group_labels Group labels for @p grouped_value, corresponding to group keys.
 * @param replace_policy Specify the position of replacement values relative to null values.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate device memory of the returned column.
 */
std::unique_ptr<column> group_replace_nulls(cudf::column_view const& grouped_value,
                                            device_span<size_type const> group_labels,
                                            cudf::replace_policy replace_policy,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
