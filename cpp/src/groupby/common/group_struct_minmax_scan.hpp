/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf {
class column_view;
}

namespace cudf::groupby::detail {

// A private owner lets MIN/MAX share the nested comparator's scan kernel.
void scan_struct_argminmax(column_view const& values,
                           device_span<size_type const> group_labels,
                           size_type* output,
                           bool is_min,
                           cuda::stream_ref stream,
                           cudf::memory_resources mr);

}  // namespace cudf::groupby::detail
