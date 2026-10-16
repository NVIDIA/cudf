/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf::groupby {

/// Combines row values with equal targets within each warp.
void compute_warp_aggregates(table_view const& values,
                             host_span<aggregation::Kind const> kinds,
                             device_span<size_type const> targets,
                             mutable_table_view const& output,
                             cuda::stream_ref stream,
                             cudf::memory_resources mr);

}  // namespace cudf::groupby
