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

/// Thread-block size of warp aggregation. Rows of the same thread block share one partial.
inline constexpr int partial_aggregation_block_size = 256;

/// Combines row values with equal group IDs within each warp. Rows with a negative ID are skipped.
/// `output` holds `partials_per_group` copies of `capacity` groups; each thread block updates one
/// copy, so concurrent blocks spread their atomics over different copies of each group.
void compute_warp_aggregates(table_view const& values,
                             host_span<aggregation::Kind const> kinds,
                             device_span<size_type const> group_ids,
                             size_type capacity,
                             size_type partials_per_group,
                             mutable_table_view const& output,
                             cuda::stream_ref stream,
                             cudf::memory_resources mr);

}  // namespace cudf::groupby
