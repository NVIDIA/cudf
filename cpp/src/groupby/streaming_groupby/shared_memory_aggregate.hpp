/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf::groupby::detail::streaming {

/**
 * @brief Aggregates a batch into the streaming results using per-block shared memory copies of
 * the whole result table.
 *
 * The aggregation is only performed when all `num_groups` results fit in shared memory and the
 * batch is large enough to amortize the per-block flush to global memory.
 *
 * @param num_rows Number of rows in the batch
 * @param target_indices Dense result row of each batch row, or `CUDF_SIZE_TYPE_SENTINEL` to skip
 * @param num_groups Upper bound (exclusive) of every valid value in `target_indices`
 * @param input_values Value columns of the batch, one per aggregation
 * @param output_values Result columns, one per aggregation
 * @param d_input_values Device view of `input_values`
 * @param d_output_values Device view of `output_values`
 * @param agg_kinds Aggregation kinds, one per value column
 * @param d_agg_kinds Device copy of `agg_kinds`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return `true` if the kernel was launched, `false` if the caller must aggregate the batch
 */
bool shared_memory_aggregate(size_type num_rows,
                             size_type const* target_indices,
                             size_type num_groups,
                             table_view const& input_values,
                             table_view const& output_values,
                             table_device_view const& d_input_values,
                             mutable_table_device_view const& d_output_values,
                             host_span<aggregation::Kind const> agg_kinds,
                             aggregation::Kind const* d_agg_kinds,
                             cuda::stream_ref stream);

}  // namespace cudf::groupby::detail::streaming
