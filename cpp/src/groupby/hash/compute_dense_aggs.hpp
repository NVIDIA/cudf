/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <cuda/stream>

#include <cstdint>

namespace cudf {
class table_device_view;
class mutable_table_device_view;
}  // namespace cudf

namespace cudf::groupby::detail::hash {
/**
 * @brief Keep dense aggregation's device instantiation in one TU
 *
 * Hash and streaming groupby share this fixed-signature owner to avoid emitting the same
 * aggregation dispatch kernel in each caller TU. Callers retain their device views so streaming
 * groupby can reuse its persistent results view without additional allocations.
 */
void compute_single_pass_aggs_dense_output(size_type const* target_indices,
                                           aggregation::Kind const* d_agg_kinds,
                                           table_device_view const& values,
                                           mutable_table_device_view const& agg_results,
                                           int64_t num_elements,
                                           cuda::stream_ref stream);
}  // namespace cudf::groupby::detail::hash
