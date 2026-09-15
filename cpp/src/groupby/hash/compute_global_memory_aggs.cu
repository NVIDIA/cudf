/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_global_memory_aggs.cuh"

#include <span>

namespace cudf::groupby::detail::hash {

template std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs<global_set_t>(bitmask_type const* row_bitmask,
                                         table_view const& values,
                                         global_set_t const& key_set,
                                         host_span<aggregation::Kind const> h_agg_kinds,
                                         device_span<aggregation::Kind const> d_agg_kinds,
                                         std::span<int8_t const> is_agg_intermediate,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail::hash
