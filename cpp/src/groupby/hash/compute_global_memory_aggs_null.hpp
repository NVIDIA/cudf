/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_global_memory_aggs.hpp"
#include "helpers.cuh"

#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace cudf::groupby::detail::hash {

std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs_null_dense(bitmask_type const* row_bitmask,
                                      table_view const& values,
                                      nullable_global_set_t const& key_set,
                                      host_span<aggregation::Kind const> h_agg_kinds,
                                      device_span<aggregation::Kind const> d_agg_kinds,
                                      std::span<int8_t const> is_agg_intermediate,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs_null_sparse(bitmask_type const* row_bitmask,
                                       table_view const& values,
                                       nullable_global_set_t const& key_set,
                                       host_span<aggregation::Kind const> h_agg_kinds,
                                       device_span<aggregation::Kind const> d_agg_kinds,
                                       std::span<int8_t const> is_agg_intermediate,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail::hash
