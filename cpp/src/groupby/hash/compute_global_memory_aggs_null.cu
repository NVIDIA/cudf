/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_global_memory_aggs.hpp"
#include "compute_global_memory_aggs_null.hpp"
#include "helpers.cuh"

#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace cudf::groupby::detail::hash {

template <>
std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs<nullable_global_set_t>(bitmask_type const* row_bitmask,
                                                  table_view const& values,
                                                  nullable_global_set_t const& key_set,
                                                  host_span<aggregation::Kind const> h_agg_kinds,
                                                  device_span<aggregation::Kind const> d_agg_kinds,
                                                  std::span<int8_t const> is_agg_intermediate,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  return h_agg_kinds.size() > GROUPBY_DENSE_OUTPUT_THRESHOLD
           ? compute_global_memory_aggs_null_dense(row_bitmask,
                                                   values,
                                                   key_set,
                                                   h_agg_kinds,
                                                   d_agg_kinds,
                                                   is_agg_intermediate,
                                                   stream,
                                                   mr)
           : compute_global_memory_aggs_null_sparse(row_bitmask,
                                                    values,
                                                    key_set,
                                                    h_agg_kinds,
                                                    d_agg_kinds,
                                                    is_agg_intermediate,
                                                    stream,
                                                    mr);
}

}  // namespace cudf::groupby::detail::hash
