/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::groupby::detail::hash {

/**
 * @brief Determine if all of provided aggregations can be computed using shared memory kernels.
 *
 * @param agg_kinds The aggregation kinds to check
 * @param values The input values table corresponding to the aggregation kinds
 * @param grid_size The CUDA grid size to be used for launching the aggregation kernels
 * @return A pair consisting of a boolean indicating if all aggregations can be computed using
 *         shared memory kernels, and the currently available shared memory size
 */
std::pair<bool, size_type> is_shared_memory_compatible(host_span<aggregation::Kind const> agg_kinds,
                                                       table_view const& values,
                                                       size_type grid_size);

/**
 * @brief Computes all aggregations from `requests` that can run only a single pass over the data
 *        and stores the results in `cache`.
 *
 * @return A pair containing a gather map to collect the unique keys from the input keys table, and
 *         a boolean indicating if there are any compound aggregations to process further
 */
template <typename SetType>
std::pair<rmm::device_uvector<size_type>, bool> compute_single_pass_aggs(
  SetType& global_set,
  bitmask_type const* row_bitmask,
  std::span<aggregation_request const> requests,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail::hash
