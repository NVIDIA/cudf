/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/groupby.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <span>

namespace cudf::groupby::detail {
struct groupby_helper;

namespace hash {
/**
 * @brief Computes direct aggregations over the helper's cached grouping.
 *
 * Primitive reductions are computed first, batching compatible requests. Compound results are
 * then finalized from those reductions. Results already present in `cache` are reused.
 *
 * @param requests The set of columns to aggregate and the aggregations to perform
 * @param helper Cached grouping shared by all aggregation requests
 * @param cache Dense aggregation results
 * @param expose_intermediates Preserve result masks and output resource ownership for dependencies
 * that a host UDF may request dynamically
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resources used for aggregation results and temporary storage
 */
void compute_aggregations(std::span<aggregation_request const> requests,
                          groupby_helper& helper,
                          cudf::detail::result_cache& cache,
                          bool expose_intermediates,
                          cuda::stream_ref stream,
                          cudf::memory_resources mr);
}  // namespace hash
}  // namespace cudf::groupby::detail
