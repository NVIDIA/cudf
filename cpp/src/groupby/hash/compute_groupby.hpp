/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/groupby.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf::groupby::detail::hash {
/**
 * @brief Computes groupby aggregations over the helper's HashCSR grouping.
 *
 * All the aggregations which can be computed in a single pass are computed first by the same set
 * of kernels. Then using these results, compound aggregations that require multiple passes will be
 * computed on top of them. All results are stored into the in/out parameter `cache`.
 *
 * @param requests The set of columns to aggregate and the aggregations to perform
 * @param helper Grouping shared by all aggregations of the groupby operation
 * @param cache Dense aggregation results
 * @param expose_intermediates Whether to compute extracted intermediate results like requested
 * results, because a host UDF may request them from `cache`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resources used for the aggregation results and temporary storage
 */
void compute_groupby(std::span<aggregation_request const> requests,
                     groupby_helper& helper,
                     cudf::detail::result_cache* cache,
                     bool expose_intermediates,
                     cuda::stream_ref stream,
                     cudf::memory_resources mr);
}  // namespace cudf::groupby::detail::hash
