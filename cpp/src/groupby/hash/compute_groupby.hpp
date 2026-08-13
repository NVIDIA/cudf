/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cudf::groupby::detail::hash {
/**
 * @brief Computes groupby using hash table.
 *
 * First, we create a hash table that stores the indices of unique rows in
 * `keys`. The upper limit on the number of values in this map is the number
 * of rows in `keys`.
 *
 * All the aggregations which can be computed in a single pass are computed first by the same set
 * of kernels. Then using these results, compound aggregations that require multiple passes will be
 * computed on top of them. All results are stored into the in/out parameter `cache`.
 *
 * @tparam Equal Device row comparator type
 * @tparam Hash Device row hasher type
 *
 * @param keys Table whose rows act as the groupby keys
 * @param requests The set of columns to aggregate and the aggregations to perform
 * @param skip_rows_with_nulls Flag indicating whether to ignore nulls or not
 * @param d_row_equal Device row comparator
 * @param d_row_hash Device row hasher
 * @param cache Dense aggregation results
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table
 * @return Table of unique keys
 */
template <typename Equal, typename Hash>
std::unique_ptr<cudf::table> compute_groupby(table_view const& keys,
                                             std::span<aggregation_request const> requests,
                                             bool skip_rows_with_nulls,
                                             Equal const& d_row_equal,
                                             Hash const& d_row_hash,
                                             cudf::detail::result_cache* cache,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
