/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace cudf::groupby::detail::hash {

/**
 * @brief Create the table containing columns for storing aggregation results.
 *
 * For the aggregations that will be used only as temporary, intermediate result for
 * computing other aggregations, we just create the results columns for them as non-nullable to
 * avoid the extra nullmask update and null count computation overhead.
 *
 * @param output_size Number of rows in the output table
 * @param values The values columns to be aggregated
 * @param agg_kinds The aggregation kinds corresponding to each input column
 * @param is_agg_intermediate A binary values vector indicating if the corresponding aggregation
 *        will be used only as temporary, intermediate result for computing other aggregations
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return The table containing columns for storing aggregation results
 */
std::unique_ptr<table> create_results_table(size_type output_size,
                                            table_view const& values,
                                            host_span<aggregation::Kind const> agg_kinds,
                                            std::span<int8_t const> is_agg_intermediate,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @brief Perform some final computation for the aggregation results such as null count and move
 * the result columns into a `result_cache` object.
 *
 * @param values The values columns
 * @param aggregations The aggregation to compute corresponding to each values column
 * @param agg_results The table containing columns storing aggregation results
 * @param cache The cache object to store the extracted aggregation results
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void finalize_output(table_view const& values,
                     std::vector<std::unique_ptr<aggregation>> const& aggregations,
                     std::unique_ptr<table>& agg_results,
                     cudf::detail::result_cache* cache,
                     cuda::stream_ref stream);

}  // namespace cudf::groupby::detail::hash
