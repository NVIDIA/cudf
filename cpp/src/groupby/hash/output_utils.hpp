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

#include <rmm/device_uvector.hpp>

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
 * @brief Return an array containing indices of (unique) keys in `key_set`.
 *
 * @tparam SetType Type of the key hash set
 *
 * @param key_set Key hash set
 * @param num_total_keys Number of total keys
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned array
 * @return An array containing indices of unique keys retrieved from `key_set`
 */
template <typename SetType>
rmm::device_uvector<size_type> extract_populated_keys(SetType const& key_set,
                                                      size_type num_total_keys,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr);

/**
 * @brief Compute and return a mapping array that maps from the original input keys to their
 * positions in the input array which contains indices of the unique keys.
 *
 * Note that the output mapping array only covers the keys with indices existing in the input array,
 * leaving other keys with uninitialized mapping values.
 *
 * @param num_total_keys Number of total keys
 * @param unique_key_indices Array containing indices of the unique keys
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned array
 * @return An array mapping from the original input keys to their positions in the input array
 */
rmm::device_uvector<size_type> compute_key_transform_map(
  size_type num_total_keys,
  device_span<size_type const> unique_key_indices,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Transform from row indices of the keys in the input keys table into indices of these keys
 * in the output unique keys table.
 *
 * Note that the positions (indices) of all output unique keys must be covered in the array
 * `transform_map`. This is guaranteed as it was generated in `extract_populated_keys` function.
 *
 * @param input The indices of the keys to transform
 * @param transform_map The mapping array from the input keys table to the output unique keys table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned array
 * @return A device vector mapping each input row to its output row index
 */
rmm::device_uvector<size_type> compute_target_indices(device_span<size_type const> input,
                                                      device_span<size_type const> transform_map,
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
