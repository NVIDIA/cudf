/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf {
namespace groupby {
namespace detail {
/**
 * @brief Internal API to calculate groupwise cumulative sum
 *
 * @param values Grouped values to get sum of
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> sum_scan(column_view const& values,
                                 device_span<size_type const> group_labels,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr);

/**
 * @brief Internal API to calculate groupwise cumulative product
 *
 * Behaviour is undefined for signed integral types if any groupwise product overflows the type.
 *
 * @param values Grouped values to get product of
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> product_scan(column_view const& values,
                                     device_span<size_type const> group_labels,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr);

/**
 * @brief Internal API to calculate groupwise cumulative minimum value
 *
 * @param values Grouped values to get minimum from
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> min_scan(column_view const& values,
                                 device_span<size_type const> group_labels,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr);

/**
 * @brief Internal API to calculate groupwise cumulative maximum value
 *
 * @param values Grouped values to get maximum from
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> max_scan(column_view const& values,
                                 device_span<size_type const> group_labels,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr);

/**
 * @brief Internal API to calculate cumulative number of values in each group
 *
 * @param values Grouped values to get valid rows from
 * @param nulls Indicates whether nulls should be included in the count or not
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of type INT32 of count values
 */
std::unique_ptr<column> count_scan(column_view const& values,
                                   null_policy nulls,
                                   device_span<size_type const> group_labels,
                                   cuda::stream_ref stream,
                                   cudf::memory_resources mr);

/**
 * @brief Internal API to calculate groupwise min rank value
 *
 * @param grouped_values column or struct column that rows within a group are sorted by
 * @param value_order column of type INT32 that contains the order of the values in the
 * grouped_values column
 * @param group_labels ID of group that the corresponding value belongs to
 * @param group_offsets group index offsets with group ID indices
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of type size_type of rank values
 */
std::unique_ptr<column> min_rank_scan(column_view const& grouped_values,
                                      column_view const& value_order,
                                      device_span<size_type const> group_labels,
                                      device_span<size_type const> group_offsets,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr);

/**
 * @brief Internal API to calculate groupwise max rank value
 *
 * @details  @copydetails min_rank_scan(column_view const& grouped_values,
 *                                      column_view const& value_order,
 *                                      device_span<size_type const> group_labels,
 *                                      device_span<size_type const> group_offsets,
 *                                      cuda::stream_ref stream,
 *                                      cudf::memory_resources mr)
 */
std::unique_ptr<column> max_rank_scan(column_view const& grouped_values,
                                      column_view const& value_order,
                                      device_span<size_type const> group_labels,
                                      device_span<size_type const> group_offsets,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr);

/**
 * @brief Internal API to calculate groupwise first rank value
 *
 * @details  @copydetails min_rank_scan(column_view const& grouped_values,
 *                                      column_view const& value_order,
 *                                      device_span<size_type const> group_labels,
 *                                      device_span<size_type const> group_offsets,
 *                                      cuda::stream_ref stream,
 *                                      cudf::memory_resources mr)
 */
std::unique_ptr<column> first_rank_scan(column_view const& grouped_values,
                                        column_view const& value_order,
                                        device_span<size_type const> group_labels,
                                        device_span<size_type const> group_offsets,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr);

/**
 * @brief Internal API to calculate groupwise average rank value
 *
 * @details  @copydetails min_rank_scan(column_view const& grouped_values,
 *                                      column_view const& value_order,
 *                                      device_span<size_type const> group_labels,
 *                                      device_span<size_type const> group_offsets,
 *                                      cuda::stream_ref stream,
 *                                      cudf::memory_resources mr)
 */
std::unique_ptr<column> average_rank_scan(column_view const& grouped_values,
                                          column_view const& value_order,
                                          device_span<size_type const> group_labels,
                                          device_span<size_type const> group_offsets,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr);

/**
 * @brief Internal API to calculate groupwise dense rank value
 *
 * @param grouped_values column or struct column that rows within a group are sorted by
 * @param value_order Order of values within groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param group_offsets group index offsets with group ID indices
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of type size_type of dense rank values
 */
std::unique_ptr<column> dense_rank_scan(column_view const& grouped_values,
                                        column_view const& value_order,
                                        device_span<size_type const> group_labels,
                                        device_span<size_type const> group_offsets,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr);

/**
 * @brief Convert groupwise rank to groupwise percentage rank
 *
 * @param method rank method
 * @param percentage enum to denote the type of conversion ranks to percentage in range (0,1]
 * @param rank Groupwise rank column
 * @param count Groupwise count column
 * @param group_labels ID of group that the corresponding value belongs to
 * @param group_offsets group index offsets with group ID indices
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of type double of rank values

 */
std::unique_ptr<column> group_rank_to_percentage(rank_method const method,
                                                 rank_percentage const percentage,
                                                 column_view const& rank,
                                                 column_view const& count,
                                                 device_span<size_type const> group_labels,
                                                 device_span<size_type const> group_offsets,
                                                 cuda::stream_ref stream,
                                                 cudf::memory_resources mr);

/**
 * @brief Scatter groupwise ranks from grouped order back to the input order
 *
 * @param rank Groupwise rank column of type INT32 or FLOAT64 without nulls
 * @param scatter_map Input row of each rank
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of rank values in input order
 */
std::unique_ptr<column> scatter_rank(column_view const& rank,
                                     column_view const& scatter_map,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
