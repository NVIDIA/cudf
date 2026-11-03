/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @brief Computes the exclusive scan of a column.
 *
 * The null values are skipped for the operation, and if an input element at `i` is null, then the
 * output element at `i` will also be null.
 *
 * The identity value for the column type as per the aggregation type is used for the value of the
 * first element in the output column.
 *
 * Struct columns are allowed with aggregation types Min and Max.
 *
 * @throws cudf::logic_error if column data_type is not an arithmetic type or struct type but the
 *                           `agg` is not Min or Max.
 *
 * @param input The input column view for the scan.
 * @param agg Aggregation operator applied by the scan
 * @param null_handling Exclude null values when computing the result if null_policy::EXCLUDE.
 *                      Include nulls if null_policy::INCLUDE. Any operation with a null results in
 *                      a null.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned scalar's device memory.
 * @returns Column with scan results.
 */
std::unique_ptr<column> scan_exclusive(column_view const& input,
                                       scan_aggregation const& agg,
                                       null_policy null_handling,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @brief Computes the inclusive scan of a column.
 *
 * The null values are skipped for the operation, and if an input element at `i` is null, then the
 * output element at `i` will also be null.
 *
 * String and struct columns are allowed with aggregation types Min and Max.
 *
 * @throws cudf::logic_error if column data_type is not an arithmetic type or string/struct types
 *                           but the `agg` is not Min or Max.
 *
 * @param input The input column view for the scan.
 * @param agg Aggregation operator applied by the scan
 * @param null_handling Exclude null values when computing the result if null_policy::EXCLUDE.
 *                      Include nulls if null_policy::INCLUDE. Any operation with a null results in
 *                      a null.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned scalar's device memory.
 * @returns Column with scan results.
 */
std::unique_ptr<column> scan_inclusive(column_view const& input,
                                       scan_aggregation const& agg,
                                       null_policy null_handling,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @brief Generate row ranks for a column.
 *
 * @param order_by Input column to generate ranks for.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return rank values.
 */
std::unique_ptr<column> inclusive_rank_scan(column_view const& order_by,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @brief Generate row dense ranks for a column.
 *
 * @param order_by Input column to generate ranks for.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return rank values.
 */
CUDF_EXPORT
std::unique_ptr<column> inclusive_dense_rank_scan(column_view const& order_by,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr);

/**
 * @brief Generate row ONE_NORMALIZED percent ranks for a column.
 * Also, knowns as ANSI SQL PERCENT RANK.
 * Calculated by (rank - 1) / (count - 1).
 *
 * @param order_by Input column to generate ranks for.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return rank values.
 */
std::unique_ptr<column> inclusive_one_normalized_percent_rank_scan(
  column_view const& order_by, cuda::stream_ref stream, rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
