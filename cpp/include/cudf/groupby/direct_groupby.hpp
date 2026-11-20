/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace groupby {

/**
 * @addtogroup aggregation_groupby
 * @{
 * @file
 * @brief Direct groupby aggregation APIs for dense integer keys
 */

/**
 * @brief Performs grouped aggregations on the specified values using keys whose values directly
 * determine the group slot
 *
 * The keys are treated as a perfect hash of the groups: each key value in `[0, capacity)`
 * addresses its own aggregation slot. No hashing or key comparison is performed. Typical sources of
 * such keys are `cudf::key_remapping`, dictionary indices, or dense integer primary keys.
 *
 * The values to aggregate and the aggregations to perform are specified in `aggregation_request`s
 * exactly as in `groupby::aggregate`. For each request, an `aggregation_result` is returned holding
 * one column per requested aggregation in the order specified in the request.
 *
 * The returned key column contains each distinct key value that occurs in `keys`, in ascending
 * order. Element `i` across all aggregation results belongs to the group at row `i` of the key
 * column. A group whose values are all null is retained with a null result, matching
 * `groupby::aggregate`.
 *
 * Supported aggregations are SUM, MIN, MAX, COUNT_VALID, COUNT_ALL and MEAN on fixed-width values.
 *
 * @note Behavior is undefined if any key value is not less than `capacity`.
 * @note Temporary device memory proportional to `capacity` is allocated for each result column
 * regardless of the number of distinct keys.
 *
 * @throw cudf::data_type_error if `keys` is not of type UINT32
 * @throw std::invalid_argument if `keys` contains nulls
 * @throw std::invalid_argument if `capacity` exceeds the maximum value of `size_type`
 * @throw std::invalid_argument if any request's values size differs from the keys size
 * @throw std::invalid_argument if any aggregation is not supported
 *
 * @param keys The key column containing pre-hashed keys in `[0, capacity)`
 * @param requests The set of columns to aggregate and the aggregations to perform
 * @param capacity The number of aggregation slots, one per possible key value
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 *
 * @return Pair containing the column of distinct keys and a vector of aggregation_results for each
 * request in the same order as specified in `requests`
 */
[[nodiscard]] std::pair<std::unique_ptr<column>, std::vector<aggregation_result>> direct_aggregate(
  column_view const& keys,
  std::span<aggregation_request const> requests,
  std::size_t capacity,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace groupby
}  // namespace CUDF_EXPORT cudf
