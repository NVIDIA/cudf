/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace cudf::groupby::detail {

template <typename RequestType>
inline std::vector<aggregation_result> extract_results(std::span<RequestType const> requests,
                                                       cudf::detail::result_cache& cache,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  std::vector<aggregation_result> results(requests.size());
  std::unordered_map<std::pair<column_view, std::reference_wrapper<aggregation const>>,
                     column_view,
                     cudf::detail::pair_column_aggregation_hash,
                     cudf::detail::pair_column_aggregation_equal_to>
    repeated_result;
  for (size_t i = 0; i < requests.size(); i++) {
    for (auto&& agg : requests[i].aggregations) {
      if (cache.has_result(requests[i].values, *agg)) {
        results[i].results.emplace_back(cache.release_result(requests[i].values, *agg));
        repeated_result[{requests[i].values, *agg}] = results[i].results.back()->view();
      } else {
        auto it = repeated_result.find({requests[i].values, *agg});
        if (it != repeated_result.end()) {
          results[i].results.emplace_back(std::make_unique<column>(it->second, stream, mr));
        } else {
          CUDF_FAIL("Cannot extract result from the cache");
        }
      }
    }
  }
  return results;
}

/**
 * @brief Compute a combined null bitmask for multi-column keys.
 *
 * @return Pair of {buffer, raw_pointer} where pointer is null if no nulls exist.
 */
std::pair<rmm::device_buffer, bitmask_type const*> compute_row_bitmask(
  table_view const& keys, rmm::cuda_stream_view stream);

/// Whether the given aggregation kind is supported by hash-based groupby.
constexpr bool is_hash_aggregation(aggregation::Kind k)
{
  switch (k) {
    case aggregation::SUM:
    case aggregation::SUM_OVERFLOW:
    case aggregation::SUM_OF_SQUARES:
    case aggregation::PRODUCT:
    case aggregation::MIN:
    case aggregation::MAX:
    case aggregation::COUNT_VALID:
    case aggregation::COUNT_ALL:
    case aggregation::ARGMIN:
    case aggregation::ARGMAX:
    case aggregation::MEAN:
    case aggregation::M2:
    case aggregation::STD:
    case aggregation::VARIANCE: return true;
    default: return false;
  }
}

}  // namespace cudf::groupby::detail
