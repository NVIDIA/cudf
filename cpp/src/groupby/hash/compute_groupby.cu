/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_groupby.hpp"
#include "compute_single_pass_aggs.hpp"
#include "extract_single_pass_aggs.hpp"
#include "hash_compound_agg_finalizer.hpp"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/groupby/groupby_helper.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/array>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {

namespace {

// The result cache shares results across requests on the same values column. Normalize the
// extracted reductions with the same equality so an earlier compound request cannot choose the
// resource or nullability of a later requested result, or allocate a result that the cache drops.
auto extract_hash_groupby_aggs(std::span<aggregation_request const> requests,
                               cudf::detail::result_cache const& cache,
                               cuda::stream_ref stream,
                               bool expose_intermediates)
{
  using aggregation_set =
    std::unordered_set<std::pair<column_view, std::reference_wrapper<aggregation const>>,
                       cudf::detail::pair_column_aggregation_hash,
                       cudf::detail::pair_column_aggregation_equal_to>;
  aggregation_set requested;
  for (auto const& request : requests) {
    for (auto const& agg : request.aggregations) {
      requested.emplace(request.values, *agg);
    }
  }

  auto [values, kinds, aggs, is_intermediate, has_compound] =
    extract_single_pass_aggs(requests, stream, true);
  aggregation_set extracted;
  std::vector<column_view> unique_values;
  unique_values.reserve(aggs.size());
  for (std::size_t i = 0; i < aggs.size(); ++i) {
    if (cache.has_result(values.column(i), *aggs[i])) { continue; }
    auto const key =
      std::pair<column_view, std::reference_wrapper<aggregation const>>{values.column(i), *aggs[i]};
    if (!extracted.insert(key).second) { continue; }
    auto const output       = unique_values.size();
    kinds[output]           = kinds[i];
    is_intermediate[output] = !expose_intermediates && !requested.contains(key);
    if (output != i) { aggs[output] = std::move(aggs[i]); }
    unique_values.push_back(values.column(i));
  }
  kinds.resize(unique_values.size());
  aggs.resize(unique_values.size());
  is_intermediate.resize(unique_values.size());
  return std::tuple{table_view{unique_values},
                    std::move(kinds),
                    std::move(aggs),
                    std::move(is_intermediate),
                    has_compound};
}

}  // namespace

void compute_aggregations(std::span<aggregation_request const> requests,
                          groupby_helper& helper,
                          cudf::detail::result_cache& cache,
                          cuda::stream_ref stream,
                          cudf::memory_resources mr,
                          bool expose_intermediates)
{
  if (std::all_of(requests.begin(), requests.end(), [&](auto const& request) {
        return std::all_of(request.aggregations.begin(),
                           request.aggregations.end(),
                           [&](auto const& agg) { return cache.has_result(request.values, *agg); });
      })) {
    return;
  }

  auto const temp_mr = mr.get_temporary_mr();

  // Compute only missing single-pass results, preserving the extracted batching and fusion order.
  auto const [values, agg_kinds, aggs, is_agg_intermediate, has_compound_aggs] =
    extract_hash_groupby_aggs(requests, cache, stream, expose_intermediates);

  // Counts without null filtering come directly from the group offsets.
  auto const needs_reduction = [&] {
    for (size_type i = 0; i < values.num_columns(); ++i) {
      if (agg_kinds[i] != aggregation::COUNT_ALL &&
          (agg_kinds[i] != aggregation::COUNT_VALID || values.column(i).has_nulls())) {
        return true;
      }
    }
    return false;
  }();
  if (values.num_columns() != 0) {
    auto results = [&] {
      if (needs_reduction) {
        return compute_single_pass_aggs(
          values, agg_kinds, is_agg_intermediate, helper.reduction_groups(stream), stream, mr);
      }
      auto const grouped =
        grouped_rows{helper.unordered_grouped_order(stream),
                     helper.group_offsets(stream),
                     rmm::device_uvector<size_type>{0, stream, temp_mr},
                     rmm::device_uvector<size_type>{0, stream, temp_mr},
                     rmm::device_uvector<cuda::std::array<size_type, 2>>{0, stream, temp_mr},
                     rmm::device_uvector<size_type>{0, stream, temp_mr}};
      return compute_single_pass_aggs(values, agg_kinds, is_agg_intermediate, grouped, stream, mr);
    }();
    for (std::size_t i = 0; i < results.size(); ++i) {
      cache.add_result(values.column(i), *aggs[i], std::move(results[i]));
    }
  }

  if (has_compound_aggs) {
    for (auto const& request : requests) {
      auto const& agg_v = request.aggregations;
      auto const& col   = request.values;

      // The finalizers only combine the single-pass results with linear transformations such as
      // addition/multiplication (e.g. for variance/stddev); they do not aggregate further.
      auto const finalizer = hash_compound_agg_finalizer(col, &cache, nullptr, stream, mr);
      for (auto&& agg : agg_v) {
        if (cache.has_result(col, *agg)) { continue; }
        cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
      }
    }
  }
}

}  // namespace cudf::groupby::detail::hash
