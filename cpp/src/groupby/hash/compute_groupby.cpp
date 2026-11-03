/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_groupby.hpp"

#include "compute_single_pass_aggs.hpp"
#include "extract_single_pass_aggs.hpp"
#include "hash_compound_agg_finalizer.hpp"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/groupby/groupby_helper.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/buffer>
#include <cuda/std/array>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <ranges>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {

namespace {

// The result cache shares results across requests on the same values column. Normalize the
// extracted reductions with the same equality so an earlier compound request cannot choose the
// resource or nullability of a later requested result, or allocate a result that the cache drops.
// For the same reason, skip reductions that an earlier call cached, such as the SUM and
// COUNT_VALID of the STD that CORRELATION computes before its MEAN. A host UDF may read any cached
// result, so `expose_intermediates` computes every extracted reduction like a requested result.
auto extract_hash_groupby_aggs(std::span<aggregation_request const> requests,
                               cudf::detail::result_cache const& cache,
                               bool expose_intermediates,
                               cuda::stream_ref stream)
{
  auto [values, kinds, aggs, is_intermediate, has_compound] =
    extract_single_pass_aggs(requests, stream);
  auto const is_cached = [&](std::size_t i) {
    return cache.has_result(values.column(i), *aggs[i]);
  };
  if (requests.size() <= 1 && !expose_intermediates &&
      std::ranges::none_of(std::views::iota(std::size_t{0}, aggs.size()), is_cached)) {
    return std::tuple{
      values, std::move(kinds), std::move(aggs), std::move(is_intermediate), has_compound};
  }

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

  aggregation_set extracted;
  std::vector<column_view> distinct_values;
  distinct_values.reserve(aggs.size());
  for (std::size_t i = 0; i < aggs.size(); ++i) {
    auto const key =
      std::pair<column_view, std::reference_wrapper<aggregation const>>{values.column(i), *aggs[i]};
    if (is_cached(i) || !extracted.insert(key).second) { continue; }
    auto const output       = distinct_values.size();
    kinds[output]           = kinds[i];
    is_intermediate[output] = !expose_intermediates && !requested.contains(key);
    if (output != i) { aggs[output] = std::move(aggs[i]); }
    distinct_values.push_back(values.column(i));
  }
  kinds.resize(distinct_values.size());
  aggs.resize(distinct_values.size());
  is_intermediate.resize(distinct_values.size());
  return std::tuple{table_view{distinct_values},
                    std::move(kinds),
                    std::move(aggs),
                    std::move(is_intermediate),
                    has_compound};
}

}  // namespace

void compute_groupby(std::span<aggregation_request const> requests,
                     groupby_helper& helper,
                     cudf::detail::result_cache* cache,
                     bool expose_intermediates,
                     cuda::stream_ref stream,
                     cudf::memory_resources mr)
{
  // In case of no requests, only the distinct keys are needed and the helper builds them alone.
  if (requests.empty()) { return; }

  auto const temp_mr             = mr.get_temporary_mr();
  auto const temporary_resources = mr.as_temporary();

  // Determine which grouping outputs the requested aggregations need.
  auto const [values, agg_kinds, aggs, is_agg_intermediate, has_compound_aggs] =
    extract_hash_groupby_aggs(requests, *cache, expose_intermediates, stream);

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
  // The helper caches the reduction schedule over its grouped rows for later callers.
  std::vector<std::unique_ptr<column>> results;
  if (needs_reduction) {
    results = compute_single_pass_aggs(
      values, agg_kinds, is_agg_intermediate, helper.reduction_groups(stream, mr), stream, mr);
  } else {
    auto const grouped =
      grouped_rows{device_span<size_type const>{},
                   helper.group_offsets(stream, mr),
                   cuda::device_buffer<size_type>{stream, temp_mr},
                   cuda::device_buffer<size_type>{stream, temp_mr},
                   cuda::device_buffer<cuda::std::array<size_type, 2>>{stream, temp_mr},
                   cuda::device_buffer<size_type>{stream, temp_mr}};
    results = compute_single_pass_aggs(values, agg_kinds, is_agg_intermediate, grouped, stream, mr);
  }
  for (std::size_t i = 0; i < results.size(); ++i) {
    cache->add_result(values.column(i), *aggs[i], std::move(results[i]));
  }

  if (has_compound_aggs) {
    // Requested M2 results must be cached on the output resource before VARIANCE or STD asks
    // for an intermediate M2, regardless of the order of requests on a shared values column.
    for (auto const& request : requests) {
      auto const finalizer = hash_compound_agg_finalizer(request.values, cache, stream, mr);
      for (auto const& agg : request.aggregations) {
        if (agg->kind == aggregation::M2) {
          cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
        }
      }
    }
    for (auto const& request : requests) {
      auto const& agg_v = request.aggregations;
      auto const& col   = request.values;

      // The finalizers only combine the single-pass results with linear transformations such as
      // addition/multiplication (e.g. for variance/stddev); they do not aggregate further.
      auto const finalizer = hash_compound_agg_finalizer(col, cache, stream, mr);
      for (auto&& agg : agg_v) {
        if (agg->kind == aggregation::VARIANCE || agg->kind == aggregation::STD) {
          // Explicit M2 outputs were finalized above. Any missing M2 is only an intermediate
          // for this ordinary groupby; the shared finalizer also serves streaming groupby.
          auto const m2_agg = make_m2_aggregation();
          auto const m2_finalizer =
            hash_compound_agg_finalizer(col, cache, stream, temporary_resources);
          cudf::detail::aggregation_dispatcher(m2_agg->kind, m2_finalizer, *m2_agg);
        }
        cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
      }
    }
  }
}

}  // namespace cudf::groupby::detail::hash
