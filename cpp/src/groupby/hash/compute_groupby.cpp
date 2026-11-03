/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_groupby.hpp"

#include "compute_single_pass_aggs.hpp"
#include "extract_single_pass_aggs.hpp"
#include "group_keys.hpp"
#include "hash_compound_agg_finalizer.hpp"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/buffer>
#include <cuda/std/array>

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
                               cuda::stream_ref stream)
{
  if (requests.size() <= 1) { return extract_single_pass_aggs(requests, stream); }

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
    extract_single_pass_aggs(requests, stream);
  aggregation_set extracted;
  std::vector<column_view> unique_values;
  unique_values.reserve(aggs.size());
  for (std::size_t i = 0; i < aggs.size(); ++i) {
    auto const key =
      std::pair<column_view, std::reference_wrapper<aggregation const>>{values.column(i), *aggs[i]};
    if (!extracted.insert(key).second) { continue; }
    auto const output       = unique_values.size();
    kinds[output]           = kinds[i];
    is_intermediate[output] = !requested.contains(key);
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

std::unique_ptr<table> compute_groupby(table_view const& keys,
                                       std::span<aggregation_request const> requests,
                                       null_policy include_null_keys,
                                       cudf::detail::result_cache* cache,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  auto const temp_mr             = mr.get_temporary_mr();
  auto const temporary_resources = mr.as_temporary();

  // Determine which grouping outputs the requested aggregations need.
  auto const [values, agg_kinds, aggs, is_agg_intermediate, has_compound_aggs] =
    extract_hash_groupby_aggs(requests, stream);

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
  auto const groups = group_keys(
    keys, include_null_keys, !requests.empty(), needs_reduction, stream, temporary_resources);

  auto const gather_keys = [&] {
    return cudf::detail::gather(keys,
                                groups.key_indices,
                                out_of_bounds_policy::DONT_CHECK,
                                cudf::negative_index_policy::NOT_ALLOWED,
                                stream,
                                mr);
  };

  // In case of no requests, we still need to generate a set of unique keys.
  if (requests.empty()) { return gather_keys(); }

  auto const rows = groups.grouped_rows ? device_span<size_type const>{groups.grouped_rows->data(),
                                                                       groups.grouped_rows->size()}
                                        : device_span<size_type const>{};
  auto const offsets =
    device_span<size_type const>{groups.group_offsets->data(), groups.group_offsets->size()};
  auto const grouped =
    needs_reduction
      ? make_grouped_rows(rows, offsets, stream, temporary_resources)
      : grouped_rows{rows,
                     offsets,
                     cuda::device_buffer<size_type>{stream, temp_mr},
                     cuda::device_buffer<size_type>{stream, temp_mr},
                     cuda::device_buffer<cuda::std::array<size_type, 2>>{stream, temp_mr},
                     cuda::device_buffer<size_type>{stream, temp_mr}};
  auto results =
    compute_single_pass_aggs(values, agg_kinds, is_agg_intermediate, grouped, stream, mr);
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

  return gather_keys();
}

}  // namespace cudf::groupby::detail::hash
