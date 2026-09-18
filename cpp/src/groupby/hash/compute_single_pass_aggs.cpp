/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_single_pass_aggs.hpp"

#include "single_pass_reductions.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {

std::unique_ptr<column> make_size_type_column(reduction_context const& ctx,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr)
{
  return make_fixed_width_column(data_type{type_to_id<size_type>()},
                                 ctx.num_groups,
                                 mask_state::UNALLOCATED,
                                 stream,
                                 mr.get_output_mr());
}

namespace {

/// Calls `f.template operator()<K>(args...)` for the reduction kind `kind`.
template <typename F, typename... Args>
auto dispatch_reduction_kind(aggregation::Kind kind, F&& f, Args&&... args)
{
  switch (kind) {
    case aggregation::SUM:
      return f.template operator()<aggregation::SUM>(std::forward<Args>(args)...);
    case aggregation::PRODUCT:
      return f.template operator()<aggregation::PRODUCT>(std::forward<Args>(args)...);
    case aggregation::SUM_OF_SQUARES:
      return f.template operator()<aggregation::SUM_OF_SQUARES>(std::forward<Args>(args)...);
    case aggregation::MIN:
      return f.template operator()<aggregation::MIN>(std::forward<Args>(args)...);
    case aggregation::MAX:
      return f.template operator()<aggregation::MAX>(std::forward<Args>(args)...);
    case aggregation::ARGMIN:
      return f.template operator()<aggregation::ARGMIN>(std::forward<Args>(args)...);
    case aggregation::ARGMAX:
      return f.template operator()<aggregation::ARGMAX>(std::forward<Args>(args)...);
    case aggregation::SUM_OVERFLOW:
      return f.template operator()<aggregation::SUM_OVERFLOW>(std::forward<Args>(args)...);
    default: CUDF_FAIL("Unsupported hash groupby aggregation");
  }
}

struct compute_reduction_fn {
  reduction_context const& ctx;

  template <aggregation::Kind K>
  std::unique_ptr<column> operator()(cuda::stream_ref stream, cudf::memory_resources mr) const
  {
    return compute_reduction<K>(ctx, stream, mr);
  }
};

struct compute_reductions_fn {
  host_span<reduction_context const> contexts;
  std::span<int8_t const> is_intermediate;

  template <aggregation::Kind K>
  std::vector<std::unique_ptr<column>> operator()(cuda::stream_ref stream,
                                                  cudf::memory_resources mr) const
  {
    if constexpr (K == aggregation::SUM || K == aggregation::SUM_OF_SQUARES ||
                  K == aggregation::PRODUCT || K == aggregation::MIN || K == aggregation::MAX) {
      return compute_reductions<K>(contexts, is_intermediate, stream, mr);
    } else {
      CUDF_FAIL("Unsupported batched hash groupby aggregation");
    }
  }
};

std::unique_ptr<column> compute_aggregation(aggregation::Kind kind,
                                            reduction_context const& ctx,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  switch (kind) {
    case aggregation::COUNT_VALID: return count_groups(ctx, true, stream, mr);
    case aggregation::COUNT_ALL: return count_groups(ctx, false, stream, mr);
    default: return dispatch_reduction_kind(kind, compute_reduction_fn{ctx}, stream, mr);
  }
}

}  // namespace

bool is_single_pass_agg_supported(data_type values_type, aggregation::Kind kind)
{
  if (cudf::is_nested(values_type)) { return false; }
  if (kind == aggregation::COUNT_VALID || kind == aggregation::COUNT_ALL) { return true; }
  if (values_type.id() == type_id::EMPTY) { return false; }
  return type_dispatcher(values_type,
                         [kind]<typename T>() { return is_reduction_supported<T>(kind); });
}

std::vector<std::unique_ptr<column>> compute_single_pass_aggs(
  table_view const& values,
  host_span<aggregation::Kind const> agg_kinds,
  std::span<int8_t const> is_agg_intermediate,
  grouped_rows const& grouped,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  CUDF_EXPECTS(values.num_columns() == static_cast<size_type>(agg_kinds.size()),
               "The number of values columns and aggregation kinds must be the same.");
  CUDF_EXPECTS(values.num_columns() == static_cast<size_type>(is_agg_intermediate.size()),
               "The number of values columns and intermediate flags must be the same.");

  auto const num_groups = static_cast<size_type>(grouped.offsets.size() - 1);
  auto const num_aggs   = agg_kinds.size();

  // Returns one past the last of the consecutive additive aggregations on the column of `begin`
  // that can be computed together with the aggregation at `begin`.
  auto const fused_end = [&](std::size_t begin, data_type values_type) {
    auto const& col = values.column(begin);
    if (!is_fusable_sum(agg_kinds[begin]) ||
        !is_single_pass_agg_supported(values_type, aggregation::SUM_OF_SQUARES)) {
      return begin + 1;
    }
    auto end = begin + 1;
    while (end < num_aggs && is_fusable_sum(agg_kinds[end]) &&
           cudf::detail::is_shallow_equivalent(col, values.column(end)) &&
           std::find(agg_kinds.begin() + begin, agg_kinds.begin() + end, agg_kinds[end]) ==
             agg_kinds.begin() + end) {
      ++end;
    }
    return end;
  };

  auto const minmax_sum_end = [&](std::size_t begin, data_type values_type) {
    if (!is_fusable_minmax_sum(agg_kinds[begin]) ||
        !is_single_pass_agg_supported(values_type, aggregation::SUM) ||
        !is_single_pass_agg_supported(values_type, aggregation::MIN) ||
        !is_single_pass_agg_supported(values_type, aggregation::MAX)) {
      return begin + 1;
    }
    auto end = begin + 1;
    while (end < num_aggs && is_fusable_minmax_sum(agg_kinds[end]) &&
           cudf::detail::is_shallow_equivalent(values.column(begin), values.column(end)) &&
           std::find(agg_kinds.begin() + begin, agg_kinds.begin() + end, agg_kinds[end]) ==
             agg_kinds.begin() + end) {
      ++end;
    }
    auto const sum =
      std::find(agg_kinds.begin() + begin, agg_kinds.begin() + end, aggregation::SUM);
    auto const sum_index = static_cast<std::size_t>(sum - agg_kinds.begin());
    // Do not consume a SUM that already participates in the existing additive fusion.
    if (sum == agg_kinds.begin() + end || fused_end(sum_index, values_type) > sum_index + 1) {
      return begin + 1;
    }
    return end;
  };

  // Keep each same-input fused run intact; otherwise batch adjacent compatible reductions.
  auto const batch_end = [&](std::size_t begin, data_type values_type, bool nullable) {
    auto const kind = agg_kinds[begin];
    if (kind != aggregation::SUM && kind != aggregation::SUM_OF_SQUARES &&
        kind != aggregation::PRODUCT && kind != aggregation::MIN && kind != aggregation::MAX) {
      return begin + 1;
    }
    auto end = begin + 1;
    while (end < num_aggs && agg_kinds[end] == kind) {
      auto const& col = values.column(end);
      auto const type =
        is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
      if (type != values_type || (!is_agg_intermediate[end] && col.has_nulls()) != nullable ||
          fused_end(end, type) > end + 1 || minmax_sum_end(end, type) > end + 1) {
        break;
      }
      ++end;
    }
    return end;
  };

  std::vector<std::unique_ptr<column>> results;
  results.reserve(num_aggs);
  for (std::size_t i = 0; i < num_aggs;) {
    auto const& col = values.column(i);
    auto d_col      = column_device_view::create(col, stream, mr.get_temporary_mr());
    auto const values_type =
      is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
    auto const kind = agg_kinds[i];
    // Counts are never null, and intermediate results skip the null mask to avoid the extra work.
    auto const nullable = !is_agg_intermediate[i] && kind != aggregation::COUNT_VALID &&
                          kind != aggregation::COUNT_ALL && col.has_nulls();
    auto const ctx = reduction_context{col, *d_col, values_type, grouped, num_groups, nullable};

    auto const sums_end    = fused_end(i, values_type);
    auto const extrema_end = minmax_sum_end(i, values_type);
    auto end               = std::max(sums_end, extrema_end);
    if (end > i + 1) {
      auto const compute_fused =
        extrema_end > sums_end ? compute_fused_minmax_sum : compute_fused_sums;
      auto fused = compute_fused(ctx,
                                 host_span<aggregation::Kind const>{agg_kinds}.subspan(i, end - i),
                                 is_agg_intermediate.subspan(i, end - i),
                                 stream,
                                 mr);
      std::move(fused.begin(), fused.end(), std::back_inserter(results));
    } else if (end = batch_end(i, values_type, nullable); end > i + 1) {
      std::vector<decltype(d_col)> device_views;
      std::vector<reduction_context> contexts;
      device_views.reserve(end - i);
      contexts.reserve(end - i);
      device_views.push_back(std::move(d_col));
      contexts.push_back(ctx);
      for (auto j = i + 1; j < end; ++j) {
        auto const& next = values.column(j);
        device_views.push_back(column_device_view::create(next, stream, mr.get_temporary_mr()));
        contexts.push_back(
          {next, *device_views.back(), values_type, grouped, num_groups, nullable});
      }
      auto batch = dispatch_reduction_kind(
        kind, compute_reductions_fn{contexts, is_agg_intermediate.subspan(i, end - i)}, stream, mr);
      std::move(batch.begin(), batch.end(), std::back_inserter(results));
    } else {
      auto const resources = is_agg_intermediate[i] ? cudf::memory_resources{mr.get_temporary_mr(),
                                                                             mr.get_temporary_mr()}
                                                    : mr;
      results.push_back(compute_aggregation(kind, ctx, stream, resources));
    }
    i = end;
  }
  return results;
}

}  // namespace cudf::groupby::detail::hash
