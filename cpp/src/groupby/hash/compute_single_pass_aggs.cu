/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "single_pass_reductions.cuh"

#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

namespace cudf::groupby::detail::hash {
namespace single_pass {

/// A group is valid when any of its rows is valid.
std::pair<rmm::device_buffer, size_type> reduce_group_validity(reduction_context const& ctx,
                                                               cudf::memory_resources mr)
{
  rmm::device_uvector<bool> group_valid(ctx.num_groups, ctx.stream, mr.get_temporary_mr());
  reduce_groups(ctx.grouped,
                ctx.grouped.packed_rows,
                cuda::make_permutation_iterator(cudf::detail::make_validity_iterator(ctx.d_values),
                                                ctx.grouped.rows.begin()),
                group_valid.begin(),
                cuda::std::logical_or<bool>{},
                false,
                ctx.stream,
                mr.get_temporary_mr());
  return cudf::detail::valid_if(
    group_valid.begin(), group_valid.end(), cuda::std::identity{}, ctx.stream, mr);
}

void set_group_null_mask(column& result, reduction_context const& ctx, cudf::memory_resources mr)
{
  if (!ctx.nullable || ctx.num_groups == 0) { return; }
  auto [null_mask, null_count] = reduce_group_validity(ctx, mr);
  result.set_null_mask(std::move(null_mask), null_count);
}

std::unique_ptr<column> make_size_type_column(reduction_context const& ctx,
                                              cudf::memory_resources mr)
{
  return make_fixed_width_column(data_type{type_to_id<size_type>()},
                                 ctx.num_groups,
                                 mask_state::UNALLOCATED,
                                 ctx.stream,
                                 mr.get_output_mr());
}

std::unique_ptr<column> count_groups(reduction_context const& ctx,
                                     bool valid_only,
                                     cudf::memory_resources mr)
{
  auto result = make_size_type_column(ctx, mr);
  if (ctx.num_groups == 0) { return result; }

  if (valid_only && ctx.values.has_nulls()) {
    auto const valid_counts = cuda::transform_iterator{
      cuda::make_permutation_iterator(cudf::detail::make_validity_iterator(ctx.d_values),
                                      ctx.grouped.rows.begin()),
      [] __device__(bool valid) -> size_type { return static_cast<size_type>(valid); }};
    reduce_groups(ctx.grouped,
                  ctx.grouped.packed_rows,
                  valid_counts,
                  result->mutable_view().begin<size_type>(),
                  cuda::std::plus<size_type>{},
                  size_type{0},
                  ctx.stream,
                  mr.get_temporary_mr());
  } else {
    thrust::adjacent_difference(rmm::exec_policy_nosync(ctx.stream, mr.get_temporary_mr()),
                                ctx.grouped.offsets.begin() + 1,
                                ctx.grouped.offsets.end(),
                                result->mutable_view().begin<size_type>());
  }
  return result;
}

/// Calls `f.template operator()<K>()` for the reduction kind `kind`.
template <typename F>
auto dispatch_reduction_kind(aggregation::Kind kind, F&& f)
{
  switch (kind) {
    case aggregation::SUM: return f.template operator()<aggregation::SUM>();
    case aggregation::PRODUCT: return f.template operator()<aggregation::PRODUCT>();
    case aggregation::SUM_OF_SQUARES: return f.template operator()<aggregation::SUM_OF_SQUARES>();
    case aggregation::MIN: return f.template operator()<aggregation::MIN>();
    case aggregation::MAX: return f.template operator()<aggregation::MAX>();
    case aggregation::ARGMIN: return f.template operator()<aggregation::ARGMIN>();
    case aggregation::ARGMAX: return f.template operator()<aggregation::ARGMAX>();
    case aggregation::SUM_OVERFLOW: return f.template operator()<aggregation::SUM_OVERFLOW>();
    default: CUDF_FAIL("Unsupported hash groupby aggregation");
  }
}

struct compute_reduction_fn {
  reduction_context const& ctx;
  cudf::memory_resources mr;

  template <aggregation::Kind K>
  std::unique_ptr<column> operator()() const
  {
    return compute_reduction<K>(ctx, mr);
  }
};

template <aggregation::Kind K>
struct is_reduction_supported_fn {
  template <typename T>
  bool operator()() const
  {
    return is_reduction_supported<K, T>();
  }
};

struct is_reduction_kind_supported_fn {
  data_type values_type;

  template <aggregation::Kind K>
  bool operator()() const
  {
    return type_dispatcher(values_type, is_reduction_supported_fn<K>{});
  }
};

std::unique_ptr<column> compute_aggregation(aggregation::Kind kind,
                                            reduction_context const& ctx,
                                            cudf::memory_resources mr)
{
  switch (kind) {
    case aggregation::COUNT_VALID: return count_groups(ctx, true, mr);
    case aggregation::COUNT_ALL: return count_groups(ctx, false, mr);
    default: return dispatch_reduction_kind(kind, compute_reduction_fn{ctx, mr});
  }
}

}  // namespace single_pass

bool is_single_pass_agg_supported(data_type values_type, aggregation::Kind kind)
{
  // Values of STRUCT and LIST types are not aggregated by the hash groupby.
  if (cudf::is_nested(values_type)) { return false; }
  switch (kind) {
    case aggregation::COUNT_VALID:
    case aggregation::COUNT_ALL: return true;
    case aggregation::SUM:
    case aggregation::PRODUCT:
    case aggregation::SUM_OF_SQUARES:
    case aggregation::MIN:
    case aggregation::MAX:
    case aggregation::ARGMIN:
    case aggregation::ARGMAX:
    case aggregation::SUM_OVERFLOW:
      return single_pass::dispatch_reduction_kind(
        kind, single_pass::is_reduction_kind_supported_fn{values_type});
    default: return false;
  }
}

grouped_rows make_grouped_rows(device_span<size_type const> rows,
                               device_span<size_type const> offsets,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  auto const temp_mr    = mr.get_temporary_mr();
  auto const num_rows   = static_cast<size_type>(rows.size());
  auto const num_groups = static_cast<size_type>(offsets.size() - 1);
  grouped_rows grouped{rows,
                       offsets,
                       rmm::device_uvector<size_type>{0, stream, mr.get_output_mr()},
                       rmm::device_uvector<size_type>{0, stream, mr.get_output_mr()}};
  if (num_groups == 0) { return grouped; }

  // Small groups are packed several per block; every segment is then bounded by a shorter chunk
  // so that no thread or sub-warp is left walking a long group alone.
  auto const avg_rows   = num_rows / num_groups;
  auto const packed     = avg_rows < single_pass::min_avg_rows_per_segment;
  auto const chunk_rows = packed ? single_pass::packed_rows_per_chunk : single_pass::rows_per_chunk;
  grouped.packed_rows   = packed ? std::max<size_type>(avg_rows, 1) : 0;

  auto const policy       = rmm::exec_policy_nosync(stream, temp_mr);
  auto const chunk_counts = cudf::detail::make_counting_transform_iterator(
    0, [offsets = offsets.begin(), chunk_rows] __device__(size_type group) -> size_type {
      return cudf::util::div_rounding_up_safe(offsets[group + 1] - offsets[group], chunk_rows);
    });
  grouped.group_chunks.resize(offsets.size(), stream);
  grouped.group_chunks.set_element_to_zero_async(0, stream);
  thrust::inclusive_scan(
    policy, chunk_counts, chunk_counts + num_groups, grouped.group_chunks.begin() + 1);
  auto const num_chunks = grouped.group_chunks.back_element(stream);
  if (num_chunks == num_groups) {
    // No group spans several chunks, so the groups are the segments.
    grouped.group_chunks.resize(0, stream);
    grouped.group_chunks.shrink_to_fit(stream);
    return grouped;
  }

  grouped.chunk_offsets.resize(static_cast<std::size_t>(num_chunks) + 1, stream);
  thrust::tabulate(
    policy,
    grouped.chunk_offsets.begin(),
    grouped.chunk_offsets.end(),
    [offsets          = offsets.begin(),
     group_chunks     = grouped.group_chunks.begin(),
     group_chunks_end = grouped.group_chunks.end(),
     num_chunks,
     num_rows,
     chunk_rows] __device__(size_type chunk) -> size_type {
      if (chunk == num_chunks) { return num_rows; }
      auto const group = static_cast<size_type>(
        cuda::std::upper_bound(group_chunks, group_chunks_end, chunk) - group_chunks - 1);
      return offsets[group] + (chunk - group_chunks[group]) * chunk_rows;
    });
  return grouped;
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
    if (!single_pass::is_fusable_sum(agg_kinds[begin]) ||
        !is_single_pass_agg_supported(values_type, aggregation::SUM_OF_SQUARES)) {
      return begin + 1;
    }
    auto end = begin + 1;
    while (end < num_aggs && single_pass::is_fusable_sum(agg_kinds[end]) &&
           cudf::detail::is_shallow_equivalent(col, values.column(end)) &&
           std::find(agg_kinds.begin() + begin, agg_kinds.begin() + end, agg_kinds[end]) ==
             agg_kinds.begin() + end) {
      ++end;
    }
    return end;
  };

  std::vector<std::unique_ptr<column>> results;
  results.reserve(num_aggs);
  for (std::size_t i = 0; i < num_aggs;) {
    auto const& col  = values.column(i);
    auto const d_col = column_device_view::create(col, stream, mr.get_temporary_mr());
    auto const values_type =
      is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
    auto const kind = agg_kinds[i];
    // Counts are never null, and intermediate results skip the null mask to avoid the extra work.
    auto const nullable = !is_agg_intermediate[i] && kind != aggregation::COUNT_VALID &&
                          kind != aggregation::COUNT_ALL && col.has_nulls();
    auto const ctx = single_pass::reduction_context{
      col, *d_col, values_type, grouped, num_groups, nullable, stream};

    auto const end = fused_end(i, values_type);
    if (end > i + 1) {
      auto fused = single_pass::compute_fused_sums(
        ctx,
        host_span<aggregation::Kind const>{agg_kinds}.subspan(i, end - i),
        is_agg_intermediate.subspan(i, end - i),
        mr);
      std::move(fused.begin(), fused.end(), std::back_inserter(results));
    } else {
      auto const resources = is_agg_intermediate[i] ? cudf::memory_resources{mr.get_temporary_mr(),
                                                                             mr.get_temporary_mr()}
                                                    : mr;
      results.push_back(single_pass::compute_aggregation(kind, ctx, resources));
    }
    i = end;
  }
  return results;
}

}  // namespace cudf::groupby::detail::hash
