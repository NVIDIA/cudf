/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "binaryop/compiled/binary_ops.hpp"
#include "binaryop/compiled/operation.cuh"
#include "groupby/common/m2_var_std.hpp"
#include "hash_compound_agg_finalizer.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/stream>

#include <algorithm>
#include <ranges>

namespace cudf::groupby::detail::hash {

hash_compound_agg_finalizer::hash_compound_agg_finalizer(column_view const& col,
                                                         cudf::detail::result_cache* cache,
                                                         cuda::stream_ref stream,
                                                         cudf::memory_resources mr)
  : col{col},
    input_type{is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type()},
    cache{cache},
    stream{stream},
    mr{mr}
{
}

void finalize_means(table_view const& values,
                    host_span<aggregation::Kind const> kinds,
                    mutable_table_view const& results,
                    cudf::detail::result_cache const* cache,
                    cuda::stream_ref stream)
{
  auto const columns   = std::views::iota(size_type{0}, results.num_columns());
  auto const count_agg = make_count_aggregation();
  for (auto const i : columns) {
    auto const mean = results.column(i);
    if (kinds[i] != aggregation::MEAN || mean.is_empty()) { continue; }
    auto const count = std::ranges::find_if(columns, [&](size_type j) {
      return kinds[j] == aggregation::COUNT_VALID &&
             cudf::detail::is_shallow_equivalent(values.column(i), values.column(j));
    });
    // An earlier call may have cached the count, e.g. the STD that CORRELATION computes first,
    // in which case the extraction skipped it.
    auto const has_cached_count =
      cache != nullptr && cache->has_result(values.column(i), *count_agg);
    CUDF_EXPECTS(count != columns.end() || has_cached_count,
                 "MEAN requires the valid count of its values.");
    auto const counts = count != columns.end() ? column_view{results.column(*count)}
                                               : cache->get_result(values.column(i), *count_agg);
    // The sums already carry the MEAN validity, so divide only their values in place.
    auto const sums = column_view{mean.type(), mean.size(), mean.head(), nullptr, 0, mean.offset()};
    auto out =
      mutable_column_view{mean.type(), mean.size(), mean.head(), nullptr, 0, mean.offset()};
    cudf::binops::compiled::apply_binary_op<cudf::binops::compiled::ops::Div>(
      out, sums, counts, false, false, stream);
  }
}

// Enables conversion of ARGMIN/ARGMAX into MIN/MAX
std::unique_ptr<column> gather_argminmax(hash_compound_agg_finalizer const& finalizer,
                                         aggregation const& agg)
{
  auto const& col = finalizer.col;
  auto arg_result = finalizer.cache->get_result(col, agg);
  // We make a view of ARG(MIN/MAX) result without a null mask and gather
  // using this map. The values in data buffer of ARG(MIN/MAX) result
  // corresponding to null values was initialized to ARG(MIN/MAX)_SENTINEL
  // which is an out of bounds index value (-1) and causes the gathered
  // value to be null.
  column_view null_removed_map(data_type(type_to_id<size_type>()),
                               arg_result.size(),
                               static_cast<void const*>(arg_result.data<size_type>()),
                               nullptr,
                               0);
  auto result = cudf::detail::gather(
    table_view{{col}},
    null_removed_map,
    col.nullable() ? cudf::out_of_bounds_policy::NULLIFY : cudf::out_of_bounds_policy::DONT_CHECK,
    cudf::negative_index_policy::NOT_ALLOWED,
    finalizer.stream,
    finalizer.mr);
  return std::move(result->release()[0]);
}

// Helper for MIN/MAX finalization - shared logic for compound types (e.g., strings)
template <typename MakeArgAggFn>
void finalize_minmax_for_compound_types(hash_compound_agg_finalizer const& finalizer,
                                        aggregation const& agg,
                                        MakeArgAggFn make_arg_agg)
{
  if (finalizer.cache->has_result(finalizer.col, agg)) { return; }
  if (finalizer.input_type.id() == type_id::STRING || cudf::is_nested(finalizer.input_type)) {
    auto transformed_agg = make_arg_agg();
    finalizer.cache->add_result(finalizer.col, agg, gather_argminmax(finalizer, *transformed_agg));
  }  // else: no-op, since this is only relevant for compound aggregations
}

// Specialization for MIN aggregation
template <>
void hash_compound_agg_finalizer::operator()<aggregation::MIN>(aggregation const& agg) const
{
  finalize_minmax_for_compound_types(*this, agg, make_argmin_aggregation<>);
}

// Specialization for MAX aggregation
template <>
void hash_compound_agg_finalizer::operator()<aggregation::MAX>(aggregation const& agg) const
{
  finalize_minmax_for_compound_types(*this, agg, make_argmax_aggregation<>);
}

// Helper for VARIANCE/STD finalization - shared logic for M2-based computations
template <typename ComputeFn>
void finalize_var_std(hash_compound_agg_finalizer const& finalizer,
                      aggregation const& agg,
                      size_type ddof,
                      ComputeFn compute_fn)
{
  if (finalizer.cache->has_result(finalizer.col, agg)) { return; }

  auto const m2_agg = make_m2_aggregation();
  // The producer has already cached M2 and its matching valid count.
  auto const count_agg    = make_count_aggregation();
  auto const m2_result    = finalizer.cache->get_result(finalizer.col, *m2_agg);
  auto const count_result = finalizer.cache->get_result(finalizer.col, *count_agg);

  auto output =
    compute_fn(m2_result, count_result, ddof, finalizer.stream, finalizer.mr.get_output_mr());
  finalizer.cache->add_result(finalizer.col, agg, std::move(output));
}

// Specialization for VARIANCE aggregation
template <>
void hash_compound_agg_finalizer::operator()<aggregation::VARIANCE>(aggregation const& agg) const
{
  auto const& var_agg = dynamic_cast<cudf::detail::var_aggregation const&>(agg);
  finalize_var_std(*this, agg, var_agg._ddof, compute_variance);
}

// Specialization for STD aggregation
template <>
void hash_compound_agg_finalizer::operator()<aggregation::STD>(aggregation const& agg) const
{
  auto const& std_agg = dynamic_cast<cudf::detail::std_aggregation const&>(agg);
  finalize_var_std(*this, agg, std_agg._ddof, compute_std);
}

}  // namespace cudf::groupby::detail::hash
