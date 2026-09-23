/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/common/m2_var_std.hpp"
#include "hash_compound_agg_finalizer.hpp"
#include "helpers.cuh"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/types.hpp>

#include <cuda/stream>

namespace cudf::groupby::detail::hash {

hash_compound_agg_finalizer::hash_compound_agg_finalizer(column_view const& col,
                                                         cudf::detail::result_cache* cache,
                                                         bitmask_type const* d_row_bitmask,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
  : col{col},
    input_type{is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type()},
    cache{cache},
    d_row_bitmask{d_row_bitmask},
    stream{stream},
    mr{mr}
{
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
  if (finalizer.input_type.id() == type_id::STRING) {
    auto transformed_agg = make_arg_agg();
    finalizer.cache->add_result(finalizer.col, agg, gather_argminmax(finalizer, *transformed_agg));
  }  // else: no-op, since this is only relevant for compound aggregations
  // TODO: support other nested types.
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

// Specialization for MEAN aggregation
template <>
void hash_compound_agg_finalizer::operator()<aggregation::MEAN>(aggregation const& agg) const
{
  if (cache->has_result(col, agg)) { return; }

  auto const sum_agg           = make_sum_aggregation();
  auto const count_agg         = make_count_aggregation();
  auto const sum_result        = cache->get_result(col, *sum_agg);
  auto const count_result      = cache->get_result(col, *count_agg);
  auto const sum_without_nulls = [&] {
    if (sum_result.null_count() == 0) { return sum_result; }
    return column_view{
      sum_result.type(), sum_result.size(), sum_result.head(), nullptr, 0, sum_result.offset()};
  }();

  // Perform division without any null masks, and generate the null mask for the result later.
  // This is because the null mask (if exists) is just needed to be copied from the sum result,
  // and copying is faster than running the `bitmask_and` kernel.
  auto result =
    cudf::detail::binary_operation(sum_without_nulls,
                                   count_result,
                                   binary_operator::DIV,
                                   cudf::detail::target_type(input_type, aggregation::MEAN),
                                   stream,
                                   mr);
  // SUM result only has nulls if it is an input aggregation, not intermediate-only aggregation.
  if (sum_result.has_nulls()) {
    result->set_null_mask(cudf::detail::copy_bitmask(sum_result, stream, mr),
                          sum_result.null_count());
  } else if (col.has_nulls()) {  // SUM aggregation is only intermediate result, thus it is
                                 // forced to be non-nullable
    auto [null_mask, null_count] = cudf::detail::valid_if(
      count_result.begin<size_type>(),
      count_result.end<size_type>(),
      [] __device__(size_type const count) -> bool { return count > 0; },
      stream,
      mr);
    if (null_count > 0) { result->set_null_mask(std::move(null_mask), null_count); }
  }
  cache->add_result(col, agg, std::move(result));
}

// Specialization for M2 aggregation
template <>
void hash_compound_agg_finalizer::operator()<aggregation::M2>(aggregation const& agg) const
{
  if (cache->has_result(col, agg)) { return; }

  auto const sum_sqr_agg    = make_sum_of_squares_aggregation();
  auto const sum_agg        = make_sum_aggregation();
  auto const count_agg      = make_count_aggregation();
  auto const sum_sqr_result = cache->get_result(col, *sum_sqr_agg);
  auto const sum_result     = cache->get_result(col, *sum_agg);
  auto const count_result   = cache->get_result(col, *count_agg);

  auto output = compute_m2(input_type, sum_sqr_result, sum_result, count_result, stream, mr);
  cache->add_result(col, agg, std::move(output));
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
  // Since M2 is a compound aggregation, we need to "finalize" it using aggregation finalizer.
  cudf::detail::aggregation_dispatcher(m2_agg->kind, finalizer, *m2_agg);
  auto const count_agg    = make_count_aggregation();
  auto const m2_result    = finalizer.cache->get_result(finalizer.col, *m2_agg);
  auto const count_result = finalizer.cache->get_result(finalizer.col, *count_agg);

  auto output = compute_fn(m2_result, count_result, ddof, finalizer.stream, finalizer.mr);
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
