/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/common/utils.hpp"
#include "groupby/sort/functors.hpp"
#include "groupby/sort/group_reductions.hpp"
#include "groupby/sort/group_scan.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace groupby {
namespace detail {
/**
 * @brief Functor to dispatch aggregation with
 *
 * This functor is to be used with `aggregation_dispatcher` to compute the
 * appropriate aggregation. If the values on which to run the aggregation are
 * unchanged, then this functor should be re-used. This is because it stores
 * memoised sorted and/or grouped values and re-using will save on computation
 * of these values.
 */
struct scan_result_functor final : store_result_functor {
  using store_result_functor::store_result_functor;
  template <aggregation::Kind k>
  void operator()(aggregation const& agg)
  {
    CUDF_FAIL("Unsupported groupby scan aggregation");
  }

 private:
  column_view get_grouped_values()
  {
    // early exit if presorted
    if (is_presorted()) { return values; }

    // TODO (dm): After implementing single pass multi-agg, explore making a
    //            cache of all grouped value columns rather than one at a time
    if (grouped_values)
      return grouped_values->view();
    else
      return (grouped_values = helper.grouped_values(values, stream, mr))->view();
  };
};

template <>
void scan_result_functor::operator()<aggregation::SUM>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  cache.add_result(
    values,
    agg,
    detail::sum_scan(
      get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::PRODUCT>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  cache.add_result(
    values,
    agg,
    detail::product_scan(
      get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::MIN>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  cache.add_result(
    values,
    agg,
    detail::min_scan(
      get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::MAX>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  cache.add_result(
    values,
    agg,
    detail::max_scan(
      get_grouped_values(), helper.num_groups(stream), helper.group_labels(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::COUNT_ALL>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  cache.add_result(
    values,
    agg,
    detail::count_scan(values, null_policy::INCLUDE, helper.group_labels(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::COUNT_VALID>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  cache.add_result(
    values,
    agg,
    detail::count_scan(
      get_grouped_values(), null_policy::EXCLUDE, helper.group_labels(stream), stream, mr));
}

template <>
void scan_result_functor::operator()<aggregation::RANK>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(values),
               "Unsupported list type in grouped rank scan.");
  auto const& rank_agg         = dynamic_cast<cudf::detail::rank_aggregation const&>(agg);
  auto const& group_labels     = helper.group_labels(stream);
  auto const group_labels_view = column_view(cudf::device_span<size_type const>(group_labels));
  auto const gather_map        = [&]() {
    if (is_presorted()) {  // assumes both keys and values are sorted, Spark does this.
      return cudf::detail::sequence(
        group_labels.size(),
        *cudf::make_fixed_width_scalar(
          size_type{0}, stream, cudf::get_current_device_resource_ref()),
        stream,
        cudf::get_current_device_resource_ref());
    } else {
      auto sort_order = (rank_agg._method == rank_method::FIRST ? cudf::detail::stable_sorted_order
                                                                       : cudf::detail::sorted_order);
      return sort_order(table_view({group_labels_view, get_grouped_values()}),
                               {order::ASCENDING, rank_agg._column_order},
                               {null_order::AFTER, rank_agg._null_precedence},
                        stream,
                        cudf::get_current_device_resource_ref());
    }
  }();

  auto rank_scan = [&]() {
    switch (rank_agg._method) {
      case rank_method::FIRST: return detail::first_rank_scan;
      case rank_method::AVERAGE: return detail::average_rank_scan;
      case rank_method::DENSE: return detail::dense_rank_scan;
      case rank_method::MIN: return detail::min_rank_scan;
      case rank_method::MAX: return detail::max_rank_scan;
      default: CUDF_FAIL("Unsupported rank method in groupby scan");
    }
  }();
  auto result = rank_scan(get_grouped_values(),
                          *gather_map,
                          helper.group_labels(stream),
                          helper.group_offsets(stream),
                          stream,
                          cudf::get_current_device_resource_ref());
  if (rank_agg._percentage != rank_percentage::NONE) {
    auto count = get_grouped_values().nullable() and rank_agg._null_handling == null_policy::EXCLUDE
                   ? detail::group_count_valid(get_grouped_values(),
                                               helper.group_labels(stream),
                                               helper.num_groups(stream),
                                               stream,
                                               cudf::get_current_device_resource_ref())
                   : detail::group_count_all(helper.group_offsets(stream),
                                             helper.num_groups(stream),
                                             stream,
                                             cudf::get_current_device_resource_ref());
    result     = detail::group_rank_to_percentage(rank_agg._method,
                                              rank_agg._percentage,
                                              *result,
                                              *count,
                                              helper.group_labels(stream),
                                              helper.group_offsets(stream),
                                              stream,
                                              mr);
  }
  result = std::move(
    cudf::detail::scatter(table_view{{*result}}, *gather_map, table_view{{*result}}, stream, mr)
      ->release()[0]);
  if (rank_agg._null_handling == null_policy::EXCLUDE) {
    auto const values = get_grouped_values();
    result->set_null_mask(cudf::detail::copy_bitmask(values, stream, mr), values.null_count());
  }
  cache.add_result(values, agg, std::move(result));
}
}  // namespace detail

// Sort-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby::sort_scan(
  std::span<scan_request const> requests,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // We're going to start by creating a cache of results so that aggs that
  // depend on other aggs will not have to be recalculated. e.g. mean depends on
  // sum and count. std depends on mean and count
  cudf::detail::result_cache cache(requests.size());

  for (auto const& request : requests) {
    auto store_functor =
      detail::scan_result_functor(request.values, helper(), cache, stream, mr, _keys_are_sorted);
    for (auto const& aggregation : request.aggregations) {
      // TODO (dm): single pass compute all supported reductions
      cudf::detail::aggregation_dispatcher(aggregation->kind, store_functor, *aggregation);
    }
  }

  auto results = detail::extract_results(requests, cache, stream, mr);

  return std::pair(helper().sorted_keys(stream, mr), std::move(results));
}
}  // namespace groupby
}  // namespace cudf
