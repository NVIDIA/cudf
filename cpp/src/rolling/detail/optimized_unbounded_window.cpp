/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction/detail/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf::detail {
namespace {

/// Converts rolling_aggregation to corresponding reduce/groupby_aggregation.
template <typename Base>
struct aggregation_converter {
  template <aggregation::Kind k>
  std::unique_ptr<Base> operator()() const
  {
    if constexpr (std::is_same_v<Base, cudf::groupby_aggregation> and
                  k == aggregation::Kind::COUNT_ALL) {
      // Note: COUNT_ALL cannot be used as a cudf::reduce_aggregation; cudf::reduce does not support
      // it.
      return cudf::make_count_aggregation<Base>(null_policy::INCLUDE);
    } else if constexpr (std::is_same_v<Base, cudf::groupby_aggregation> and
                         k == aggregation::Kind::COUNT_VALID) {
      // Note: COUNT_ALL cannot be used as a cudf::reduce_aggregation; cudf::reduce does not support
      // it.
      return cudf::make_count_aggregation<Base>(null_policy::EXCLUDE);
    } else if constexpr (k == aggregation::Kind::SUM) {
      return cudf::make_sum_aggregation<Base>();
    } else if constexpr (k == aggregation::Kind::MIN) {
      return cudf::make_min_aggregation<Base>();
    } else if constexpr (k == aggregation::Kind::MAX) {
      return cudf::make_max_aggregation<Base>();
    } else {
      CUDF_FAIL("Unsupported aggregation kind for optimized unbounded windows.");
    }
  }
};

template <typename Base>
std::unique_ptr<Base> convert_to(cudf::rolling_aggregation const& aggr)
{
  return cudf::detail::aggregation_dispatcher(aggr.kind, aggregation_converter<Base>{});
}

/// Compute unbounded rolling window via groupby-aggregation.
/// Used for input that has groupby key columns.
std::unique_ptr<column> aggregation_based_rolling_window(table_view const& group_keys,
                                                         column_view const& input,
                                                         rolling_aggregation const& aggr,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(group_keys.num_columns() > 0,
               "Ungrouped rolling window not supported in aggregation path.");

  auto agg_requests = std::vector<cudf::groupby::aggregation_request>{};
  agg_requests.emplace_back();
  agg_requests.front().values = input;
  agg_requests.front().aggregations.push_back(convert_to<cudf::groupby_aggregation>(aggr));

  auto group_by = cudf::groupby::groupby{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES};
  auto aggregation_results =
    group_by.aggregate(agg_requests, stream, cudf::get_current_device_resource_ref());
  auto const& aggregation_result_col = aggregation_results.second.front().results.front();

  using cudf::groupby::detail::sort::sort_groupby_helper;
  auto helper = sort_groupby_helper{group_keys, cudf::null_policy::INCLUDE, cudf::sorted::YES, {}};
  auto const& group_labels = helper.group_labels(stream);

  auto result_columns = cudf::detail::gather(cudf::table_view{{*aggregation_result_col}},
                                             group_labels,
                                             cudf::out_of_bounds_policy::DONT_CHECK,
                                             cudf::negative_index_policy::NOT_ALLOWED,
                                             stream,
                                             mr)
                          ->release();
  return std::move(result_columns.front());
}

/// Compute unbounded rolling window via cudf::reduce.
/// Used for input that has no groupby keys. i.e. The window spans the column.
std::unique_ptr<column> reduction_based_rolling_window(column_view const& input,
                                                       rolling_aggregation const& aggr,
                                                       cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr)
{
  auto const reduce_results = [&] {
    auto const return_dtype = cudf::detail::target_type(input.type(), aggr.kind);
    if (aggr.kind == aggregation::COUNT_ALL) {
      return cudf::make_fixed_width_scalar(
        input.size(), stream, cudf::get_current_device_resource_ref());
    } else if (aggr.kind == aggregation::COUNT_VALID) {
      return cudf::make_fixed_width_scalar(
        input.size() - input.null_count(), stream, cudf::get_current_device_resource_ref());
    } else {
      return cudf::reduction::detail::reduce(input,
                                             *convert_to<cudf::reduce_aggregation>(aggr),
                                             return_dtype,
                                             std::nullopt,
                                             stream,
                                             cudf::get_current_device_resource_ref());
    }
  }();
  // Blow up results into separate column.
  return cudf::make_column_from_scalar(*reduce_results, input.size(), stream, mr);
}
}  // namespace

bool can_optimize_unbounded_window(bool unbounded_preceding,
                                   bool unbounded_following,
                                   size_type min_periods,
                                   rolling_aggregation const& agg)
{
  auto is_supported = [](auto const& agg) {
    switch (agg.kind) {
      case cudf::aggregation::Kind::COUNT_ALL: [[fallthrough]];
      case cudf::aggregation::Kind::COUNT_VALID: [[fallthrough]];
      case cudf::aggregation::Kind::SUM: [[fallthrough]];
      case cudf::aggregation::Kind::MIN: [[fallthrough]];
      case cudf::aggregation::Kind::MAX: return true;
      default:
        // COLLECT_LIST and COLLECT_SET can be added at a later date.
        // Other aggregations do not fit into the [UNBOUNDED, UNBOUNDED]
        // category. For instance:
        // 1. Ranking functions (ROW_NUMBER, RANK, DENSE_RANK, PERCENT_RANK)
        //    use [UNBOUNDED PRECEDING, CURRENT ROW].
        // 2. LEAD/LAG are defined on finite row boundaries.
        return false;
    }
  };

  return unbounded_preceding && unbounded_following && (min_periods == 1) && is_supported(agg);
}

std::unique_ptr<column> optimized_unbounded_window(table_view const& group_keys,
                                                   column_view const& input,
                                                   rolling_aggregation const& aggr,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  return group_keys.num_columns() > 0
           ? aggregation_based_rolling_window(group_keys, input, aggr, stream, mr)
           : reduction_based_rolling_window(input, aggr, stream, mr);
}
}  // namespace cudf::detail
