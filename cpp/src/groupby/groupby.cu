/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/groupby/group_replace_nulls.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction/detail/histogram.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/iterator>

#include <memory>
#include <utility>

namespace cudf {
namespace groupby {
// Constructor
groupby::groupby(table_view const& keys,
                 null_policy include_null_keys,
                 sorted keys_are_sorted,
                 std::vector<order> const& column_order,
                 std::vector<null_order> const& null_precedence)
  : _keys{keys},
    _include_null_keys{include_null_keys},
    _keys_are_sorted{keys_are_sorted},
    _column_order{column_order},
    _null_precedence{null_precedence}
{
}

// Select hash vs. sort groupby implementation
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby::dispatch_aggregation(
  std::span<aggregation_request const> requests,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // If sort groupby has been called once on this groupby object, then
  // always use sort groupby from now on. Because once keys are sorted,
  // all the aggs that can be done by hash groupby are efficiently done by
  // sort groupby as well.
  // Only use hash groupby if the keys aren't sorted and all requests can be
  // satisfied with a hash implementation
  if (_keys_are_sorted == sorted::NO and not _helper and
      detail::hash::can_use_hash_groupby(requests)) {
    return detail::hash::groupby(_keys, requests, _include_null_keys, stream, mr);
  } else {
    return sort_aggregate(requests, stream, mr);
  }
}

// Destructor
// Needs to be in source file because sort_groupby_helper was forward declared
groupby::~groupby() = default;

namespace {

/**
 * @brief Factory to construct empty result columns.
 *
 * Adds special handling for COLLECT_LIST/COLLECT_SET, because:
 * 1. `make_empty_column()` does not support construction of nested columns.
 * 2. Empty lists need empty child columns, to persist type information.
 * Adds special handling for RANK, because it needs to return double type column when rank_method is
 * AVERAGE or percentage is true.
 */
struct empty_column_constructor {
  column_view values;
  aggregation const& agg;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename ValuesType, aggregation::Kind k>
  std::unique_ptr<cudf::column> operator()() const
  {
    using namespace cudf;
    using namespace cudf::detail;

    if constexpr (k == aggregation::Kind::COLLECT_LIST || k == aggregation::Kind::COLLECT_SET ||
                  k == aggregation::Kind::TOP_K) {
      return make_lists_column(0, make_empty_column(type_id::INT32), empty_like(values), 0, {});
    }

    if constexpr (k == aggregation::Kind::HISTOGRAM) {
      return make_lists_column(0,
                               make_empty_column(type_to_id<size_type>()),
                               cudf::reduction::detail::make_empty_histogram_like(values),
                               0,
                               {});
    }
    if constexpr (k == aggregation::Kind::MERGE_HISTOGRAM) { return empty_like(values); }

    if constexpr (k == aggregation::Kind::SUM_OVERFLOW) {
      // SUM_OVERFLOW returns a struct with sum (same type as input) and overflow (bool)
      // children
      std::vector<std::unique_ptr<cudf::column>> children;
      children.push_back(make_empty_column(values.type()));
      children.push_back(make_empty_column(cudf::data_type{cudf::type_id::BOOL8}));
      return make_structs_column(0, std::move(children), 0, {}, stream, mr);
    }

    if constexpr (k == aggregation::Kind::RANK) {
      auto const& rank_agg = dynamic_cast<cudf::detail::rank_aggregation const&>(agg);
      if (rank_agg._method == cudf::rank_method::AVERAGE or
          rank_agg._percentage != rank_percentage::NONE)
        return make_empty_column(type_to_id<double>());
      return make_empty_column(target_type(values.type(), k));
    }

    // If `values` is LIST typed, and the aggregation results match the type,
    // construct empty results based on `values`.
    // Most generally, this applies if input type matches output type.
    //
    // Note: `target_type_t` is not recursive, and `ValuesType` does not consider children.
    //       It is important that `COLLECT_LIST` and `COLLECT_SET` are handled before this
    //       point, because `COLLECT_LIST(LIST)` produces `LIST<LIST>`, but `target_type_t`
    //       wouldn't know the difference.
    if constexpr (std::is_same_v<target_type_t<ValuesType, k>, ValuesType>) {
      return empty_like(values);
    }

    if constexpr (k == aggregation::Kind::HOST_UDF) {
      auto const& udf_base_ptr =
        dynamic_cast<cudf::detail::host_udf_aggregation const&>(agg).udf_ptr;
      auto const udf_ptr = dynamic_cast<groupby_host_udf const*>(udf_base_ptr.get());
      CUDF_EXPECTS(udf_ptr != nullptr, "Invalid HOST_UDF instance for groupby aggregation.");
      return udf_ptr->get_empty_output(stream, mr);
    }

    return make_empty_column(target_type(values.type(), k));
  }
};

/// Make an empty table with appropriate types for requested aggs
template <typename RequestType>
auto empty_results(std::span<RequestType const> requests,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
{
  std::vector<aggregation_result> empty_results;

  std::transform(requests.begin(),
                 requests.end(),
                 std::back_inserter(empty_results),
                 [stream, mr](auto const& request) {
                   std::vector<std::unique_ptr<column>> results;

                   std::transform(request.aggregations.begin(),
                                  request.aggregations.end(),
                                  std::back_inserter(results),
                                  [&request, stream, mr](auto const& agg) {
                                    return cudf::detail::dispatch_type_and_aggregation(
                                      request.values.type(),
                                      agg->kind,
                                      empty_column_constructor{request.values, *agg, stream, mr});
                                  });

                   return aggregation_result{std::move(results)};
                 });

  return empty_results;
}

/// Verifies the agg requested on the request's values is valid
template <typename RequestType>
void verify_valid_requests(std::span<RequestType const> requests)
{
  CUDF_EXPECTS(
    std::all_of(
      requests.begin(),
      requests.end(),
      [](auto const& request) {
        return std::all_of(
          request.aggregations.begin(), request.aggregations.end(), [&request](auto const& agg) {
            auto values_type = cudf::is_dictionary(request.values.type())
                                 ? cudf::dictionary_column_view(request.values).keys().type()
                                 : request.values.type();
            return cudf::detail::is_valid_aggregation(values_type, agg->kind);
          });
      }),
    "Invalid type/aggregation combination.");

  // Additional validation for SUM_OVERFLOW: only signed integers and decimals are supported
  for (auto const& request : requests) {
    for (auto const& agg : request.aggregations) {
      if (agg->kind == aggregation::SUM_OVERFLOW) {
        CUDF_EXPECTS(
          cudf::detail::is_valid_aggregation(request.values.type(), aggregation::SUM_OVERFLOW),
          "SUM_OVERFLOW aggregation only supports signed integer types and decimal types. "
          "Unsigned integers, bool, dictionary columns, and other types are not supported.");
      }
    }
  }
}

}  // namespace

// Compute aggregation requests
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby::aggregate(
  std::span<aggregation_request const> requests,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    std::all_of(requests.begin(),
                requests.end(),
                [this](auto const& request) { return request.values.size() == _keys.num_rows(); }),
    "Size mismatch between request values and groupby keys.");

  verify_valid_requests(requests);

  if (_keys.num_rows() == 0) { return {empty_like(_keys), empty_results(requests, stream, mr)}; }

  return dispatch_aggregation(requests, stream, mr);
}

// Compute scan requests
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby::scan(
  std::span<scan_request const> requests,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    std::all_of(requests.begin(),
                requests.end(),
                [this](auto const& request) { return request.values.size() == _keys.num_rows(); }),
    "Size mismatch between request values and groupby keys.");

  verify_valid_requests(requests);

  if (_keys.num_rows() == 0) {
    return std::pair(empty_like(_keys), empty_results(requests, stream, mr));
  }

  return sort_scan(requests, stream, mr);
}

groupby::groups groupby::get_groups(table_view values,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto grouped_keys = helper().sorted_keys(stream, mr);

  auto const& group_offsets       = helper().group_offsets(stream);
  auto const group_offsets_vector = cudf::detail::make_std_vector(group_offsets, stream);

  if (not values.is_empty()) {
    auto grouped_values = cudf::detail::gather(values,
                                               helper().key_sort_order(stream),
                                               cudf::out_of_bounds_policy::DONT_CHECK,
                                               cudf::negative_index_policy::NOT_ALLOWED,
                                               stream,
                                               mr);
    return groupby::groups{
      std::move(grouped_keys), std::move(group_offsets_vector), std::move(grouped_values)};
  } else {
    return groupby::groups{std::move(grouped_keys), std::move(group_offsets_vector)};
  }
}

std::pair<std::unique_ptr<table>, std::unique_ptr<table>> groupby::replace_nulls(
  table_view const& values,
  std::span<cudf::replace_policy const> replace_policies,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(_keys.num_rows() == values.num_rows(),
               "Size mismatch between group labels and value.");
  CUDF_EXPECTS(static_cast<cudf::size_type>(replace_policies.size()) == values.num_columns(),
               "Size mismatch between num_columns and replace_policies.");

  if (values.is_empty()) { return std::pair(empty_like(_keys), empty_like(values)); }

  auto const& group_labels = helper().group_labels(stream);
  std::vector<std::unique_ptr<column>> results;
  results.reserve(values.num_columns());
  std::transform(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{values.num_columns()},
    std::back_inserter(results),
    [&](auto i) {
      bool nullable       = values.column(i).nullable();
      auto final_mr       = nullable ? cudf::get_current_device_resource_ref() : mr;
      auto grouped_values = helper().grouped_values(values.column(i), stream, final_mr);
      return nullable ? detail::group_replace_nulls(
                          *grouped_values, group_labels, replace_policies[i], stream, mr)
                      : std::move(grouped_values);
    });

  return std::pair(std::move(helper().sorted_keys(stream, mr)),
                   std::make_unique<table>(std::move(results)));
}

// Get the sort helper object
detail::sort::sort_groupby_helper& groupby::helper()
{
  if (_helper) return *_helper;
  _helper = std::make_unique<detail::sort::sort_groupby_helper>(
    _keys, _include_null_keys, _keys_are_sorted, _null_precedence);
  return *_helper;
};

std::pair<std::unique_ptr<table>, std::unique_ptr<table>> groupby::shift(
  table_view const& values,
  std::span<size_type const> offsets,
  std::vector<std::reference_wrapper<scalar const>> const& fill_values,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(values.num_columns() == static_cast<size_type>(fill_values.size()),
               "Mismatch number of fill_values and columns.");
  CUDF_EXPECTS(std::equal(values.begin(),
                          values.end(),
                          fill_values.cbegin(),
                          fill_values.cend(),
                          [](auto const& col, auto const& scalar) {
                            return cudf::have_same_types(col, scalar.get());
                          }),
               "values and fill_value should have the same type.",
               cudf::data_type_error);
  std::vector<std::unique_ptr<column>> results;
  auto const& group_offsets = helper().group_offsets(stream);
  std::transform(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{values.num_columns()},
    std::back_inserter(results),
    [&](size_type i) {
      auto grouped_values =
        helper().grouped_values(values.column(i), stream, cudf::get_current_device_resource_ref());
      return cudf::detail::segmented_shift(
        grouped_values->view(), group_offsets, offsets[i], fill_values[i].get(), stream, mr);
    });

  return std::pair(helper().sorted_keys(stream, mr),
                   std::make_unique<cudf::table>(std::move(results)));
}

}  // namespace groupby
}  // namespace cudf
