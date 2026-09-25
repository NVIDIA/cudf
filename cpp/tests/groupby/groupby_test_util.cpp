/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby_test_util.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

namespace {

std::unique_ptr<cudf::table> reorder_aggregation_values(cudf::column_view values,
                                                        cudf::column_view group_order)
{
  if (group_order.size() == 0 || values.size() == group_order.size()) {
    return cudf::gather(cudf::table_view{{values}}, group_order);
  }

  // Multiple quantiles produce a contiguous block of results for each key.
  CUDF_EXPECTS(values.size() % group_order.size() == 0, "Unexpected aggregation result size");
  auto const values_per_group   = values.size() / group_order.size();
  auto const [groups, validity] = cudf::test::to_host<cudf::size_type>(group_order);
  std::vector<cudf::size_type> indices;
  indices.reserve(values.size());
  for (auto group : groups) {
    for (cudf::size_type value = 0; value < values_per_group; ++value) {
      indices.push_back(group * values_per_group + value);
    }
  }
  cudf::test::fixed_width_column_wrapper<cudf::size_type> order(indices.begin(), indices.end());
  return cudf::gather(cudf::table_view{{values}}, order);
}

}  // namespace

void test_single_agg(cudf::column_view const& keys,
                     cudf::column_view const& values,
                     cudf::column_view const& expect_keys,
                     cudf::column_view const& expect_vals,
                     std::unique_ptr<cudf::groupby_aggregation>&& agg,
                     include_nth_aggregation include_nth,
                     cudf::null_policy include_null_keys,
                     cudf::sorted keys_are_sorted,
                     std::vector<cudf::order> const& column_order,
                     std::vector<cudf::null_order> const& null_precedence,
                     cudf::sorted reference_keys_are_sorted,
                     test_streaming use_streaming,
                     std::source_location const& location)
{
  SCOPED_TRACE("Original failure location: " + std::string{location.file_name()} + ":" +
               std::to_string(location.line()));

  auto const [sorted_expect_keys, sorted_expect_vals] = [&]() {
    if (reference_keys_are_sorted == cudf::sorted::NO) {
      auto const sort_expect_order =
        cudf::sorted_order(cudf::table_view{{expect_keys}}, column_order, null_precedence);
      auto sorted_expect_keys = cudf::gather(cudf::table_view{{expect_keys}}, *sort_expect_order);
      auto sorted_expect_vals = reorder_aggregation_values(expect_vals, *sort_expect_order);
      return std::make_pair(std::move(sorted_expect_keys), std::move(sorted_expect_vals));
    } else {
      auto sorted_expect_keys = std::make_unique<cudf::table>(cudf::table_view{{expect_keys}});
      auto sorted_expect_vals = std::make_unique<cudf::table>(cudf::table_view{{expect_vals}});
      return std::make_pair(std::move(sorted_expect_keys), std::move(sorted_expect_vals));
    }
  }();

  // --- Standard groupby path ---
  {
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = values;

    requests[0].aggregations.push_back(std::unique_ptr<cudf::groupby_aggregation>{
      dynamic_cast<cudf::groupby_aggregation*>(agg->clone().release())});

    if (include_nth == include_nth_aggregation::YES) {
      // Exercise the same reductions in mixed aggregation requests.
      requests[0].aggregations.push_back(
        cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
    }

    // since the default behavior of cudf::groupby(...) for an empty null_precedence vector is
    // null_order::AFTER whereas for cudf::sorted_order(...) it's null_order::BEFORE
    auto const precedence = null_precedence.empty()
                              ? std::vector<cudf::null_order>(1, cudf::null_order::BEFORE)
                              : null_precedence;

    cudf::groupby::groupby gb_obj(
      cudf::table_view({keys}), include_null_keys, keys_are_sorted, column_order, precedence);

    auto result = gb_obj.aggregate(requests, cudf::test::get_default_stream());

    auto const sort_order  = cudf::sorted_order(result.first->view(), column_order, precedence);
    auto const sorted_keys = cudf::gather(result.first->view(), *sort_order);
    auto const sorted_vals =
      reorder_aggregation_values(result.second[0].results[0]->view(), *sort_order);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_expect_keys, *sorted_keys);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_expect_vals->get_column(0),
                                        sorted_vals->get_column(0));
  }

  // --- Streaming groupby path (single-batch, validates against same expected output) ---
  // Skip streaming for: empty input (finalize throws), dictionary values (unsupported by
  // streaming's row operators), ARGMIN/ARGMAX (batch-local row indices), or pre-sorted
  // keys (streaming doesn't support sorted mode).
  auto const skip_streaming =
    keys.size() == 0 || expect_keys.size() == 0 ||
    values.type().id() == cudf::type_id::DICTIONARY32 || agg->kind == cudf::aggregation::ARGMIN ||
    agg->kind == cudf::aggregation::ARGMAX || keys_are_sorted == cudf::sorted::YES;

  if (use_streaming == test_streaming::YES && !skip_streaming) {
    SCOPED_TRACE("streaming groupby path");

    cudf::table_view data{{keys, values}};
    std::vector<cudf::size_type> key_indices{0};

    cudf::groupby::streaming_aggregation_request sreq;
    sreq.column_index = 1;
    sreq.aggregation  = std::unique_ptr<cudf::groupby_aggregation>{
      dynamic_cast<cudf::groupby_aggregation*>(agg->clone().release())};

    std::vector<cudf::groupby::streaming_aggregation_request> sreqs;
    sreqs.push_back(std::move(sreq));

    auto const max_distinct_keys = std::max(keys.size(), cudf::size_type{64});
    cudf::groupby::streaming_groupby sgb(key_indices, sreqs, max_distinct_keys, include_null_keys);
    sgb.aggregate(data, cudf::test::get_default_stream());
    auto [skeys, sresults] = sgb.finalize(cudf::test::get_default_stream());

    auto const precedence = null_precedence.empty()
                              ? std::vector<cudf::null_order>(1, cudf::null_order::BEFORE)
                              : null_precedence;

    auto const sort_order  = cudf::sorted_order(skeys->view(), column_order, precedence);
    auto const sorted_keys = cudf::gather(skeys->view(), *sort_order);
    auto const sorted_vals =
      cudf::gather(cudf::table_view({sresults[0].results[0]->view()}), *sort_order);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_expect_keys, *sorted_keys);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_expect_vals->get_column(0),
                                        sorted_vals->get_column(0));
  }
}

void test_sum_agg(cudf::column_view const& keys,
                  cudf::column_view const& values,
                  cudf::column_view const& expected_keys,
                  cudf::column_view const& expected_values,
                  std::source_location const& location)
{
  auto const do_test = [&](auto const include_nth_option) {
    test_single_agg(keys,
                    values,
                    expected_keys,
                    expected_values,
                    cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                    include_nth_option,
                    cudf::null_policy::INCLUDE,
                    cudf::sorted::NO,
                    {},
                    {},
                    cudf::sorted::NO,
                    test_streaming::NO,
                    location);
  };
  do_test(include_nth_aggregation::YES);
  do_test(include_nth_aggregation::NO);
}

void test_single_scan(cudf::column_view const& keys,
                      cudf::column_view const& values,
                      cudf::column_view const& expect_keys,
                      cudf::column_view const& expect_vals,
                      std::unique_ptr<cudf::groupby_scan_aggregation>&& agg,
                      cudf::null_policy include_null_keys,
                      cudf::sorted keys_are_sorted,
                      std::vector<cudf::order> const& column_order,
                      std::vector<cudf::null_order> const& null_precedence,
                      std::source_location const& location)
{
  SCOPED_TRACE("Original failure location: " + std::string{location.file_name()} + ":" +
               std::to_string(location.line()));

  std::vector<cudf::groupby::scan_request> requests;
  requests.emplace_back();
  requests[0].values = values;
  requests[0].aggregations.push_back(std::move(agg));

  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), include_null_keys, keys_are_sorted, column_order, null_precedence);

  auto result = gb_obj.scan(requests);

  // Group order is unspecified; preserve each group's row order when normalizing it.
  auto const actual_order   = cudf::stable_sorted_order(result.first->view());
  auto const expected_order = cudf::stable_sorted_order(cudf::table_view{{expect_keys}});
  auto const actual_keys    = cudf::gather(result.first->view(), *actual_order);
  auto const actual_values =
    cudf::gather(cudf::table_view{{result.second[0].results[0]->view()}}, *actual_order);
  auto const expected_keys   = cudf::gather(cudf::table_view{{expect_keys}}, *expected_order);
  auto const expected_values = cudf::gather(cudf::table_view{{expect_vals}}, *expected_order);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected_keys, *actual_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_values->get_column(0), actual_values->get_column(0));
}
