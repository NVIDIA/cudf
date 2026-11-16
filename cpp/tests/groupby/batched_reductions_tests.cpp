/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

struct GroupbyBatchedReductionsTest : cudf::test::BaseFixture {};

TEST_F(GroupbyBatchedReductionsTest, NullableSumsShareValidityWithoutRequestedCount)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{2, 1, 2, 1, 3, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> values{{2, 1, 3, 0, 0, 0},
                                                         {true, true, true, false, false, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sum{{1, 5, 0}, {true, true, false}};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_squares{{1, 13, 0}, {true, true, false}};
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests.front().values = values;
  requests.front().aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests.front().aggregations.push_back(
    cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::groupby gb(cudf::table_view{{keys}});
  auto result      = gb.aggregate(requests);
  auto const order = cudf::sorted_order(result.first->view());
  auto sorted_keys = cudf::gather(result.first->view(), order->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted_keys->get_column(0));
  auto const& outputs = result.second.front().results;
  ASSERT_EQ(outputs.size(), 2);
  auto sorted =
    cudf::gather(cudf::table_view{{outputs[0]->view(), outputs[1]->view()}}, order->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sum, sorted->get_column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_squares, sorted->get_column(1));
}

TEST_F(GroupbyBatchedReductionsTest, NullableSumWithOneExtremum)
{
  // Exercise scalar, warp and multi-chunk reductions, plus an all-null group. Each call requests
  // only one extremum, and both result orders must retain the shared validity mask.
  std::array<cudf::size_type, 4> const sizes{3, 33, 1025, 2};
  std::vector<int32_t> key_data;
  std::vector<int32_t> value_data;
  std::vector<bool> validity;
  for (cudf::size_type group = 0; group < static_cast<cudf::size_type>(sizes.size()); ++group) {
    for (cudf::size_type row = 0; row < sizes[group]; ++row) {
      key_data.push_back(group);
      value_data.push_back(group * 100 + row - 500);
      validity.push_back(group != 3 && row % 3 != 1);
    }
  }
  cudf::test::fixed_width_column_wrapper<int32_t> keys(key_data.begin(), key_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> values(
    value_data.begin(), value_data.end(), validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sum{{-998, -8448, 144625, 0},
                                                               {true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_min{{-500, -400, -300, 0},
                                                               {true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_max{{-498, -368, 723, 0},
                                                               {true, true, true, false}};

  for (bool const minimum : {false, true}) {
    for (bool const sum_first : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "minimum=" << minimum << " sum_first=" << sum_first);
      std::vector<cudf::groupby::aggregation_request> requests(1);
      requests[0].values = values;
      auto& aggs         = requests[0].aggregations;
      auto sum           = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
      auto extremum      = minimum ? cudf::make_min_aggregation<cudf::groupby_aggregation>()
                                   : cudf::make_max_aggregation<cudf::groupby_aggregation>();
      if (sum_first) {
        aggs.push_back(std::move(sum));
        aggs.push_back(std::move(extremum));
      } else {
        aggs.push_back(std::move(extremum));
        aggs.push_back(std::move(sum));
      }
      cudf::groupby::groupby gb(cudf::table_view{{keys}});
      auto const [result_keys, results] = gb.aggregate(requests);
      ASSERT_EQ(results.size(), 1);
      ASSERT_EQ(results[0].results.size(), 2);
      auto const order  = cudf::sorted_order(result_keys->view());
      auto const sorted = cudf::gather(
        cudf::table_view{
          {result_keys->view().column(0), *results[0].results[0], *results[0].results[1]}},
        order->view());
      auto const expected_extremum =
        minimum ? cudf::column_view(expected_min) : cudf::column_view(expected_max);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted->get_column(0));
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sum, sorted->get_column(sum_first ? 1 : 2));
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_extremum, sorted->get_column(sum_first ? 2 : 1));
    }
  }
}

TEST_F(GroupbyBatchedReductionsTest, OmitAllValidMasksFromNullableInputs)
{
  for (bool const has_null_group : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "has_null_group=" << has_null_group);
    cudf::test::fixed_width_column_wrapper<int32_t> keys{2, 1, 2, 1, 3, 3};
    auto const make_values = [&] {
      return cudf::test::fixed_width_column_wrapper<int32_t>{
        {2, 1, 3, 0, 4, 0}, {true, true, true, false, !has_null_group, false}};
    };
    // Separate columns exercise the selected-row, fused extrema and fused additive paths.
    auto arg_values     = make_values();
    auto extrema_values = make_values();
    auto sum_values     = make_values();
    std::vector<cudf::groupby::aggregation_request> requests(3);
    requests[0].values = arg_values;
    requests[0].aggregations.push_back(cudf::make_argmin_aggregation<cudf::groupby_aggregation>());
    requests[0].aggregations.push_back(cudf::make_argmax_aggregation<cudf::groupby_aggregation>());
    requests[1].values = extrema_values;
    requests[1].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
    requests[1].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
    requests[1].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    requests[2].values = sum_values;
    requests[2].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    requests[2].aggregations.push_back(
      cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>());

    cudf::groupby::groupby gb(cudf::table_view{{keys}});
    auto const [result_keys, results] = gb.aggregate(requests);
    std::vector<cudf::column_view> outputs;
    for (auto const& request_result : results) {
      for (auto const& output : request_result.results) {
        EXPECT_EQ(output->nullable(), has_null_group);
        EXPECT_EQ(output->null_count(), has_null_group ? 1 : 0);
        outputs.push_back(output->view());
      }
    }
    auto const order       = cudf::sorted_order(result_keys->view());
    auto const sorted_keys = cudf::gather(result_keys->view(), order->view());
    auto const sorted      = cudf::gather(cudf::table_view{outputs}, order->view());
    cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{1, 2, 3};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_argmin{
      {1, 0, 4}, {true, true, !has_null_group}};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_argmax{
      {1, 2, 4}, {true, true, !has_null_group}};
    cudf::test::fixed_width_column_wrapper<int32_t> expected_min{{1, 2, 4},
                                                                 {true, true, !has_null_group}};
    cudf::test::fixed_width_column_wrapper<int32_t> expected_max{{1, 3, 4},
                                                                 {true, true, !has_null_group}};
    cudf::test::fixed_width_column_wrapper<int64_t> expected_sum{{1, 5, 4},
                                                                 {true, true, !has_null_group}};
    cudf::test::fixed_width_column_wrapper<int64_t> expected_squares{{1, 13, 16},
                                                                     {true, true, !has_null_group}};
    std::array<cudf::column_view, 7> const expected{expected_argmin,
                                                    expected_argmax,
                                                    expected_min,
                                                    expected_max,
                                                    expected_sum,
                                                    expected_sum,
                                                    expected_squares};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted_keys->get_column(0));
    ASSERT_EQ(sorted->num_columns(), expected.size());
    for (cudf::size_type i = 0; i < sorted->num_columns(); ++i) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected[i], sorted->get_column(i));
    }
  }
}
