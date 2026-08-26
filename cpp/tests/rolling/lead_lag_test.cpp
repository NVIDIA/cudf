/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <initializer_list>
#include <memory>
#include <string>

struct LeadLagWindowTest : public cudf::test::BaseFixture {};

template <typename T>
struct TypedLeadLagWindowTest : public cudf::test::BaseFixture {};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypes,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::DurationTypes,
                                        cudf::test::TimestampTypes>;

TYPED_TEST_SUITE(TypedLeadLagWindowTest, TypesForTest);

TYPED_TEST(TypedLeadLagWindowTest, LeadLagBasics)
{
  using T = int32_t;

  auto const input_col =
    cudf::test::fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50}.release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_3_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(3));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lead_3_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{3, 4, 5, -1, -1, -1, 30, 40, 50, -1, -1, -1},
                                              {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0}}
      .release()
      ->view());

  auto lag_2_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lag_2_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{-1, -1, 0, 1, 2, 3, -1, -1, 0, 10, 20, 30},
                                              {0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, LeadLagWithNulls)
{
  using T = TypeParam;

  auto const input_col =
    cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                              {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
      .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_3_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(3));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lead_3_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{3, 4, 5, -1, -1, -1, 30, 40, 50, -1, -1, -1},
                                              {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0}}
      .release()
      ->view());

  auto const lag_2_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lag_2_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{-1, -1, 0, 1, -1, 3, -1, -1, 0, 10, -1, 30},
                                              {0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithDefaults)
{
  using T = TypeParam;

  auto const input_col =
    cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                              {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
      .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto const default_value =
    cudf::make_fixed_width_scalar(cudf::test::detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto lead_3_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lead_3_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{3, 4, 5, 99, 99, 99, 30, 40, 50, 99, 99, 99},
                                              {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
      .release()
      ->view());

  auto const lag_2_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lag_2_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{99, 99, 0, 1, -1, 3, 99, 99, 0, 10, -1, 30},
                                              {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithDefaultsContainingNulls)
{
  using T = TypeParam;

  auto const input_col =
    cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                              {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
      .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const default_outputs =
    cudf::test::fixed_width_column_wrapper<T>{{-1, 99, -1, 99, 99, -1, 99, 99, -1, 99, 99, -1},
                                              {0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}}
      .release();

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_3_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lead_3_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{3, 4, 5, 99, 99, -1, 30, 40, 50, 99, 99, -1},
                                              {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0}}
      .release()
      ->view());

  auto const lag_2_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lag_2_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{-1, 99, 0, 1, -1, 3, 99, 99, 0, 10, -1, 30},
                                              {0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithOutOfRangeOffsets)
{
  using T = TypeParam;

  auto const input_col =
    cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                              {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
      .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const default_value =
    cudf::make_fixed_width_scalar(cudf::test::detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_30_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(30));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lead_30_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                              {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      .release()
      ->view());

  auto const lag_20_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(20));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lag_20_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99},
                                              {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithZeroOffsets)
{
  using T = TypeParam;

  auto const input_col =
    cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                              {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
      .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_0_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(0));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*lead_0_output_col, *input_col);

  auto const lag_0_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(0));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*lag_0_output_col, *input_col);
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithNegativeOffsets)
{
  using T = TypeParam;

  auto const input_col =
    cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                              {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
      .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const default_value =
    cudf::make_fixed_width_scalar(cudf::test::detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lag_minus_3_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(-3));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lag_minus_3_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{3, 4, 5, 99, 99, 99, 30, 40, 50, 99, 99, 99},
                                              {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
      .release()
      ->view());

  auto const lead_minus_2_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(-2));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lead_minus_2_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{99, 99, 0, 1, -1, 3, 99, 99, 0, 10, -1, 30},
                                              {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithNoGrouping)
{
  using T = TypeParam;

  auto const input_col =
    cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5}, {1, 1, 0, 1, 1, 1}}.release();

  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{}};

  auto const default_value =
    cudf::make_fixed_width_scalar(cudf::test::detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_3_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(3));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lead_3_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{3, 4, 5, 99, 99, 99}, {1, 1, 1, 1, 1, 1}}
      .release()
      ->view());

  auto const lag_2_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lag_2_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{99, 99, 0, 1, -1, 3}, {1, 1, 1, 1, 0, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithAllNullInput)
{
  using T = TypeParam;

  auto const input_col = cudf::test::fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return false;
    })}.release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const default_value =
    cudf::make_fixed_width_scalar(cudf::test::detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_3_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lead_3_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{-1, -1, -1, 99, 99, 99, -1, -1, -1, 99, 99, 99},
                                              {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1}}
      .release()
      ->view());

  auto const lag_2_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 *default_outputs,
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *lag_2_output_col,
    cudf::test::fixed_width_column_wrapper<T>{{99, 99, -1, -1, -1, -1, 99, 99, -1, -1, -1, -1},
                                              {1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, DefaultValuesWithoutLeadLag)
{
  // Test that passing default values for window-functions
  // other than lead/lag lead to cudf::logic_error.

  using T = TypeParam;

  auto const input_col = cudf::test::fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return true;
    })}.release();

  auto const grouping_key  = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const default_value =
    cudf::make_fixed_width_scalar(cudf::test::detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto const assert_aggregation_fails = [&](auto&& aggr) {
    EXPECT_THROW(
      cudf::grouped_rolling_window(grouping_keys,
                                   input_col->view(),
                                   default_outputs->view(),
                                   preceding,
                                   following,
                                   min_periods,
                                   *cudf::make_count_aggregation<cudf::rolling_aggregation>()),
      cudf::logic_error);
  };

  auto aggs = {cudf::make_count_aggregation<cudf::rolling_aggregation>(),
               cudf::make_min_aggregation<cudf::rolling_aggregation>()};
  std::for_each(
    aggs.begin(), aggs.end(), [&](auto& agg) { assert_aggregation_fails(std::move(agg)); });
}

template <typename T>
struct TypedNestedLeadLagWindowTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedNestedLeadLagWindowTest, TypesForTest);

TYPED_TEST(TypedNestedLeadLagWindowTest, NumericListsWithNullsAllOver)
{
  using T   = TypeParam;
  using lcw = cudf::test::lists_column_wrapper<T, int32_t>;

  auto null_at_2       = cudf::test::iterators::null_at(2);
  auto const input_col = lcw{{{0, 0},
                              {1, 1},
                              {2, 2},
                              {3, 3, 3},
                              {{4, 4, 4, 4}, null_at_2},
                              {5, 5, 5, 5, 5},
                              {0, 0},
                              {10, 10},
                              {20, 20},
                              {30, 30, 30},
                              {40, 40, 40, 40},
                              {{50, 50, 50, 50, 50}, null_at_2}},
                             null_at_2}
                           .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_3_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(3));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lead_3_output_col->view(),
                                      lcw{{{3, 3, 3},
                                           {{4, 4, 4, 4}, null_at_2},
                                           {5, 5, 5, 5, 5},
                                           {},
                                           {},
                                           {},
                                           {30, 30, 30},
                                           {40, 40, 40, 40},
                                           {{50, 50, 50, 50, 50}, null_at_2},
                                           {},
                                           {},
                                           {}},
                                          cudf::test::iterators::nulls_at({3, 4, 5, 9, 10, 11})}
                                        .release()
                                        ->view());

  auto lag_1_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(1));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lag_1_output_col->view(),
                                      lcw{{{},
                                           {0, 0},
                                           {1, 1},
                                           {2, 2},
                                           {3, 3, 3},
                                           {{4, 4, 4, 4}, null_at_2},
                                           {},
                                           {0, 0},
                                           {10, 10},
                                           {20, 20},
                                           {30, 30, 30},
                                           {40, 40, 40, 40}},
                                          cudf::test::iterators::nulls_at({0, 3, 6})}
                                        .release()
                                        ->view());
}

TYPED_TEST(TypedNestedLeadLagWindowTest, NumericListsWithDefaults)
{
  using T   = TypeParam;
  using lcw = cudf::test::lists_column_wrapper<T, int32_t>;

  auto null_at_2       = cudf::test::iterators::null_at(2);
  auto const input_col = lcw{{{0, 0},
                              {1, 1},
                              {2, 2},
                              {3, 3, 3},
                              {{4, 4, 4, 4}, null_at_2},
                              {5, 5, 5, 5, 5},
                              {0, 0},
                              {10, 10},
                              {20, 20},
                              {30, 30, 30},
                              {40, 40, 40, 40},
                              {{50, 50, 50, 50, 50}, null_at_2}},
                             null_at_2}
                           .release();

  auto const defaults_col =
    lcw{
      {
        {},
        {91, 91},
        {92, 92},
        {},  // null!
        {94, 94, 94},
        {95, 95},
        {},
        {91, 91},
        {92, 92},
        {},  // null!
        {94, 94, 94},
        {95, 95},
      },
    }
      .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_3_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lead_aggregation<cudf::rolling_aggregation>(3));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lead_3_output_col->view(),
                                      lcw{{{3, 3, 3},
                                           {{4, 4, 4, 4}, null_at_2},
                                           {5, 5, 5, 5, 5},
                                           {},
                                           {},
                                           {},
                                           {30, 30, 30},
                                           {40, 40, 40, 40},
                                           {{50, 50, 50, 50, 50}, null_at_2},
                                           {},
                                           {},
                                           {}},
                                          cudf::test::iterators::nulls_at({3, 4, 5, 9, 10, 11})}
                                        .release()
                                        ->view());

  auto lag_1_output_col =
    cudf::grouped_rolling_window(grouping_keys,
                                 input_col->view(),
                                 preceding,
                                 following,
                                 min_periods,
                                 *cudf::make_lag_aggregation<cudf::rolling_aggregation>(1));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lag_1_output_col->view(),
                                      lcw{{{},
                                           {0, 0},
                                           {1, 1},
                                           {2, 2},
                                           {3, 3, 3},
                                           {{4, 4, 4, 4}, null_at_2},
                                           {},
                                           {0, 0},
                                           {10, 10},
                                           {20, 20},
                                           {30, 30, 30},
                                           {40, 40, 40, 40}},
                                          cudf::test::iterators::nulls_at({0, 3, 6})}
                                        .release()
                                        ->view());
}

TYPED_TEST(TypedNestedLeadLagWindowTest, Structs)
{
  using T   = TypeParam;
  using lcw = cudf::test::lists_column_wrapper<T, int32_t>;

  auto null_at_2 = cudf::test::iterators::null_at(2);
  auto lists_col = lcw{{{0, 0},
                        {1, 1},
                        {2, 2},
                        {3, 3, 3},
                        {{4, 4, 4, 4}, null_at_2},
                        {5, 5, 5, 5, 5},
                        {0, 0},
                        {10, 10},
                        {20, 20},
                        {30, 30, 30},
                        {40, 40, 40, 40},
                        {{50, 50, 50, 50, 50}, null_at_2}},
                       null_at_2};

  auto strings_col = cudf::test::strings_column_wrapper{{"00",
                                                         "11",
                                                         "22",
                                                         "333",
                                                         "4444",
                                                         "55555",
                                                         "00",
                                                         "1010",
                                                         "2020",
                                                         "303030",
                                                         "40404040",
                                                         "5050505050"},
                                                        cudf::test::iterators::null_at(9)};

  auto structs_col = cudf::test::structs_column_wrapper{lists_col, strings_col}.release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  // Test LEAD().
  {
    auto lead_3_output_col =
      cudf::grouped_rolling_window(grouping_keys,
                                   structs_col->view(),
                                   preceding,
                                   following,
                                   min_periods,
                                   *cudf::make_lead_aggregation<cudf::rolling_aggregation>(3));
    auto expected_lists_col   = lcw{{{3, 3, 3},
                                     {{4, 4, 4, 4}, null_at_2},
                                     {5, 5, 5, 5, 5},
                                     {},
                                     {},
                                     {},
                                     {30, 30, 30},
                                     {40, 40, 40, 40},
                                     {{50, 50, 50, 50, 50}, null_at_2},
                                     {},
                                     {},
                                     {}},
                                  cudf::test::iterators::nulls_at({3, 4, 5, 9, 10, 11})};
    auto expected_strings_col = cudf::test::strings_column_wrapper{
      {"333", "4444", "55555", "", "", "", "", "40404040", "5050505050", "", "", ""},
      cudf::test::iterators::nulls_at({3, 4, 5, 6, 9, 10, 11})};

    auto expected_structs_col =
      cudf::test::structs_column_wrapper{{expected_lists_col, expected_strings_col},
                                         cudf::test::iterators::nulls_at({3, 4, 5, 9, 10, 11})}
        .release();

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lead_3_output_col->view(), expected_structs_col->view());
  }

  // Test LAG()
  {
    auto lag_1_output_col =
      cudf::grouped_rolling_window(grouping_keys,
                                   structs_col->view(),
                                   preceding,
                                   following,
                                   min_periods,
                                   *cudf::make_lag_aggregation<cudf::rolling_aggregation>(1));
    auto expected_lists_col = lcw{{{},  // null.
                                   {0, 0},
                                   {1, 1},
                                   {},  // null.
                                   {3, 3, 3},
                                   {{4, 4, 4, 4}, null_at_2},
                                   {},  // null.
                                   {0, 0},
                                   {10, 10},
                                   {20, 20},
                                   {30, 30, 30},
                                   {40, 40, 40, 40}},
                                  cudf::test::iterators::nulls_at({0, 3, 6})};
    auto expected_strings_col =
      cudf::test::strings_column_wrapper{{"",  // null.
                                          "00",
                                          "11",
                                          "22",
                                          "333",
                                          "4444",
                                          "",  // null.
                                          "00",
                                          "1010",
                                          "2020",
                                          "",  // null.
                                          "40404040"},
                                         cudf::test::iterators::nulls_at({0, 6, 10})};

    auto expected_structs_col =
      cudf::test::structs_column_wrapper{{expected_lists_col, expected_strings_col},
                                         cudf::test::iterators::nulls_at({0, 6})}
        .release();

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lag_1_output_col->view(), expected_structs_col->view());
  }
}

struct LeadLagNonFixedWidthTest : cudf::test::BaseFixture {};

TEST_F(LeadLagNonFixedWidthTest, StringsNoDefaults)
{
  auto input_col =
    cudf::test::strings_column_wrapper{{"",
                                        "A_1",
                                        "A_22",
                                        "A_333",
                                        "A_4444",
                                        "A_55555",
                                        "B_0",
                                        "",
                                        "B_22",
                                        "B_333",
                                        "B_4444",
                                        "B_55555"},
                                       cudf::test::iterators::nulls_at(std::vector{0, 7})}
      .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_2 = grouped_rolling_window(grouping_keys,
                                       input_col->view(),
                                       preceding,
                                       following,
                                       min_periods,
                                       *cudf::make_lead_aggregation<cudf::rolling_aggregation>(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    lead_2->view(),
    cudf::test::strings_column_wrapper{
      {"A_22", "A_333", "A_4444", "A_55555", "", "", "B_22", "B_333", "B_4444", "B_55555", "", ""},
      cudf::test::iterators::nulls_at(std::vector{4, 5, 10, 11})});

  auto lag_1 = grouped_rolling_window(grouping_keys,
                                      input_col->view(),
                                      preceding,
                                      following,
                                      min_periods,
                                      *cudf::make_lag_aggregation<cudf::rolling_aggregation>(1));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    lag_1->view(),
    cudf::test::strings_column_wrapper{
      {"", "", "A_1", "A_22", "A_333", "A_4444", "", "B_0", "", "B_22", "B_333", "B_4444"},
      cudf::test::iterators::nulls_at(std::vector{0, 1, 6, 8})});
}

TEST_F(LeadLagNonFixedWidthTest, StringsWithDefaults)
{
  auto input_col =
    cudf::test::strings_column_wrapper{{"",
                                        "A_1",
                                        "A_22",
                                        "A_333",
                                        "A_4444",
                                        "A_55555",
                                        "B_0",
                                        "",
                                        "B_22",
                                        "B_333",
                                        "B_4444",
                                        "B_55555"},
                                       cudf::test::iterators::nulls_at(std::vector{0, 7})}
      .release();

  auto defaults_col = cudf::test::strings_column_wrapper{"9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999"}
                        .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_2 = grouped_rolling_window(grouping_keys,
                                       input_col->view(),
                                       defaults_col->view(),
                                       preceding,
                                       following,
                                       min_periods,
                                       *cudf::make_lead_aggregation<cudf::rolling_aggregation>(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lead_2->view(),
                                      cudf::test::strings_column_wrapper{"A_22",
                                                                         "A_333",
                                                                         "A_4444",
                                                                         "A_55555",
                                                                         "9999",
                                                                         "9999",
                                                                         "B_22",
                                                                         "B_333",
                                                                         "B_4444",
                                                                         "B_55555",
                                                                         "9999",
                                                                         "9999"});

  auto lag_1 = grouped_rolling_window(grouping_keys,
                                      input_col->view(),
                                      defaults_col->view(),
                                      preceding,
                                      following,
                                      min_periods,
                                      *cudf::make_lag_aggregation<cudf::rolling_aggregation>(1));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    lag_1->view(),
    cudf::test::strings_column_wrapper{
      {"9999", "", "A_1", "A_22", "A_333", "A_4444", "9999", "B_0", "", "B_22", "B_333", "B_4444"},
      cudf::test::iterators::nulls_at(std::vector{1, 8})});
}

TEST_F(LeadLagNonFixedWidthTest, StringsWithDefaultsNoGroups)
{
  auto input_col =
    cudf::test::strings_column_wrapper{{"",
                                        "A_1",
                                        "A_22",
                                        "A_333",
                                        "A_4444",
                                        "A_55555",
                                        "B_0",
                                        "",
                                        "B_22",
                                        "B_333",
                                        "B_4444",
                                        "B_55555"},
                                       cudf::test::iterators::nulls_at(std::vector{0, 7})}
      .release();

  auto defaults_col = cudf::test::strings_column_wrapper{"9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999",
                                                         "9999"}
                        .release();

  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto lead_2 = grouped_rolling_window(grouping_keys,
                                       input_col->view(),
                                       defaults_col->view(),
                                       preceding,
                                       following,
                                       min_periods,
                                       *cudf::make_lead_aggregation<cudf::rolling_aggregation>(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    lead_2->view(),
    cudf::test::strings_column_wrapper{{"A_22",
                                        "A_333",
                                        "A_4444",
                                        "A_55555",
                                        "B_0",
                                        "",
                                        "B_22",
                                        "B_333",
                                        "B_4444",
                                        "B_55555",
                                        "9999",
                                        "9999"},
                                       cudf::test::iterators::null_at(5)});

  auto lag_1 = grouped_rolling_window(grouping_keys,
                                      input_col->view(),
                                      defaults_col->view(),
                                      preceding,
                                      following,
                                      min_periods,
                                      *cudf::make_lag_aggregation<cudf::rolling_aggregation>(1));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    lag_1->view(),
    cudf::test::strings_column_wrapper{{"9999",
                                        "",
                                        "A_1",
                                        "A_22",
                                        "A_333",
                                        "A_4444",
                                        "A_55555",
                                        "B_0",
                                        "",
                                        "B_22",
                                        "B_333",
                                        "B_4444"},
                                       cudf::test::iterators::nulls_at(std::vector{1, 8})});
}

TEST_F(LeadLagNonFixedWidthTest, Dictionary)
{
  auto input_col = cudf::test::dictionary_column_wrapper<std::string>({"",
                                                                       "A_1",
                                                                       "A_22",
                                                                       "A_333",
                                                                       "A_4444",
                                                                       "A_55555",
                                                                       "B_0",
                                                                       "",
                                                                       "B_22",
                                                                       "B_333",
                                                                       "B_4444",
                                                                       "B_55555"});
  auto const grouping_key =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  {
    auto agg = cudf::make_lead_aggregation<cudf::rolling_aggregation>(2);
    auto lead_2 =
      grouped_rolling_window(grouping_keys, input_col, preceding, following, min_periods, *agg);

    auto decoded  = cudf::dictionary::decode(lead_2->view());
    auto expected = cudf::test::strings_column_wrapper(
      {"A_22", "A_333", "A_4444", "A_55555", "", "", "B_22", "B_333", "B_4444", "B_55555", "", ""},
      {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(decoded->view(), expected);
  }

  {
    auto agg = cudf::make_lag_aggregation<cudf::rolling_aggregation>(1);
    auto lag_1 =
      grouped_rolling_window(grouping_keys, input_col, preceding, following, min_periods, *agg);

    auto decoded  = cudf::dictionary::decode(lag_1->view());
    auto expected = cudf::test::strings_column_wrapper(
      {"", "", "A_1", "A_22", "A_333", "A_4444", "", "B_0", "", "B_22", "B_333", "B_4444"},
      {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(decoded->view(), expected);
  }
}
