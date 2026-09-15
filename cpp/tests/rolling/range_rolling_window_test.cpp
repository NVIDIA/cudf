/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/iterator>

#include <src/rolling/detail/range_window_bounds.hpp>

#include <utility>
#include <vector>

template <typename T, typename R = int32_t>
using fwcw = cudf::test::fixed_width_column_wrapper<T>;

using int_col  = fwcw<int32_t>;
using size_col = fwcw<cudf::size_type>;

template <typename T, typename R = typename T::rep>
using time_col = fwcw<T, R>;

using lists_col = cudf::test::lists_column_wrapper<int32_t>;

template <typename ScalarT>
struct window_exec {
 public:
  window_exec(cudf::column_view gby,
              cudf::column_view oby,
              cudf::order ordering,
              cudf::column_view agg,
              ScalarT preceding_scalar,
              ScalarT following_scalar,
              cudf::size_type min_periods = 1)
    : gby_column(std::move(gby)),
      oby_column(std::move(oby)),
      order(ordering),
      agg_column(std::move(agg)),
      preceding(std::move(preceding_scalar)),
      following(std::move(following_scalar)),
      min_periods(min_periods)
  {
  }

  cudf::size_type num_rows() { return gby_column.size(); }

  std::unique_ptr<cudf::column> operator()(
    std::unique_ptr<cudf::rolling_aggregation> const& agg) const
  {
    auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{gby_column}};

    return cudf::grouped_range_rolling_window(grouping_keys,
                                              oby_column,
                                              order,
                                              agg_column,
                                              cudf::range_window_bounds::get(preceding),
                                              cudf::range_window_bounds::get(following),
                                              min_periods,
                                              *agg);
  }

 private:
  cudf::column_view gby_column;  // Groupby column.
  cudf::column_view oby_column;  // Orderby column.
  cudf::order order;             // Ordering for `oby_column`.
  cudf::column_view agg_column;  // Aggregation column.
  ScalarT preceding;             // Preceding window scalar.
  ScalarT following;             // Following window scalar.
  cudf::size_type min_periods = 1;
};  // struct window_exec;

struct RangeRollingTest : public cudf::test::BaseFixture {};

template <typename T>
struct TypedTimeRangeRollingTest : RangeRollingTest {};

TYPED_TEST_SUITE(TypedTimeRangeRollingTest, cudf::test::TimestampTypes);

template <typename WindowExecT>
void verify_results_for_ascending(WindowExecT exec)
{
  auto const n_rows       = exec.num_rows();
  auto const all_valid    = cuda::make_constant_iterator<bool>(true);
  auto const all_invalid  = cuda::make_constant_iterator<bool>(false);
  auto const last_invalid = cuda::transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0}, [&n_rows](auto i) { return i != (n_rows - 1); });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE))
      ->view(),
    size_col{{1, 2, 2, 3, 2, 3, 3, 4, 4, 1}, all_valid});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_count_aggregation<cudf::rolling_aggregation>())->view(),
    size_col{{1, 2, 2, 3, 2, 3, 3, 4, 4, 0}, all_valid});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_sum_aggregation<cudf::rolling_aggregation>())->view(),
    fwcw<int64_t>{{0, 12, 12, 12, 8, 17, 17, 18, 18, 1}, last_invalid});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_min_aggregation<cudf::rolling_aggregation>())->view(),
    int_col{{0, 4, 4, 2, 2, 3, 3, 1, 1, 1}, last_invalid});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_max_aggregation<cudf::rolling_aggregation>())->view(),
    int_col{{0, 8, 8, 6, 6, 9, 9, 9, 9, 1}, last_invalid});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_mean_aggregation<cudf::rolling_aggregation>())->view(),
    fwcw<double>{{0.0, 6.0, 6.0, 4.0, 4.0, 17.0 / 3, 17.0 / 3, 4.5, 4.5, 1.0}, last_invalid});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    exec(cudf::make_collect_list_aggregation<cudf::rolling_aggregation>())->view(),
    lists_col{{{0},
               {8, 4},
               {8, 4},
               {4, 6, 2},
               {6, 2},
               {9, 3, 5},
               {9, 3, 5},
               {9, 3, 5, 1},
               {9, 3, 5, 1},
               {{0}, all_invalid}},
              all_valid});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    exec(cudf::make_collect_list_aggregation<cudf::rolling_aggregation>(cudf::null_policy::EXCLUDE))
      ->view(),
    lists_col{{{0},
               {8, 4},
               {8, 4},
               {4, 6, 2},
               {6, 2},
               {9, 3, 5},
               {9, 3, 5},
               {9, 3, 5, 1},
               {9, 3, 5, 1},
               {}},
              all_valid});
}

TYPED_TEST(TypedTimeRangeRollingTest, TimestampASC)
{
  // Confirm that timestamp columns can be used in range queries
  // at all resolutions, given the right duration column type.

  using TimeT     = TypeParam;
  using DurationT = cudf::detail::range_type<TimeT>;
  using time_col  = fwcw<TimeT>;

  // clang-format off
  auto gby_column  = int_col { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto agg_column  = int_col {{0, 8, 4, 6, 2, 9, 3, 5, 1, 7},
                              {true, true, true, true, true, true, true, true, true, false}};
  auto time_column = time_col{ 1, 5, 6, 8, 9, 2, 2, 3, 4, 9};
  // clang-format on

  auto exec =
    window_exec(gby_column,
                time_column,
                cudf::order::ASCENDING,
                agg_column,
                cudf::duration_scalar<DurationT>{DurationT{2}, true},   // 2 "durations" preceding.
                cudf::duration_scalar<DurationT>{DurationT{1}, true});  // 1 "durations" following.

  verify_results_for_ascending(exec);
}

template <typename WindowExecT>
void verify_results_for_descending(WindowExecT exec)
{
  auto const all_valid     = cuda::make_constant_iterator<bool>(true);
  auto const all_invalid   = cuda::make_constant_iterator<bool>(false);
  auto const first_invalid = cuda::transform_iterator(cuda::counting_iterator<cudf::size_type>{0},
                                                      [](auto i) { return i != 0; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE))
      ->view(),
    size_col{{1, 4, 4, 3, 3, 2, 3, 2, 2, 1}, all_valid});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_count_aggregation<cudf::rolling_aggregation>())->view(),
    size_col{{0, 4, 4, 3, 3, 2, 3, 2, 2, 1}, all_valid});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_sum_aggregation<cudf::rolling_aggregation>())->view(),
    fwcw<int64_t>{{1, 18, 18, 17, 17, 8, 12, 12, 12, 0}, first_invalid});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_min_aggregation<cudf::rolling_aggregation>())->view(),
    int_col{{1, 1, 1, 3, 3, 2, 2, 4, 4, 0}, first_invalid});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_max_aggregation<cudf::rolling_aggregation>())->view(),
    int_col{{1, 9, 9, 9, 9, 6, 6, 8, 8, 0}, first_invalid});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    exec(cudf::make_mean_aggregation<cudf::rolling_aggregation>())->view(),
    fwcw<double>{{1.0, 4.5, 4.5, 17.0 / 3, 17.0 / 3, 4.0, 4.0, 6.0, 6.0, 0.0}, first_invalid});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    exec(cudf::make_collect_list_aggregation<cudf::rolling_aggregation>())->view(),
    lists_col{{{{0}, all_invalid},
               {1, 5, 3, 9},
               {1, 5, 3, 9},
               {5, 3, 9},
               {5, 3, 9},
               {2, 6},
               {2, 6, 4},
               {4, 8},
               {4, 8},
               {0}},
              all_valid});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    exec(cudf::make_collect_list_aggregation<cudf::rolling_aggregation>(cudf::null_policy::EXCLUDE))
      ->view(),
    lists_col{{{},
               {1, 5, 3, 9},
               {1, 5, 3, 9},
               {5, 3, 9},
               {5, 3, 9},
               {2, 6},
               {2, 6, 4},
               {4, 8},
               {4, 8},
               {0}},
              all_valid});
}

TYPED_TEST(TypedTimeRangeRollingTest, TimestampDESC)
{
  // Confirm that timestamp columns can be used in range queries
  // at all resolutions, given the right duration column type.
  using TimeT     = TypeParam;
  using DurationT = cudf::detail::range_type<TimeT>;
  using time_col  = fwcw<TimeT>;

  // clang-format off
  auto gby_column  = int_col { 5, 5, 5, 5, 5, 1, 1, 1, 1, 1};
  auto agg_column  = int_col {{7, 1, 5, 3, 9, 2, 6, 4, 8, 0},
                              {false, true, true, true, true, true, true, true, true, true}};
  auto time_column = time_col{ 9, 4, 3, 2, 2, 9, 8, 6, 5, 1};
  // clang-format on

  auto exec =
    window_exec(gby_column,
                time_column,
                cudf::order::DESCENDING,
                agg_column,
                cudf::duration_scalar<DurationT>{DurationT{1}, true},   // 1 "durations" preceding.
                cudf::duration_scalar<DurationT>{DurationT{2}, true});  // 2 "durations" following.

  verify_results_for_descending(exec);
}

template <typename T>
struct TypedIntegralRangeRollingTest : RangeRollingTest {};

TYPED_TEST_SUITE(TypedIntegralRangeRollingTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(TypedIntegralRangeRollingTest, OrderByASC)
{
  // Confirm that integral ranges work with integral orderby columns,
  // in ascending order.
  using T = TypeParam;

  // clang-format off
  auto gby_column = int_col { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto agg_column = int_col {{0, 8, 4, 6, 2, 9, 3, 5, 1, 7},
                             {true, true, true, true, true, true, true, true, true, false}};
  auto oby_column = fwcw<T>{  1, 5, 6, 8, 9, 2, 2, 3, 4, 9};
  // clang-format on

  auto exec = window_exec(gby_column,
                          oby_column,
                          cudf::order::ASCENDING,
                          agg_column,
                          cudf::numeric_scalar<T>(2),   // 2 preceding.
                          cudf::numeric_scalar<T>(1));  // 1 following.

  verify_results_for_ascending(exec);
}

TYPED_TEST(TypedIntegralRangeRollingTest, OrderByDesc)
{
  // Confirm that integral ranges work with integral orderby columns,
  // in descending order.
  using T = TypeParam;

  // clang-format off
  auto gby_column  = int_col { 5, 5, 5, 5, 5, 1, 1, 1, 1, 1};
  auto agg_column  = int_col {{7, 1, 5, 3, 9, 2, 6, 4, 8, 0},
                              {false, true, true, true, true, true, true, true, true, true}};
  auto oby_column  = fwcw<T>{  9, 4, 3, 2, 2, 9, 8, 6, 5, 1};
  // clang-format on

  auto exec = window_exec(gby_column,
                          oby_column,
                          cudf::order::DESCENDING,
                          agg_column,
                          cudf::numeric_scalar<T>(1),   // 1 preceding.
                          cudf::numeric_scalar<T>(2));  // 2 following.

  verify_results_for_descending(exec);
}

template <typename T>
struct TypedRangeRollingNullsTest : public RangeRollingTest {};

using TypesUnderTest = cudf::test::IntegralTypesNotBool;

TYPED_TEST_SUITE(TypedRangeRollingNullsTest, TypesUnderTest);

template <typename T>
auto do_count_over_window(cudf::column_view grouping_col,
                          cudf::column_view order_by,
                          cudf::order order,
                          cudf::column_view aggregation_col,
                          cudf::range_window_bounds&& preceding =
                            cudf::range_window_bounds::get(cudf::numeric_scalar<T>{T{1}, true}),
                          cudf::range_window_bounds&& following =
                            cudf::range_window_bounds::get(cudf::numeric_scalar<T>{T{1}, true}))
{
  auto const min_periods   = cudf::size_type{1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_col}};

  return cudf::grouped_range_rolling_window(
    grouping_keys,
    order_by,
    order,
    aggregation_col,
    std::move(preceding),
    std::move(following),
    min_periods,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>());
}

TYPED_TEST(TypedRangeRollingNullsTest, CountSingleGroupOrderByASCNullsFirst)
{
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Aggregation column.
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  // OrderBy column.
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const output = do_count_over_window<T>(grp_col, oby_col, cudf::order::ASCENDING, agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {4, 4, 4, 4, 1, 2, 2, 3, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, CountSingleGroupOrderByASCNullsLast)
{
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Aggregation column.
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  // OrderBy column.
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {1, 1, 1, 1, 1, 1, 0, 0, 0, 0}};

  auto const output = do_count_over_window<T>(grp_col, oby_col, cudf::order::ASCENDING, agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {2, 3, 3, 3, 2, 1, 4, 4, 4, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, CountMultiGroupOrderByASCNullsFirst)
{
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  // Aggregation column.
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // OrderBy column.
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{1, 2, 2, 1, 2, 1, 2, 3, 4, 5},
                                                                 {0, 0, 0, 1, 1, 0, 0, 1, 1, 1}};

  auto const output = do_count_over_window<T>(grp_col, oby_col, cudf::order::ASCENDING, agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {3, 3, 3, 2, 2, 2, 2, 2, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, CountMultiGroupOrderByASCNullsLast)
{
  using T = int32_t;

  // Groupby column.
  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  // Aggregation column.
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // OrderBy column.
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{
    {1, 2, 2, 1, 3, 1, 2, 3, 4, 5},
    {true, true, true, false, false, true, true, true, false, false}};

  auto const output = do_count_over_window<T>(grp_col, oby_col, cudf::order::ASCENDING, agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {3, 3, 3, 2, 2, 2, 3, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, CountSingleGroupOrderByDESCNullsFirst)
{
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Aggregation column.
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  // OrderBy column.
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
                                                                 {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const output = do_count_over_window<T>(grp_col, oby_col, cudf::order::DESCENDING, agg_col);
  ;

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {4, 4, 4, 4, 1, 2, 2, 3, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, CountSingleGroupOrderByDESCNullsLast)
{
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Aggregation column.
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  // OrderBy column.
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
                                                                 {1, 1, 1, 1, 1, 1, 0, 0, 0, 0}};

  auto const output = do_count_over_window<T>(grp_col, oby_col, cudf::order::DESCENDING, agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {2, 3, 3, 3, 2, 1, 4, 4, 4, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, CountMultiGroupOrderByDESCNullsFirst)
{
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  // Aggregation column.
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // OrderBy column.
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{4, 3, 2, 1, 0, 9, 8, 7, 6, 5},
                                                                 {0, 0, 0, 1, 1, 0, 0, 1, 1, 1}};

  auto const output = do_count_over_window<T>(grp_col, oby_col, cudf::order::DESCENDING, agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {3, 3, 3, 2, 2, 2, 2, 2, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, CountMultiGroupOrderByDESCNullsLast)
{
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  // Aggregation column.
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // OrderBy column.
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{4, 3, 2, 1, 0, 9, 8, 7, 6, 5},
                                                                 {1, 1, 1, 0, 0, 1, 1, 1, 0, 0}};

  auto const output = do_count_over_window<T>(grp_col, oby_col, cudf::order::DESCENDING, agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {2, 3, 2, 2, 2, 2, 3, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, CountSingleGroupAllNullOrderBys)
{
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Aggregation column.
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  // OrderBy column.
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

  auto const output = do_count_over_window<T>(grp_col, oby_col, cudf::order::ASCENDING, agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {9, 9, 9, 9, 9, 9, 9, 9, 9, 9}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, CountMultiGroupAllNullOrderBys)
{
  using T = TypeParam;

  // Groupby column.
  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  // Aggregation column.
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  // OrderBy column.
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};

  auto const output = do_count_over_window<T>(grp_col, oby_col, cudf::order::ASCENDING, agg_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {2, 3, 3, 3, 2, 4, 4, 4, 4, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, UnboundedPrecedingWindowSingleGroupOrderByASCNullsFirst)
{
  using T = TypeParam;

  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const output = do_count_over_window<T>(
    grp_col,
    oby_col,
    cudf::order::ASCENDING,
    agg_col,
    cudf::range_window_bounds::unbounded(cudf::data_type{cudf::type_to_id<T>()}),
    cudf::range_window_bounds::get(cudf::numeric_scalar<T>{1, true}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {4, 4, 4, 4, 5, 6, 7, 8, 9, 9}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}

TYPED_TEST(TypedRangeRollingNullsTest, UnboundedFollowingWindowSingleGroupOrderByASCNullsFirst)
{
  using T = TypeParam;

  auto const grp_col = cudf::test::fixed_width_column_wrapper<T>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto const agg_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  auto const oby_col = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                 {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const output = do_count_over_window<T>(
    grp_col,
    oby_col,
    cudf::order::ASCENDING,
    agg_col,
    cudf::range_window_bounds::get(cudf::numeric_scalar<T>{1, true}),
    cudf::range_window_bounds::unbounded(cudf::data_type{cudf::type_to_id<T>()}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output->view(),
                                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                                   {9, 9, 9, 9, 5, 5, 4, 4, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
}
