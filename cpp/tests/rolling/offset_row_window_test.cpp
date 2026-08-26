/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/rolling.hpp>

template <typename T>
using fwcw = cudf::test::fixed_width_column_wrapper<T>;
template <typename T>
using decimals_column = cudf::test::fixed_point_column_wrapper<T>;
using ints_column     = fwcw<int32_t>;
using bigints_column  = fwcw<int64_t>;
using strings_column  = cudf::test::strings_column_wrapper;
using lists_column    = cudf::test::lists_column_wrapper<int32_t>;
using column_ptr      = std::unique_ptr<cudf::column>;
using cudf::test::iterators::all_nulls;
using cudf::test::iterators::no_nulls;
using cudf::test::iterators::nulls_at;

auto constexpr null = int32_t{0};  // NULL representation for int32_t;

// clang-tidy doesn't think std::transform can handle a
// cuda::constant_iterator, so this is a workaround that uses nulls_at
// instead of no_nulls
auto no_nulls_list() { return nulls_at({}); }

struct OffsetRowWindowTest : public cudf::test::BaseFixture {
  struct rolling_runner {
    cudf::window_bounds _preceding, _following;
    cudf::size_type _min_periods;
    bool _grouped = true;
    ints_column const _keys;    // {0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
    ints_column const _values;  // {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    rolling_runner(cudf::window_bounds const& preceding,
                   cudf::window_bounds const& following,
                   cudf::size_type min_periods_ = 1)
      : _preceding{preceding},
        _following{following},
        _min_periods{min_periods_},
        _keys{0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
        _values{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    {
    }

    rolling_runner& min_periods(cudf::size_type min_periods_)
    {
      _min_periods = min_periods_;
      return *this;
    }

    rolling_runner& grouped(bool grouped_)
    {
      _grouped = grouped_;
      return *this;
    }

    std::unique_ptr<cudf::column> operator()(cudf::rolling_aggregation const& agg) const
    {
      auto const grouping_keys =
        _grouped ? std::vector<cudf::column_view>{_keys} : std::vector<cudf::column_view>{};
      return cudf::grouped_rolling_window(
        cudf::table_view{grouping_keys}, _values, _preceding, _following, _min_periods, agg);
    }
  };
};

auto const AGG_COUNT_NON_NULL =
  cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::EXCLUDE);
auto const AGG_COUNT_ALL =
  cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE);
auto const AGG_MIN          = cudf::make_min_aggregation<cudf::rolling_aggregation>();
auto const AGG_MAX          = cudf::make_max_aggregation<cudf::rolling_aggregation>();
auto const AGG_SUM          = cudf::make_sum_aggregation<cudf::rolling_aggregation>();
auto const AGG_COLLECT_LIST = cudf::make_collect_list_aggregation<cudf::rolling_aggregation>();

TEST_F(OffsetRowWindowTest, OffsetRowWindow_Grouped_3_to_Minus_1)
{
  auto const preceding = cudf::window_bounds::get(3);
  auto const following = cudf::window_bounds::get(-1);
  auto run_rolling     = rolling_runner{preceding, following};
  run_rolling.min_periods(1).grouped(true);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_NON_NULL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 0, 1, 2, 2}, nulls_at({0, 6})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_ALL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 0, 1, 2, 2}, nulls_at({0, 6})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_MIN), ints_column{{null, 0, 0, 1, 2, 3, null, 6, 6, 7}, nulls_at({0, 6})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_MAX), ints_column{{null, 0, 1, 2, 3, 4, null, 6, 7, 8}, nulls_at({0, 6})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_SUM),
    bigints_column{{null, 0, 1, 3, 5, 7, null, 6, 13, 15}, nulls_at({0, 6})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{}, {0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {}, {6}, {6, 7}, {7, 8}},
                 nulls_at({0, 6})});

  run_rolling.min_periods(0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_NON_NULL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 0, 1, 2, 2}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_ALL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 0, 1, 2, 2}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{}, {0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {}, {6}, {6, 7}, {7, 8}}, no_nulls()});
}

TEST_F(OffsetRowWindowTest, OffsetRowWindow_Ungrouped_3_to_Minus_1)
{
  auto const preceding = cudf::window_bounds::get(3);
  auto const following = cudf::window_bounds::get(-1);
  auto run_rolling     = rolling_runner{preceding, following};
  run_rolling.min_periods(1).grouped(false);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_NON_NULL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 2, 2, 2, 2}, nulls_at({0})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_ALL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 2, 2, 2, 2}, nulls_at({0})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_MIN),
                                 ints_column{{null, 0, 0, 1, 2, 3, 4, 5, 6, 7}, nulls_at({0})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_MAX),
                                 ints_column{{null, 0, 1, 2, 3, 4, 5, 6, 7, 8}, nulls_at({0})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_SUM), bigints_column{{null, 0, 1, 3, 5, 7, 9, 11, 13, 15}, nulls_at({0})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{}, {0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}},
                 nulls_at({0})});

  run_rolling.min_periods(0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_NON_NULL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 2, 2, 2, 2}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_ALL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 2, 2, 2, 2}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{}, {0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}},
                 no_nulls()});
}

TEST_F(OffsetRowWindowTest, OffsetRowWindow_Grouped_0_to_2)
{
  auto const preceding = cudf::window_bounds::get(0);
  auto const following = cudf::window_bounds::get(2);
  auto run_rolling     = rolling_runner{preceding, following};
  run_rolling.min_periods(1).grouped(true);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COUNT_NON_NULL),
    ints_column{{2, 2, 2, 2, 1, null, 2, 2, 1, null}, nulls_at({5, 9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COUNT_ALL),
    ints_column{{2, 2, 2, 2, 1, null, 2, 2, 1, null}, nulls_at({5, 9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_MIN), ints_column{{1, 2, 3, 4, 5, null, 7, 8, 9, null}, nulls_at({5, 9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_MAX), ints_column{{2, 3, 4, 5, 5, null, 8, 9, 9, null}, nulls_at({5, 9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_SUM),
    bigints_column{{3, 5, 7, 9, 5, null, 15, 17, 9, null}, nulls_at({5, 9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5}, {}, {7, 8}, {8, 9}, {9}, {}},
                 nulls_at({5, 9})});

  run_rolling.min_periods(0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_NON_NULL),
                                 ints_column{{2, 2, 2, 2, 1, 0, 2, 2, 1, 0}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_ALL),
                                 ints_column{{2, 2, 2, 2, 1, 0, 2, 2, 1, 0}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5}, {}, {7, 8}, {8, 9}, {9}, {}},
                 no_nulls_list()});
}

TEST_F(OffsetRowWindowTest, OffsetRowWindow_Ungrouped_0_to_2)
{
  auto const preceding = cudf::window_bounds::get(0);
  auto const following = cudf::window_bounds::get(2);
  auto run_rolling     = rolling_runner{preceding, following};
  run_rolling.min_periods(1).grouped(false);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_NON_NULL),
                                 ints_column{{2, 2, 2, 2, 2, 2, 2, 2, 1, null}, nulls_at({9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_ALL),
                                 ints_column{{2, 2, 2, 2, 2, 2, 2, 2, 1, null}, nulls_at({9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_MIN),
                                 ints_column{{1, 2, 3, 4, 5, 6, 7, 8, 9, null}, nulls_at({9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_MAX),
                                 ints_column{{2, 3, 4, 5, 6, 7, 8, 9, 9, null}, nulls_at({9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_SUM), bigints_column{{3, 5, 7, 9, 11, 13, 15, 17, 9, null}, nulls_at({9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9}, {}},
                 nulls_at({9})});

  run_rolling.min_periods(0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_NON_NULL),
                                 ints_column{{2, 2, 2, 2, 2, 2, 2, 2, 1, 0}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_ALL),
                                 ints_column{{2, 2, 2, 2, 2, 2, 2, 2, 1, 0}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9}, {}},
                 no_nulls_list()});
}

// To test that preceding bounds are clamped correctly at group boundaries.
TEST_F(OffsetRowWindowTest, TestNegativeBoundsClamp)
{
  auto const grp_iter =
    cudf::detail::make_counting_transform_iterator(cudf::size_type{0}, [](auto const& i) {
      return i / 10;  // 0-9 in the first group, 10-19 in the second, etc.
    });
  auto const agg_iter = cuda::make_constant_iterator(1);

  auto const grp = ints_column(grp_iter, grp_iter + 30);
  auto const agg = ints_column(agg_iter, agg_iter + 30);

  auto const min_periods = 0;
  auto const rolling_sum = [&](auto const preceding, auto const following) {
    return cudf::grouped_rolling_window(
      cudf::table_view{{grp}}, agg, preceding, following, min_periods, *AGG_SUM);
  };

  // Testing negative preceding.
  for (auto const preceding : {0, -1, -2, -5, -10, -20, -50}) {
    auto const results      = rolling_sum(preceding, 100);
    auto const expected_fun = [&](auto const& i) {
      assert(preceding < 1);
      auto const index_in_group = i % 10;
      auto const start          = std::min(-(preceding - 1) + index_in_group, 10);
      return int64_t{10 - start};
    };
    auto const expected_iter =
      cudf::detail::make_counting_transform_iterator(cudf::size_type{0}, expected_fun);
    auto const expected = bigints_column(expected_iter, expected_iter + 30, no_nulls());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  // Testing negative following.
  for (auto const following : {-1, -2, -5, -10, -20, -50}) {
    auto const results      = rolling_sum(100, following);
    auto const expected_fun = [&](auto const& i) {
      assert(following < 0);
      auto const index_in_group = i % 10;
      auto const end            = std::max(index_in_group + following, -1);
      return int64_t{end + 1};
    };
    auto const expected_iter =
      cudf::detail::make_counting_transform_iterator(cudf::size_type{0}, expected_fun);
    auto const expected = bigints_column(expected_iter, expected_iter + 30, no_nulls());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(OffsetRowWindowTest, CheckGroupBoundaries)
{
  auto grp_iter =
    cudf::detail::make_counting_transform_iterator(cudf::size_type{0}, [](auto const& i) {
      if (i < 10) return 1;
      if (i < 20) return 2;
      return 3;
    });
  auto const grp = ints_column(grp_iter, grp_iter + 30);
  auto const agg = ints_column(grp_iter, grp_iter + 30);
  {
    auto const results =
      cudf::grouped_rolling_window(cudf::table_view{{grp}},
                                   agg,
                                   -80,
                                   100,
                                   1,
                                   *cudf::make_max_aggregation<cudf::rolling_aggregation>());
    auto const null_iter = cuda::make_constant_iterator<int32_t>(null);
    auto const expected  = ints_column(null_iter, null_iter + 30, all_nulls());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
  {
    auto const results =
      cudf::grouped_rolling_window(cudf::table_view{{grp}},
                                   agg,
                                   -1,
                                   4,
                                   1,
                                   *cudf::make_min_aggregation<cudf::rolling_aggregation>());
    auto const expected =
      ints_column{{1, 1, 1, 1,    1,    1, 1, 1, null, null, 2, 2, 2, 2,    2,
                   2, 2, 2, null, null, 3, 3, 3, 3,    3,    3, 3, 3, null, null},
                  nulls_at({8, 9, 18, 19, 28, 29})};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}
