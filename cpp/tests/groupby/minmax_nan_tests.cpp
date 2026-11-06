/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <cmath>
#include <limits>
#include <vector>

// min/max follow the IEEE 754-2019 minimum/maximum operations for floating-point values:
// NaN propagates and -0 is less than +0. argmin/argmax select the first NaN.

template <typename T>
struct GroupbyMinMaxNaNTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupbyMinMaxNaNTest, cudf::test::FloatingPointTypes);

TYPED_TEST(GroupbyMinMaxNaNTest, MinMaxArgMinArgMax)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();
  using keys_col     = cudf::test::fixed_width_column_wrapper<int32_t>;
  using vals_col     = cudf::test::fixed_width_column_wrapper<T>;
  using idx_col      = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  // groups: [1, NaN, 3], [NaN, 1, 3], [NaN, NaN], [1, 3, 2], [null(NaN), 5]
  auto const keys = keys_col{0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4};
  auto const vals =
    vals_col{{T{1}, NaN, T{3}, NaN, T{1}, T{3}, NaN, NaN, T{1}, T{3}, T{2}, NaN, T{5}},
             {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1}};
  auto const expect_keys = keys_col{0, 1, 2, 3, 4};

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  vals_col{NaN, NaN, NaN, T{1}, T{5}},
                  cudf::make_min_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  vals_col{NaN, NaN, NaN, T{3}, T{5}},
                  cudf::make_max_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  idx_col{1, 3, 6, 8, 12},
                  cudf::make_argmin_aggregation<cudf::groupby_aggregation>(),
                  cudf::null_policy::EXCLUDE,
                  cudf::sorted::NO,
                  {},
                  {},
                  cudf::sorted::NO,
                  test_streaming::NO);
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  idx_col{1, 3, 6, 9, 12},
                  cudf::make_argmax_aggregation<cudf::groupby_aggregation>(),
                  cudf::null_policy::EXCLUDE,
                  cudf::sorted::NO,
                  {},
                  {},
                  cudf::sorted::NO,
                  test_streaming::NO);
}

TYPED_TEST(GroupbyMinMaxNaNTest, Large)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();

  // Rows are interleaved across 4 groups so each group is updated from many threads
  cudf::size_type constexpr size = 100'000;
  std::vector<int32_t> keys(size);
  std::vector<T> values(size);
  for (cudf::size_type i = 0; i < size; ++i) {
    keys[i]   = i % 4;
    values[i] = static_cast<T>(i);
  }
  // groups 1 and 2 contain NaNs; groups 0 and 3 do not
  for (auto row : {50'001, 70'002, 90'002}) {
    values[row] = NaN;
  }
  auto const keys_col = cudf::test::fixed_width_column_wrapper<int32_t>(keys.begin(), keys.end());
  auto const vals_col = cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end());
  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
  using vals_wrapper     = cudf::test::fixed_width_column_wrapper<T>;
  using idx_wrapper      = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  test_single_agg(keys_col,
                  vals_col,
                  expect_keys,
                  vals_wrapper{T{0}, NaN, NaN, T{3}},
                  cudf::make_min_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys_col,
                  vals_col,
                  expect_keys,
                  vals_wrapper{T{99'996}, NaN, NaN, T{99'999}},
                  cudf::make_max_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys_col,
                  vals_col,
                  expect_keys,
                  idx_wrapper{0, 50'001, 70'002, 3},
                  cudf::make_argmin_aggregation<cudf::groupby_aggregation>(),
                  cudf::null_policy::EXCLUDE,
                  cudf::sorted::NO,
                  {},
                  {},
                  cudf::sorted::NO,
                  test_streaming::NO);
  test_single_agg(keys_col,
                  vals_col,
                  expect_keys,
                  idx_wrapper{99'996, 50'001, 70'002, 99'999},
                  cudf::make_argmax_aggregation<cudf::groupby_aggregation>(),
                  cudf::null_policy::EXCLUDE,
                  cudf::sorted::NO,
                  {},
                  {},
                  cudf::sorted::NO,
                  test_streaming::NO);
}

TYPED_TEST(GroupbyMinMaxNaNTest, SignedZeros)
{
  using T = TypeParam;

  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 1, 1, 1};
  auto const vals =
    cudf::test::fixed_width_column_wrapper<T>{T{0}, -T{0}, T{0}, -T{0}, T{0}, -T{0}};

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = vals;
  requests[0].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  auto gb_obj                       = cudf::groupby::groupby(cudf::table_view({keys}));
  auto const [result_keys, results] = gb_obj.aggregate(requests);

  auto const order  = cudf::sorted_order(result_keys->view());
  auto const sorted = cudf::gather(
    cudf::table_view({results[0].results[0]->view(), results[0].results[1]->view()}), *order);
  auto const mins = cudf::test::to_host<T>(sorted->get_column(0).view()).first;
  auto const maxs = cudf::test::to_host<T>(sorted->get_column(1).view()).first;
  for (std::size_t i = 0; i < mins.size(); ++i) {
    EXPECT_EQ(mins[i], T{0});
    EXPECT_TRUE(std::signbit(mins[i]));
    EXPECT_EQ(maxs[i], T{0});
    EXPECT_FALSE(std::signbit(maxs[i]));
  }
}

TYPED_TEST(GroupbyMinMaxNaNTest, Scan)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();
  using keys_col     = cudf::test::fixed_width_column_wrapper<int32_t>;
  using vals_col     = cudf::test::fixed_width_column_wrapper<T>;

  auto const keys        = keys_col{0, 0, 0, 0, 1, 1, 1};
  auto const vals        = vals_col{T{1}, NaN, T{3}, T{0}, T{2}, T{1}, T{3}};
  auto const expect_keys = keys_col{0, 0, 0, 0, 1, 1, 1};

  test_single_scan(keys,
                   vals,
                   expect_keys,
                   vals_col{T{1}, NaN, NaN, NaN, T{2}, T{1}, T{1}},
                   cudf::make_min_aggregation<cudf::groupby_scan_aggregation>());
  test_single_scan(keys,
                   vals,
                   expect_keys,
                   vals_col{T{1}, NaN, NaN, NaN, T{2}, T{2}, T{3}},
                   cudf::make_max_aggregation<cudf::groupby_scan_aggregation>());
}
