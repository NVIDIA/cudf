/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

#include <limits>
#include <vector>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_correlation_test : public cudf::test::BaseFixture {};

using supported_types =
  cudf::test::RemoveIf<cudf::test::ContainedIn<cudf::test::Types<bool>>, cudf::test::NumericTypes>;

TYPED_TEST_SUITE(groupby_correlation_test, supported_types);
using K = int32_t;

using groupby_correlation_mixed_test = groupby_correlation_test<int32_t>;

TEST_F(groupby_correlation_mixed_test, CovarianceCorrelationAndFirstValue)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{2, 1, 2, 1, 2, 1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> first{{1, 0, 2, 2, 3, 4, 99}, null_at(6)};
  cudf::test::fixed_width_column_wrapper<int32_t> second{2, 0, 4, 4, 6, 8, 99};
  cudf::test::structs_column_wrapper values{{first, second}};
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(
    cudf::make_covariance_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(
    cudf::make_correlation_aggregation<cudf::groupby_aggregation>(cudf::correlation_type::PEARSON));
  requests[0].aggregations.push_back(
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
  cudf::groupby::groupby gb(cudf::table_view{{keys}});
  auto const [result_keys, results] = gb.aggregate(requests);
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].results.size(), 3);
  auto const order  = cudf::sorted_order(result_keys->view());
  auto const output = cudf::gather(cudf::table_view{{result_keys->view().column(0),
                                                     *results[0].results[0],
                                                     *results[0].results[1],
                                                     *results[0].results[2]}},
                                   *order);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{1, 2};
  cudf::test::fixed_width_column_wrapper<double> expected_covariance{8.0, 2.0};
  cudf::test::fixed_width_column_wrapper<double> expected_correlation{1.0, 1.0};
  cudf::test::fixed_width_column_wrapper<int32_t> first_values{0, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> second_values{0, 2};
  cudf::test::structs_column_wrapper expected_first{{first_values, second_values}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, output->get_column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_covariance, output->get_column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_correlation, output->get_column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_first, output->get_column(3));
}

TYPED_TEST(groupby_correlation_test, basic)
{
  using V = TypeParam;
  using R = double;

  constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

  auto keys     = cudf::test::fixed_width_column_wrapper<K>{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2}};
  auto member_0 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2, 2, 3, 3, 1, 1, 4}};
  auto member_1 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2, 0, 3, 3, 1, 1, 2}};
  auto vals     = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R, double> expect_vals{{1.0, 0.6, nan}};

  auto agg =
    cudf::make_correlation_aggregation<cudf::groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_correlation_test, empty_cols)
{
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> member_0{}, member_1{};
  auto vals = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg =
    cudf::make_correlation_aggregation<cudf::groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_correlation_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> member_0{3, 4, 5}, member_1{6, 7, 8};
  auto vals = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg =
    cudf::make_correlation_aggregation<cudf::groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_correlation_test, zero_valid_values)
{
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> member_0({3, 4, 5}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> member_1({3, 4, 5}, all_nulls());
  auto vals = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg =
    cudf::make_correlation_aggregation<cudf::groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_correlation_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = double;

  constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                     {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> val0({9, 1, 1, 2, 2, 3, 3,-1, 1, 4, 4},
                                     {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> val1({1, 1, 1, 2, 0, 3, 3,-1, 0, 2, 2});
  // clang-format on
  auto vals = cudf::test::structs_column_wrapper{{val0, val1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  cudf::test::fixed_width_column_wrapper<R> expect_vals({1.0, 0.6, nan, 0.}, {1, 1, 1, 0});

  auto agg =
    cudf::make_correlation_aggregation<cudf::groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_correlation_test, null_values_same)
{
  using V = TypeParam;
  using R = double;

  constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                     {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> val0({9, 1, 1, 2, 2, 3, 3,-1, 1, 4, 4},
                                     {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<V> val1({1, 1, 1, 2, 0, 3, 3,-1, 0, 2, 2},
                                     {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  // clang-format on
  auto vals = cudf::test::structs_column_wrapper{{val0, val1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  cudf::test::fixed_width_column_wrapper<R> expect_vals({1.0, 0.6, nan, 0.}, {1, 1, 1, 0});

  auto agg =
    cudf::make_correlation_aggregation<cudf::groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

// keys=[1, 1, 1, 2, 2, 2, 2,   3, N, 3, 4]
// val0=[N, 2, 3, 1, N, 3, 4,   1,-1, 1, 4]
// val1=[N, 2, 3, 2,-1, 6,-6/1, 1,-1, 0, N]
// corr=[    1.0,       -0.5/0, NAN,     NAN]
TYPED_TEST(groupby_correlation_test, null_values_different)
{
  using V = TypeParam;
  using R = double;

  constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                     {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> val0({9, 1, 1, 2, 2, 3, 3,-1, 1, 4, 4},
                                     {0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> val1({1, 2, 1, 2,-1, 6, 3,-1, 0, 1, 2},
                                     {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  // clang-format on
  auto vals = cudf::test::structs_column_wrapper{{val0, val1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  cudf::test::fixed_width_column_wrapper<R> expect_vals({1.0, 0., nan, 0.}, {1, 1, 1, 0});

  auto agg =
    cudf::make_correlation_aggregation<cudf::groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_correlation_test, min_periods)
{
  using V = TypeParam;
  using R = double;

  constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

  auto keys     = cudf::test::fixed_width_column_wrapper<K>{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2}};
  auto member_0 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2, 2, 3, 3, 1, 1, 4}};
  auto member_1 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2, 0, 3, 3, 1, 1, 2}};
  auto vals     = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  cudf::test::fixed_width_column_wrapper<R, double> expect_vals1{{1.0, 0.6, nan}};
  auto agg1 = cudf::make_correlation_aggregation<cudf::groupby_aggregation>(
    cudf::correlation_type::PEARSON, 3);
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg1));

  cudf::test::fixed_width_column_wrapper<R, double> expect_vals2{{1.0, 0.6, nan}, {0, 1, 0}};
  auto agg2 = cudf::make_correlation_aggregation<cudf::groupby_aggregation>(
    cudf::correlation_type::PEARSON, 4);
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg2));

  cudf::test::fixed_width_column_wrapper<R, double> expect_vals3{{1.0, 0.6, nan}, {0, 0, 0}};
  auto agg3 = cudf::make_correlation_aggregation<cudf::groupby_aggregation>(
    cudf::correlation_type::PEARSON, 5);
  test_single_agg(keys, vals, expect_keys, expect_vals3, std::move(agg3));
}

struct groupby_dictionary_correlation_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_correlation_test, basic)
{
  using V = int16_t;
  using R = double;

  constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

  auto keys     = cudf::test::fixed_width_column_wrapper<K>{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2}};
  auto member_0 = cudf::test::dictionary_column_wrapper<V>{{1, 1, 1, 2, 2, 3, 3, 1, 1, 4}};
  auto member_1 = cudf::test::dictionary_column_wrapper<V>{{1, 1, 1, 2, 0, 3, 3, 1, 1, 2}};
  auto vals     = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R, double> expect_vals{{1.0, 0.6, nan}};

  auto agg =
    cudf::make_correlation_aggregation<cudf::groupby_aggregation>(cudf::correlation_type::PEARSON);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}
