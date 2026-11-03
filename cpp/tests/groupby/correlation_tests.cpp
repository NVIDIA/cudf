/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>

#include <limits>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_correlation_test : public cudf::test::BaseFixture {};

using supported_types =
  cudf::test::RemoveIf<cudf::test::ContainedIn<cudf::test::Types<bool>>, cudf::test::NumericTypes>;

TYPED_TEST_SUITE(groupby_correlation_test, supported_types);
using K = int32_t;

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
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
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
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
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
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
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
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
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
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
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
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
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
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
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
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg1), force_use_sort_impl::YES);

  cudf::test::fixed_width_column_wrapper<R, double> expect_vals2{{1.0, 0.6, nan}, {0, 1, 0}};
  auto agg2 = cudf::make_correlation_aggregation<cudf::groupby_aggregation>(
    cudf::correlation_type::PEARSON, 4);
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg2), force_use_sort_impl::YES);

  cudf::test::fixed_width_column_wrapper<R, double> expect_vals3{{1.0, 0.6, nan}, {0, 0, 0}};
  auto agg3 = cudf::make_correlation_aggregation<cudf::groupby_aggregation>(
    cudf::correlation_type::PEARSON, 5);
  test_single_agg(keys, vals, expect_keys, expect_vals3, std::move(agg3), force_use_sort_impl::YES);
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
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}
