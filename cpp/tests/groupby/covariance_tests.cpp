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
struct groupby_covariance_test : public cudf::test::BaseFixture {};

using supported_types =
  cudf::test::RemoveIf<cudf::test::ContainedIn<cudf::test::Types<bool>>, cudf::test::NumericTypes>;

TYPED_TEST_SUITE(groupby_covariance_test, supported_types);
using K = int32_t;

TYPED_TEST(groupby_covariance_test, invalid_types)
{
  using V = TypeParam;

  auto keys     = cudf::test::fixed_width_column_wrapper<K>{{1, 2, 2, 1}};
  auto member_0 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2}};
  // Covariance cudf::aggregations require all types are convertible to double, but
  // duration_D cannot be converted to double.
  auto member_1 =
    cudf::test::fixed_width_column_wrapper<cudf::duration_D, cudf::duration_D::rep>{{0, 0, 1, 1}};
  auto vals = cudf::test::structs_column_wrapper{{member_0, member_1}};

  auto agg = cudf::make_covariance_aggregation<cudf::groupby_aggregation>();
  EXPECT_THROW(test_single_agg(keys, vals, keys, vals, std::move(agg), force_use_sort_impl::YES),
               cudf::logic_error);
}

TYPED_TEST(groupby_covariance_test, basic)
{
  using V = TypeParam;
  using R = double;

  auto keys     = cudf::test::fixed_width_column_wrapper<K>{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2}};
  auto member_0 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2, 2, 3, 3, 1, 1, 4}};
  auto member_1 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2, 0, 3, 3, 1, 1, 2}};
  auto vals     = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R, double> expect_vals{{1.0, 1.0, 0.0}};

  auto agg = cudf::make_covariance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_covariance_test, empty_cols)
{
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> member_0{}, member_1{};
  auto vals = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_covariance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_covariance_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> member_0{3, 4, 5}, member_1{6, 7, 8};
  auto vals = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_covariance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_covariance_test, zero_valid_values)
{
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> member_0({3, 4, 5}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> member_1({3, 4, 5}, all_nulls());
  auto vals = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_covariance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_covariance_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = double;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                     {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> val0({9, 1, 1, 2, 2, 3, 3,-1, 1, 4, 4},
                                     {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> val1({1, 1, 1, 2, 0, 3, 3,-1, 0, 2, 2});
  // clang-format on
  auto vals = cudf::test::structs_column_wrapper{{val0, val1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0.5, 1.0, 0.0, -0.}, {1, 1, 1, 0});

  auto agg = cudf::make_covariance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_covariance_test, null_values_same)
{
  using V = TypeParam;
  using R = double;

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
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0.5, 1.0, 0.0, -0.}, {1, 1, 1, 0});

  auto agg = cudf::make_covariance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_covariance_test, null_values_different)
{
  using V = TypeParam;
  using R = double;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                     {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> val0({9, 1, 1, 2, 2, 3, 3,-1, 1, 4, 4},
                                     {0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> val1({1, 2, 1, 2,-1, 3, 3,-1, 0, 4, 2},
                                     {0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0});
  // clang-format on
  auto vals = cudf::test::structs_column_wrapper{{val0, val1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  cudf::test::fixed_width_column_wrapper<R> expect_vals(
    {std::numeric_limits<double>::quiet_NaN(), 1.5, 0.0, -0.}, {0, 1, 1, 0});

  auto agg = cudf::make_covariance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_covariance_test, min_periods)
{
  using V = TypeParam;
  using R = double;

  auto keys     = cudf::test::fixed_width_column_wrapper<K>{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2}};
  auto member_0 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2, 2, 3, 3, 1, 1, 4}};
  auto member_1 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2, 0, 3, 3, 1, 1, 2}};
  auto vals     = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  cudf::test::fixed_width_column_wrapper<R, double> expect_vals1{{1.0, 1.0, 0.0}};
  auto agg1 = cudf::make_covariance_aggregation<cudf::groupby_aggregation>(3);
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg1), force_use_sort_impl::YES);

  cudf::test::fixed_width_column_wrapper<R, double> expect_vals2{{1.0, 1.0, 0.0}, {0, 1, 0}};
  auto agg2 = cudf::make_covariance_aggregation<cudf::groupby_aggregation>(4);
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg2), force_use_sort_impl::YES);

  cudf::test::fixed_width_column_wrapper<R, double> expect_vals3{{1.0, 1.0, 0.0}, {0, 0, 0}};
  auto agg3 = cudf::make_covariance_aggregation<cudf::groupby_aggregation>(5);
  test_single_agg(keys, vals, expect_keys, expect_vals3, std::move(agg3), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_covariance_test, ddof)
{
  using V = TypeParam;
  using R = double;

  auto keys     = cudf::test::fixed_width_column_wrapper<K>{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2}};
  auto member_0 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2, 2, 3, 3, 1, 1, 4}};
  auto member_1 = cudf::test::fixed_width_column_wrapper<V>{{1, 1, 1, 2, 0, 3, 3, 1, 1, 2}};
  auto vals     = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  cudf::test::fixed_width_column_wrapper<R, double> expect_vals1{{2.0, 1.5, 0.0}};
  auto agg1 = cudf::make_covariance_aggregation<cudf::groupby_aggregation>(1, 2);
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg1), force_use_sort_impl::YES);

  auto const inf = std::numeric_limits<double>::infinity();
  cudf::test::fixed_width_column_wrapper<R, double> expect_vals2{{inf, 3.0, 0.0}, {0, 1, 0}};
  auto agg2 = cudf::make_covariance_aggregation<cudf::groupby_aggregation>(1, 3);
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg2), force_use_sort_impl::YES);
}

struct groupby_dictionary_covariance_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_covariance_test, basic)
{
  using V = int16_t;
  using R = double;

  auto keys     = cudf::test::fixed_width_column_wrapper<K>{{1, 2, 3, 1, 2, 2, 1, 3, 3, 2}};
  auto member_0 = cudf::test::dictionary_column_wrapper<V>{{1, 1, 1, 2, 2, 3, 3, 1, 1, 4}};
  auto member_1 = cudf::test::dictionary_column_wrapper<V>{{1, 1, 1, 2, 3, -3, 3, 1, 1, 2}};
  auto vals     = cudf::test::structs_column_wrapper{{member_0, member_1}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R, double> expect_vals{{1.0, -0.5, 0.0}};

  auto agg = cudf::make_covariance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), force_use_sort_impl::YES);
}
