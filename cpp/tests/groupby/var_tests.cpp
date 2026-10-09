/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

#include <cstddef>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_var_test : public cudf::test::BaseFixture {};

using supported_types = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_SUITE(groupby_var_test, supported_types);

TYPED_TEST(groupby_var_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  //                                                   {1, 1, 1,  2, 2, 2, 2,  3, 3, 3}
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1,        2,           3};
  //                                                   {0, 3, 6,  1, 4, 5, 9,  2, 7, 8}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({9.,      131. / 12,   31. / 3}, no_nulls());
  // clang-format on

  auto agg = cudf::make_variance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_variance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, zero_valid_keys)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_variance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_variance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});

  // clang-format off
  //                                                    {1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1,        2,         3,       4}, no_nulls());
  //                                                    {3, 6,     1, 4, 9,   2, 8,    3}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({4.5,      49. / 3,   18.,     0.}, {1, 1, 1, 0});
  // clang-format on

  auto agg = cudf::make_variance_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, ddof_non_default)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});

  // clang-format off
  //                                                    { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1,         2,         3,       4}, no_nulls());
  //                                                    { 3, 6,     1, 4, 9,   2, 8,    3}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0.,        98. / 3,   0.,      0.},
                                                        {0,         1,         0,       0});
  // clang-format on

  auto agg = cudf::make_variance_aggregation<cudf::groupby_aggregation>(2);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_var_test, dictionary)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::dictionary_column_wrapper<V>  vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  //                                                    {1, 1, 1,  2, 2, 2, 2,  3, 3, 3}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1,        2,           3      });
  //                                                    {0, 3, 6,  1, 4, 5, 9,  2, 7, 8}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({9.,      131./12,      31./3  }, no_nulls());
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_variance_aggregation<cudf::groupby_aggregation>());
}

using groupby_var_mixed_test = groupby_var_test<double>;

TEST_F(groupby_var_mixed_test, SeparateAndCombinedRequests)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<double> vals{5.0, 6.0, 3.0, 2.0};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{1, 2};
  cudf::test::fixed_width_column_wrapper<double> expected_variance{8.0, 2.0};
  cudf::test::fixed_width_column_wrapper<double> expected_quantile{3.0, 3.5};
  cudf::groupby::groupby gb_obj(cudf::table_view{{keys}});
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = vals;
  requests[0].aggregations.push_back(cudf::make_variance_aggregation<cudf::groupby_aggregation>());

  // Sorting values for quantiles must preserve variance on the reused grouping.
  for (bool const with_quantile : {false, true}) {
    if (with_quantile) {
      requests[0].aggregations.push_back(
        cudf::make_quantile_aggregation<cudf::groupby_aggregation>({0.25}));
    }
    auto result = gb_obj.aggregate(requests);
    ASSERT_EQ(result.second.size(), 1);
    ASSERT_EQ(result.second[0].results.size(), with_quantile ? 2 : 1);
    auto order       = cudf::sorted_order(result.first->view());
    auto sorted_keys = cudf::gather(result.first->view(), *order);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted_keys->get_column(0));
    for (std::size_t i = 0; i < result.second[0].results.size(); ++i) {
      auto values = cudf::gather(cudf::table_view{{*result.second[0].results[i]}}, *order);
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(i == 0 ? expected_variance : expected_quantile,
                                          values->get_column(0));
    }
  }
}
