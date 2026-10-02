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

// Adding quantiles to a request must preserve its variance result.
TYPED_TEST(groupby_var_test, SeparateAndCombinedRequests)
{
  using K = int32_t;
  using V = double;

  cudf::test::fixed_width_column_wrapper<K> keys{50, 30, 90, 80};
  cudf::test::fixed_width_column_wrapper<V> vals{380.0, 370.0, 24.0, 26.0};

  cudf::groupby::groupby gb_obj(cudf::table_view({keys}));

  auto agg1 = cudf::make_variance_aggregation<cudf::groupby_aggregation>();

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(std::move(agg1));

  auto result1 = gb_obj.aggregate(requests);

  // Add an aggregation that sorts values within each group.
  auto agg2 = cudf::make_quantile_aggregation<cudf::groupby_aggregation>({0.25});
  requests[0].aggregations.push_back(std::move(agg2));

  auto result2 = gb_obj.aggregate(requests);

  auto order1  = cudf::sorted_order(result1.first->view());
  auto order2  = cudf::sorted_order(result2.first->view());
  auto values1 = cudf::gather(cudf::table_view{{result1.second[0].results[0]->view()}}, *order1);
  auto values2 = cudf::gather(cudf::table_view{{result2.second[0].results[0]->view()}}, *order2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(values1->get_column(0), values2->get_column(0));
}
