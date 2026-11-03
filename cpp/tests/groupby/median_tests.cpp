/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_median_test : public cudf::test::BaseFixture {};

using supported_types = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_SUITE(groupby_median_test, supported_types);

TYPED_TEST(groupby_median_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // clang-format off
  //                                                   {1, 1, 1, 2, 2, 2, 2, 3, 3, 3}
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1,       2,          3};
  //                                                    {0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3.,     4.5,        7.}, no_nulls());
  // clang-format on

  auto agg = cudf::make_median_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_median_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_median_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_median_test, zero_valid_keys)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_median_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_median_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_median_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_median_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  //  { 3, 6,     1, 4, 9,   2, 8,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({4.5, 4., 5., 0.}, {1, 1, 1, 0});

  auto agg = cudf::make_median_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_median_test, dictionary)
{
  using K = int32_t;
  using V = TypeParam;
  using R = double;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{ 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::dictionary_column_wrapper<V>  vals{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  //                                                    {1, 1, 1, 2, 2, 2, 2, 3, 3, 3}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1,       2,          3      });
  //                                                    {0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3.,       4.5,       7.     }, no_nulls());
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_median_aggregation<cudf::groupby_aggregation>());
}
