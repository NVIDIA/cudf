/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>

#include <cuda/std/type_traits>

#include <type_traits>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_sum_of_squares_test : public cudf::test::BaseFixture {};

using supported_types = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_SUITE(groupby_sum_of_squares_test, supported_types);

TYPED_TEST(groupby_sum_of_squares_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = std::conditional_t<cuda::std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  //  { 1, 1, 1,  2, 2, 2, 2,  3, 3, 3}
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  //  { 0, 3, 6,  1, 4, 5, 9,  2, 7, 8}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({45., 123., 117.}, no_nulls());

  auto agg = cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_of_squares_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;
  using R = std::conditional_t<cuda::std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_of_squares_test, zero_valid_keys)
{
  using K = int32_t;
  using V = TypeParam;
  using R = std::conditional_t<cuda::std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_of_squares_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = std::conditional_t<cuda::std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_of_squares_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = std::conditional_t<cuda::std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  //  { 3, 6,     1, 4, 9,   2, 8,    3}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({45., 98., 68., 9.}, {1, 1, 1, 0});

  auto agg = cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_of_squares_test, dictionary)
{
  using K = int32_t;
  using V = TypeParam;
  using R = std::conditional_t<cuda::std::is_integral_v<V>, int64_t, V>;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::dictionary_column_wrapper<V>  vals{1, 1, 2, 2, 4, 4, 6, 6, 9, 9};

  //                                                    {1, 1, 1,  2, 2, 2, 2,  3, 3, 3}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1,        2,           3      });
  //                                                    {1, 2, 6,  1, 4, 4, 9,  2, 6, 9}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({41.,      114.,        121.   }, no_nulls());
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>());
}
