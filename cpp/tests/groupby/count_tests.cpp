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

template <typename V>
struct groupby_count_test : public cudf::test::BaseFixture {};
using K = int32_t;

TYPED_TEST_SUITE(groupby_count_test, cudf::test::AllTypes);

TYPED_TEST(groupby_count_test, basic)
{
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{3, 4, 3};

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg1 = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);

  auto agg2 = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_test, all_valid_values)
{
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4},
                                                 cudf::test::iterators::no_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{3, 2};

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto sort_agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(
    keys, vals, expect_keys, expect_vals, std::move(sort_agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_count_test, empty_cols)
{
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals;

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals;

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg1 = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_count_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg1 = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);

  auto agg2 = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

TYPED_TEST(groupby_count_test, zero_valid_values)
{
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{0};

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg1 = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);

  cudf::test::fixed_width_column_wrapper<R> expect_vals2{3};
  auto agg2 = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg2));
}

TYPED_TEST(groupby_count_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = cudf::size_type;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  // clang-format off
  //                                                    {1, 1,  2, 2, 2,  3, 3,  4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1,     2,        3,     4}, cudf::test::iterators::no_nulls());
  //                                                    {3, 6,  1, 4, 9,  2, 8,  -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({2,     3,        2,     0});
  // clang-format on

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg1 = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);

  cudf::test::fixed_width_column_wrapper<R> expect_vals2{3, 4, 2, 1};
  auto agg2 = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg2));
}

struct groupby_count_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_count_string_test, basic)
{
  using R = cudf::size_type;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{1,    3,  3,   5,   5,   0};
  cudf::test::strings_column_wrapper        vals{"1", "1", "1", "1", "1", "1"};
  // clang-format on

  cudf::test::fixed_width_column_wrapper<K> expect_keys{0, 1, 3, 5};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{1, 1, 2, 2};

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg1 = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);
}
// clang-format on

template <typename T>
struct GroupByCountFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupByCountFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(GroupByCountFixedPointTest, GroupByCount)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  using R = cudf::size_type;

  auto const scale = scale_type{-1};
  auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals  = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

  auto const expect_keys = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
  auto const expect_vals = cudf::test::fixed_width_column_wrapper<R>{3, 4, 3};

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg1 = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg1), force_use_sort_impl::YES);

  auto agg2 = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2));
}

struct groupby_dictionary_count_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_count_test, basic)
{
  using R = cudf::size_type;

  // clang-format off
  cudf::test::strings_column_wrapper        keys{"1", "3", "3", "5", "5", "0"};
  cudf::test::dictionary_column_wrapper<K>  vals{1, 1, 1, 1, 1, 1};
  cudf::test::strings_column_wrapper        expect_keys{"0", "1", "3", "5"};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{1, 1, 2, 2};
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_count_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_count_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES);
}
