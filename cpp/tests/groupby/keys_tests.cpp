/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_keys_test : public cudf::test::BaseFixture {};

using supported_types = cudf::test::
  Types<int8_t, int16_t, int32_t, int64_t, float, double, numeric::decimal32, numeric::decimal64>;

TYPED_TEST_SUITE(groupby_keys_test, supported_types);

TYPED_TEST(groupby_keys_test, basic)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::size_type;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys        { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys { 1, 2, 3 };
  cudf::test::fixed_width_column_wrapper<R> expect_vals { 3, 4, 3 };
  // clang-format on

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_keys_test, zero_valid_keys)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::size_type;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys      ( { 1, 2, 3}, all_nulls() );
  cudf::test::fixed_width_column_wrapper<V> vals        { 3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys { };
  cudf::test::fixed_width_column_wrapper<R> expect_vals { };
  // clang-format on

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_keys_test, some_null_keys)
{
  using K = TypeParam;
  using V = int32_t;
  using R = cudf::size_type;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                                        { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

                                                    //  { 1, 1, 1,  2, 2, 2, 2,  3, 3,  4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({ 1,        2,           3,     4}, no_nulls() );
                                                    //  { 0, 3, 6,  1, 4, 5, 9,  2, 8,  -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals { 3,        4,           2,     1};
  // clang-format on

  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_keys_test, include_null_keys)
{
  using K = TypeParam;
  using V = int32_t;
  using R = int64_t;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys(       { 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
                                                        { 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

                                                    //  { 1, 1, 1,  2, 2, 2, 2,  3, 3,  4,  -}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({ 1,        2,           3,     4,  3},
                                                        { 1,        1,           1,     1,  0});
                                                    //  { 0, 3, 6,  1, 4, 5, 9,  2, 8,  -,  -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals { 9,        19,          10,    4,  7};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::NO,
                  cudf::null_policy::INCLUDE);
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys)
{
  using K = TypeParam;
  using V = int32_t;
  using R = int64_t;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys        { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4};
  cudf::test::fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

  cudf::test::fixed_width_column_wrapper<K> expect_keys { 1,       2,          3,       4};
  cudf::test::fixed_width_column_wrapper<R> expect_vals { 3,       18,         24,      4};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::YES,
                  cudf::null_policy::EXCLUDE,
                  cudf::sorted::YES);
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys_descending)
{
  using K = TypeParam;
  using V = int32_t;
  using R = int64_t;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys        { 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

  cudf::test::fixed_width_column_wrapper<K> expect_keys { 4, 3,       2,          1      };
  cudf::test::fixed_width_column_wrapper<R> expect_vals { 0, 6,       22,        21      };
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::YES,
                  cudf::null_policy::EXCLUDE,
                  cudf::sorted::YES,
                  {cudf::order::DESCENDING});
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys_nullable)
{
  using K = TypeParam;
  using V = int32_t;
  using R = int64_t;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys(       { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4},
                                                        { 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

  cudf::test::fixed_width_column_wrapper<K> expect_keys({ 1,       2,          3,       4}, no_nulls() );
  cudf::test::fixed_width_column_wrapper<R> expect_vals { 3,       15,         17,      4};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::YES,
                  cudf::null_policy::EXCLUDE,
                  cudf::sorted::YES);
}

TYPED_TEST(groupby_keys_test, pre_sorted_keys_nulls_before_include_nulls)
{
  using K = TypeParam;
  using V = int32_t;
  using R = int64_t;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys(       { 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4},
                                                        { 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> vals        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4};

                                                    //  { 1, 1, 1,  -, -,  2, 2,  -,  3, 3,  4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({ 1,        2,     2,     3,  3,     4},
                                                        { 1,        0,     1,     0,  1,     1});
  cudf::test::fixed_width_column_wrapper<R> expect_vals { 3,        7,     11,    7,  17,    4};
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::YES,
                  cudf::null_policy::INCLUDE,
                  cudf::sorted::YES);
}

TYPED_TEST(groupby_keys_test, mismatch_num_rows)
{
  using K = TypeParam;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4};

  // Verify that scan throws an error when given data of mismatched sizes.
  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  EXPECT_THROW(test_single_agg(keys, vals, keys, vals, std::move(agg)), cudf::logic_error);
  auto agg2 = cudf::make_count_aggregation<cudf::groupby_scan_aggregation>();
  EXPECT_THROW(test_single_scan(keys, vals, keys, vals, std::move(agg2)), cudf::logic_error);
}

template <typename T>
using FWCW = cudf::test::fixed_width_column_wrapper<T>;

TYPED_TEST(groupby_keys_test, structs)
{
  using V = TypeParam;

  using R       = cudf::size_type;
  using STRINGS = cudf::test::strings_column_wrapper;
  using STRUCTS = cudf::test::structs_column_wrapper;

  if (std::is_same_v<V, bool>) return;

  /*
    `@` indicates null
       keys:                values:
       /+----------------+
       |s1{s2{a,b},   c}|
       +-----------------+
     0 |  { { 1, 1}, "a"}|  1
     1 |  { { 1, 2}, "b"}|  2
     2 |  {@{ 2, 1}, "c"}|  3
     3 |  {@{ 2, 1}, "c"}|  4
     4 | @{ { 2, 2}, "d"}|  5
     5 | @{ { 2, 2}, "d"}|  6
     6 |  { { 1, 1}, "a"}|  7
     7 |  {@{ 2, 1}, "c"}|  8
     8 |  { {@1, 1}, "a"}|  9
       +-----------------+
  */

  // clang-format off
  auto col_a = FWCW<V>{{ 1,   1,   2,   2,   2,   2,   1,   2,   1 }, null_at(8)};
  auto col_b = FWCW<V> { 1,   2,   1,   1,   2,   2,   1,   1,   1 };
  auto col_c = STRINGS {"a", "b", "c", "c", "d", "d", "a", "c", "a"};
  // clang-format on
  auto s2 = STRUCTS{{col_a, col_b}, nulls_at({2, 3, 7})};

  auto keys = STRUCTS{{s2, col_c}, nulls_at({4, 5})};
  auto vals = FWCW<int>{1, 2, 3, 4, 5, 6, 7, 8, 9};

  // clang-format off
  auto expected_col_a = FWCW<V>{{1,   1,   1,   2 }, null_at(2)};
  auto expected_col_b = FWCW<V>{ 1,   2,   1,   1 };
  auto expected_col_c = STRINGS{"a", "b", "a", "c"};
  // clang-format on
  auto expected_s2 = STRUCTS{{expected_col_a, expected_col_b}, null_at(3)};

  auto expect_keys = STRUCTS{{expected_s2, expected_col_c}, no_nulls()};
  auto expect_vals = FWCW<R>{6, 1, 8, 7};

  auto agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

TYPED_TEST(groupby_keys_test, lists)
{
  using R = int64_t;

  // clang-format off
  auto keys   = LCW<TypeParam> { {1,1}, {2,2}, {3,3}, {1,1}, {2,2} };
  auto values = FWCW<int32_t>  {    0,     1,     2,     3,     4  };

  auto expected_keys   = LCW<TypeParam> { {1,1}, {2,2}, {3,3} };
  auto expected_values = FWCW<R>        {    3,     5,     2  };
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, values, expected_keys, expected_values, std::move(agg));
}

struct groupby_string_keys_test : public cudf::test::BaseFixture {};

TEST_F(groupby_string_keys_test, basic)
{
  using V = int32_t;
  using R = int64_t;

  // clang-format off
  cudf::test::strings_column_wrapper        keys        { "aaa", "año", "₹1", "aaa", "año", "año", "aaa", "₹1", "₹1", "año"};
  cudf::test::fixed_width_column_wrapper<V> vals        {     0,     1,    2,     3,     4,     5,     6,    7,    8,     9};

  cudf::test::strings_column_wrapper        expect_keys({ "aaa", "año", "₹1" });
  cudf::test::fixed_width_column_wrapper<R> expect_vals {     9,    19,   17 };
  // clang-format on

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}
// clang-format on

struct groupby_dictionary_keys_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_keys_test, basic)
{
  using K = std::string;
  using V = int32_t;
  using R = int64_t;

  // clang-format off
  cudf::test::dictionary_column_wrapper<K> keys { "aaa", "año", "₹1", "aaa", "año", "año", "aaa", "₹1", "₹1", "año"};
  cudf::test::fixed_width_column_wrapper<V> vals{     0,     1,    2,     3,     4,     5,     6,    7,    8,     9};
  cudf::test::dictionary_column_wrapper<K>expect_keys  ({ "aaa", "año", "₹1" });
  cudf::test::fixed_width_column_wrapper<R> expect_vals({     9,    19,   17 });
  // clang-format on

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

struct groupby_cache_test : public cudf::test::BaseFixture {};

// To check if the cache doesn't insert multiple times to cache for the same aggregation on a
// column in the same request. If this test fails, then insert happened and the key stored in the
// cache map becomes a dangling reference. Any comparison with the same aggregation as the key will
// fail.
TEST_F(groupby_cache_test, duplicate_agggregations)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  cudf::groupby::groupby gb_obj(cudf::table_view({keys}));

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  // hash groupby
  EXPECT_NO_THROW(gb_obj.aggregate(requests));

  // sort groupby
  // WAR to force groupby to use sort implementation
  requests[0].aggregations.push_back(
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
  EXPECT_NO_THROW(gb_obj.aggregate(requests));
}

// To check if the cache doesn't insert multiple times to cache for the same aggregation on the same
// column but in different requests. If this test fails, then insert happened and the key stored in
// the cache map becomes a dangling reference. Any comparison with the same aggregation as the key
// will fail.
TEST_F(groupby_cache_test, duplicate_columns)
{
  using K = int32_t;
  using V = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  cudf::groupby::groupby gb_obj(cudf::table_view({keys}));

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests.emplace_back();
  requests[1].values = vals;
  requests[1].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  // hash groupby
  EXPECT_NO_THROW(gb_obj.aggregate(requests));

  // sort groupby
  // WAR to force groupby to use sort implementation
  requests[0].aggregations.push_back(
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
  EXPECT_NO_THROW(gb_obj.aggregate(requests));
}
