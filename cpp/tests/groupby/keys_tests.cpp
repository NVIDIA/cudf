/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>

#include <cuda/iterator>

#include <cmath>
#include <initializer_list>
#include <span>
#include <tuple>
#include <vector>

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

TYPED_TEST(groupby_keys_test, distinct_keys_with_nullable_values)
{
  using K = TypeParam;
  cudf::test::fixed_width_column_wrapper<K> keys({3, 1, 0, 2}, {1, 1, 0, 1});
  cudf::test::fixed_width_column_wrapper<int32_t> vals({30, 99, 7, 20}, {1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> included_counts{1, 0, 1, 1};

  cudf::test::fixed_width_column_wrapper<K> excluded_keys({1, 2, 3}, no_nulls());
  cudf::test::fixed_width_column_wrapper<int32_t> excluded_vals({0, 20, 30}, {0, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> excluded_counts{0, 1, 1};

  // Including the null key yields one group per input row; excluding it must omit that row.
  for (auto null_handling : {cudf::null_policy::INCLUDE, cudf::null_policy::EXCLUDE}) {
    auto const include_null   = null_handling == cudf::null_policy::INCLUDE;
    auto const& expect_keys   = include_null ? keys : excluded_keys;
    auto const& expect_vals   = include_null ? vals : excluded_vals;
    auto const& expect_counts = include_null ? included_counts : excluded_counts;
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    expect_vals,
                    cudf::make_max_aggregation<cudf::groupby_aggregation>(),
                    force_use_sort_impl::NO,
                    null_handling);
    test_single_agg(
      keys,
      vals,
      expect_keys,
      expect_counts,
      cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE),
      force_use_sort_impl::NO,
      null_handling);
  }
}

TYPED_TEST(groupby_keys_test, one_duplicate_key_with_nullable_values)
{
  using K = TypeParam;
  // Exactly one repeated key gives one fewer group than input rows, including the null key.
  cudf::test::fixed_width_column_wrapper<K> keys({3, 1, 0, 3}, {1, 1, 0, 1});
  cudf::test::fixed_width_column_wrapper<int32_t> vals({30, 99, 7, 20}, {1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 3, 0}, {1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int32_t> expect_vals({0, 30, 7}, {0, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect_counts{0, 2, 1};

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_max_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::NO,
                  cudf::null_policy::INCLUDE);
  test_single_agg(
    keys,
    vals,
    expect_keys,
    expect_counts,
    cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE),
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

using groupby_sampling_test = groupby_keys_test<int32_t>;

TEST_F(groupby_sampling_test, NearlyDistinctSampleUnderestimatesPopulation)
{
  constexpr cudf::size_type num_rows    = 1 << 21;
  constexpr cudf::size_type stride      = 64;
  constexpr cudf::size_type sample_keys = 31'000;
  constexpr cudf::size_type num_samples = num_rows / stride;

  // The periodic sample is almost entirely distinct, but still has far fewer keys than the
  // complete input. Every row outside the sample has a unique key. An undersized table must
  // restart the build without dropping rows or duplicating groups.
  std::vector<int32_t> keys_data(num_rows);
  std::vector<int32_t> expected_keys;
  std::vector<cudf::size_type> expected_counts;
  std::vector<int32_t> expected_maxima;
  expected_keys.reserve(num_rows - num_samples + sample_keys);
  expected_counts.reserve(num_rows - num_samples + sample_keys);
  expected_maxima.reserve(num_rows - num_samples + sample_keys);
  for (cudf::size_type key = 0; key < sample_keys; ++key) {
    expected_keys.push_back(key);
    expected_counts.push_back(num_samples / sample_keys + (key < num_samples % sample_keys));
    auto const last_sample = key + ((num_samples - 1 - key) / sample_keys) * sample_keys;
    expected_maxima.push_back(last_sample * stride);
  }
  for (cudf::size_type row = 0; row < num_rows; ++row) {
    if (row % stride == 0) {
      keys_data[row] = (row / stride) % sample_keys;
    } else {
      keys_data[row] = sample_keys + row;
      expected_keys.push_back(keys_data[row]);
      expected_counts.push_back(1);
      expected_maxima.push_back(row);
    }
  }

  auto const keys =
    cudf::test::fixed_width_column_wrapper<int32_t>(keys_data.begin(), keys_data.end());
  auto const expect_keys =
    cudf::test::fixed_width_column_wrapper<int32_t>(expected_keys.begin(), expected_keys.end());
  auto const expect_counts = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    expected_counts.begin(), expected_counts.end());
  test_single_agg(keys,
                  keys,
                  expect_keys,
                  expect_counts,
                  cudf::make_count_aggregation<cudf::groupby_aggregation>());

  // COUNT only needs the group offsets. MAX of the row indices also verifies that the retry
  // rebuilt the row positions and filled the grouped row order correctly.
  auto const values = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{num_rows});
  auto const expect_maxima =
    cudf::test::fixed_width_column_wrapper<int32_t>(expected_maxima.begin(), expected_maxima.end());
  test_single_agg(keys,
                  values,
                  expect_keys,
                  expect_maxima,
                  cudf::make_max_aggregation<cudf::groupby_aggregation>());

  // Without requests the retry rebuilds the representative key rows instead of group slots.
  cudf::groupby::groupby gb_obj(cudf::table_view({keys}));
  auto const result = gb_obj.aggregate({}, cudf::test::get_default_stream());
  auto const sorted_keys =
    cudf::sort(result.first->view(), {}, {}, cudf::test::get_default_stream());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_keys, sorted_keys->view().column(0));
  EXPECT_TRUE(result.second.empty());
}

TEST_F(groupby_sampling_test, NullableMaxConsumesFullPackedSegment)
{
  constexpr cudf::size_type num_groups       = 1'000;
  constexpr cudf::size_type long_group_rows  = 1'024;
  constexpr cudf::size_type short_group_rows = 5;
  constexpr cudf::size_type num_rows = long_group_rows + (num_groups - 1) * short_group_rows;

  // Groups average six rows, but fewer than one in four values is valid. The density hint
  // chooses one thread per segment; it must still consume all 1,024 rows of the longest group.
  std::vector<int32_t> keys_data;
  std::vector<double> values_data;
  std::vector<bool> validity;
  std::vector<double> expected_maxima;
  std::vector<bool> expected_validity;
  keys_data.reserve(num_rows);
  values_data.reserve(num_rows);
  validity.reserve(num_rows);
  expected_maxima.reserve(num_groups);
  expected_validity.reserve(num_groups);

  for (cudf::size_type group = 0; group < num_groups; ++group) {
    auto const group_rows = group == 0 ? long_group_rows : short_group_rows;
    auto const maximum    = static_cast<double>(group * long_group_rows + group_rows - 1);
    expected_maxima.push_back(maximum);
    expected_validity.push_back(group != 1);
    for (cudf::size_type row = 0; row < group_rows; ++row) {
      auto const valid = group != 1 && row == group_rows - 1;
      keys_data.push_back(group);
      // Null payloads exceed every valid maximum, so loading one as valid also fails the test.
      values_data.push_back(valid ? maximum : 1.0e9);
      validity.push_back(valid);
    }
  }

  auto const keys =
    cudf::test::fixed_width_column_wrapper<int32_t>(keys_data.begin(), keys_data.end());
  auto const values = cudf::test::fixed_width_column_wrapper<double>(
    values_data.begin(), values_data.end(), validity.begin());
  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{num_groups});
  auto const expect_maxima = cudf::test::fixed_width_column_wrapper<double>(
    expected_maxima.begin(), expected_maxima.end(), expected_validity.begin());
  test_single_agg(keys,
                  values,
                  expect_keys,
                  expect_maxima,
                  cudf::make_max_aggregation<cudf::groupby_aggregation>());
}

template <typename ResultValidator>
void test_groupby_memory_resources(cudf::table_view const& keys,
                                   std::span<cudf::groupby::aggregation_request const> requests,
                                   rmm::device_async_resource_ref upstream,
                                   ResultValidator&& validate)
{
  auto const stream = cudf::test::get_default_stream();
  auto harness      = cudf::test::memory_resource_test_harness{upstream};
  {
    auto result = [&] {
      // The public API accepts an output resource and uses the current resource for scratch.
      cudf::test::scoped_current_device_resource temporary_scope{harness.temporary_mr()};
      cudf::groupby::groupby gb(keys);
      return gb.aggregate(requests, stream, harness.output_mr());
    }();
    auto bytes = result.first->alloc_size();
    for (auto const& request : result.second) {
      for (auto const& column : request.results) {
        bytes += column->alloc_size();
      }
    }
    harness.expect_resource_usage(bytes,
                                  {cudf::test::output_allocation_expectation::EXACT,
                                   cudf::test::temporary_allocation_expectation::SOME},
                                  stream);
    validate(result);
  }
  harness.expect_no_live_allocations(stream);
}

// Public groupby must use the output resource only for returned allocations and release all
// allocations made through the current temporary resource before returning.
struct HashGroupbyMemoryResourcesTest
  : cudf::test::BaseFixtureWithParam<std::tuple<cudf::size_type, cudf::size_type, bool>> {};

TEST_P(HashGroupbyMemoryResourcesTest, ExplicitOutputAndTemporaryResources)
{
  auto const [num_rows, num_groups, keys_only] = GetParam();
  std::vector<int32_t> key_data(num_rows);
  std::vector<int32_t> value_data(num_rows);
  std::vector<bool> validity(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    key_data[i]   = i % num_groups;
    value_data[i] = key_data[i] + 1;
    validity[i]   = key_data[i] != 0;
  }
  auto const keys =
    cudf::test::fixed_width_column_wrapper<int32_t>(key_data.begin(), key_data.end());
  auto const values = cudf::test::fixed_width_column_wrapper<int32_t>(
    value_data.begin(), value_data.end(), validity.begin());
  std::vector<cudf::groupby::aggregation_request> requests;
  if (!keys_only) {
    requests.emplace_back();
    requests.back().values = values;
    // SUM and COUNT_VALID also reduce an unrequested SUM_OF_SQUARES into temporary storage.
    requests.back().aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    requests.back().aggregations.push_back(
      cudf::make_count_aggregation<cudf::groupby_aggregation>());
    requests.back().aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
    requests.back().aggregations.push_back(
      cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
  }

  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{num_groups});
  std::vector<int64_t> sums(num_groups);
  std::vector<int32_t> maxima(num_groups), valid_counts(num_groups), all_counts(num_groups);
  std::vector<bool> expected_validity(num_groups);
  for (cudf::size_type group = 0; group < num_groups; ++group) {
    all_counts[group]        = num_rows / num_groups;
    valid_counts[group]      = group == 0 ? 0 : all_counts[group];
    maxima[group]            = group + 1;
    sums[group]              = static_cast<int64_t>(maxima[group]) * valid_counts[group];
    expected_validity[group] = group != 0;
  }
  auto const expect_sums = cudf::test::fixed_width_column_wrapper<int64_t>(
    sums.begin(), sums.end(), expected_validity.begin());
  auto const expect_valid_counts =
    cudf::test::fixed_width_column_wrapper<int32_t>(valid_counts.begin(), valid_counts.end());
  auto const expect_maxima = cudf::test::fixed_width_column_wrapper<int32_t>(
    maxima.begin(), maxima.end(), expected_validity.begin());
  auto const expect_all_counts =
    cudf::test::fixed_width_column_wrapper<int32_t>(all_counts.begin(), all_counts.end());

  test_groupby_memory_resources(
    cudf::table_view{{keys}}, requests, this->mr(), [&](auto const& result) {
      std::vector<cudf::column_view> actual{result.first->view().column(0)};
      std::vector<cudf::column_view> expected{expect_keys};
      if (!keys_only) {
        for (auto const& column : result.second.front().results) {
          actual.push_back(*column);
        }
        expected.insert(expected.end(),
                        {expect_sums, expect_valid_counts, expect_maxima, expect_all_counts});
      }
      auto const sorted = cudf::sort(cudf::table_view{actual});
      CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{expected}, sorted->view());
    });
}

INSTANTIATE_TEST_SUITE_P(
  HashCSR,
  HashGroupbyMemoryResourcesTest,
  ::testing::Values(std::tuple{cudf::size_type{512}, cudf::size_type{32}, true},
                    std::tuple{cudf::size_type{128}, cudf::size_type{128}, false},
                    std::tuple{cudf::size_type{2048}, cudf::size_type{128}, false},
                    std::tuple{cudf::size_type{32768}, cudf::size_type{1}, false},
                    std::tuple{cudf::size_type{1 << 21}, cudf::size_type{8}, false}));

struct HashGroupbyCompoundMemoryResourcesTest : public cudf::test::BaseFixture {};

TEST_F(HashGroupbyCompoundMemoryResourcesTest, NullableMeanVarianceAndStd)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{0, 0, 1, 1, 2, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> values({10, 20, 4, 6, 99, 99},
                                                         {1, 1, 1, 1, 0, 0});
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests.front().values = values;
  requests.front().aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  requests.front().aggregations.push_back(
    cudf::make_variance_aggregation<cudf::groupby_aggregation>(1));
  requests.front().aggregations.push_back(cudf::make_std_aggregation<cudf::groupby_aggregation>(1));

  cudf::test::fixed_width_column_wrapper<int32_t> expect_keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<double> expect_mean({15, 5, 0}, {1, 1, 0});
  cudf::test::fixed_width_column_wrapper<double> expect_variance({50, 2, 0}, {1, 1, 0});
  cudf::test::fixed_width_column_wrapper<double> expect_std({std::sqrt(50.0), std::sqrt(2.0), 0.0},
                                                            {1, 1, 0});

  test_groupby_memory_resources(
    cudf::table_view{{keys}}, requests, this->mr(), [&](auto const& result) {
      std::vector<cudf::column_view> actual{result.first->view().column(0)};
      for (auto const& column : result.second.front().results) {
        actual.push_back(*column);
      }
      auto const sorted = cudf::sort(cudf::table_view{actual});
      CUDF_TEST_EXPECT_TABLES_EQUAL(
        cudf::table_view{{expect_keys, expect_mean, expect_variance, expect_std}}, sorted->view());
    });
}

struct HashGroupbyRepeatedMemoryResourcesTest
  : cudf::test::BaseFixtureWithParam<std::tuple<bool, bool>> {};

TEST_P(HashGroupbyRepeatedMemoryResourcesTest, SharedValuesAndRepeatedResults)
{
  auto const [reverse_requests, moments] = GetParam();
  cudf::test::fixed_width_column_wrapper<int32_t> keys{0, 0, 1, 1, 2, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> values(
    {10, 20, 4, 6, 99, 99, 30, 50, 8, 12, 2, 4}, {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1});
  auto const value_views = cudf::slice(values, {0, 6, 6, 12});
  auto make_compound     = [&]() -> std::unique_ptr<cudf::groupby_aggregation> {
    if (moments) { return cudf::make_variance_aggregation<cudf::groupby_aggregation>(1); }
    return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  };
  auto make_dependency = [&]() -> std::unique_ptr<cudf::groupby_aggregation> {
    if (moments) { return cudf::make_m2_aggregation<cudf::groupby_aggregation>(); }
    return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  };

  // Both requests share a cache key for their dependency; duplicates must still produce two
  // independently owned output columns. The third request uses a different slice of the same
  // input allocation and must not share either result with the first two requests.
  std::vector<cudf::groupby::aggregation_request> requests(3);
  for (auto i = 0; i < 3; ++i) {
    requests[i].values = value_views[i == 2 ? 1 : 0];
    for (auto repeat = 0; repeat < 2; ++repeat) {
      requests[i].aggregations.push_back(i == 1 ? make_dependency() : make_compound());
    }
  }
  if (reverse_requests) { std::swap(requests[0], requests[1]); }

  cudf::test::fixed_width_column_wrapper<int32_t> expect_keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<double> expect_compound(
    moments ? std::initializer_list<double>{50, 2, 0} : std::initializer_list<double>{15, 5, 0},
    std::initializer_list<bool>{true, true, false});
  cudf::test::fixed_width_column_wrapper<double> expect_m2{50, 2, 0};
  cudf::test::fixed_width_column_wrapper<int64_t> expect_sum({30, 10, 0}, {1, 1, 0});
  cudf::test::fixed_width_column_wrapper<double> expect_other(
    moments ? std::initializer_list<double>{200, 8, 2} : std::initializer_list<double>{40, 10, 3});
  auto const dependency_view =
    moments ? cudf::column_view{expect_m2} : cudf::column_view{expect_sum};

  test_groupby_memory_resources(
    cudf::table_view{{keys}}, requests, this->mr(), [&](auto const& result) {
      ASSERT_EQ(result.second.size(), 3);
      for (std::size_t i = 0; i < result.second.size(); ++i) {
        auto const expected = i == 2                  ? cudf::column_view{expect_other}
                              : i == reverse_requests ? cudf::column_view{expect_compound}
                                                      : dependency_view;
        ASSERT_EQ(result.second[i].results.size(), 2);
        auto const actual = cudf::table_view{{result.first->view().column(0),
                                              *result.second[i].results[0],
                                              *result.second[i].results[1]}};
        auto const sorted = cudf::sort(actual);
        CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{expect_keys, expected, expected}},
                                      sorted->view());
      }
    });
}

INSTANTIATE_TEST_SUITE_P(HashCSR,
                         HashGroupbyRepeatedMemoryResourcesTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()));

struct HashGroupbyValidGroupsMemoryResourcesTest : cudf::test::BaseFixtureWithParam<bool> {};

TEST_P(HashGroupbyValidGroupsMemoryResourcesTest, NullableInputWithNoInvalidOutputGroups)
{
  auto const request_sum = GetParam();
  cudf::test::fixed_width_column_wrapper<int32_t> keys{0, 0, 0, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> values({10, 20, 99, 4, 6, 99},
                                                         {1, 1, 0, 1, 1, 0});
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests.front().values = values;
  requests.front().aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  requests.front().aggregations.push_back(
    cudf::make_variance_aggregation<cudf::groupby_aggregation>(1));
  requests.front().aggregations.push_back(cudf::make_std_aggregation<cudf::groupby_aggregation>(1));
  if (request_sum) {
    requests.front().aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  }

  cudf::test::fixed_width_column_wrapper<int32_t> expect_keys{0, 1};
  cudf::test::fixed_width_column_wrapper<double> expect_mean{15, 5};
  cudf::test::fixed_width_column_wrapper<double> expect_variance{50, 2};
  cudf::test::fixed_width_column_wrapper<double> expect_std{std::sqrt(50.0), std::sqrt(2.0)};
  cudf::test::fixed_width_column_wrapper<int64_t> expect_sum({30, 10}, {1, 1});
  test_groupby_memory_resources(
    cudf::table_view{{keys}}, requests, this->mr(), [&](auto const& result) {
      std::vector<cudf::column_view> actual{result.first->view().column(0)};
      for (auto const& column : result.second.front().results) {
        actual.push_back(*column);
      }
      std::vector<cudf::column_view> expected{
        expect_keys, expect_mean, expect_variance, expect_std};
      if (request_sum) { expected.push_back(expect_sum); }
      auto const sorted = cudf::sort(cudf::table_view{actual});
      CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{expected}, sorted->view());
    });
}

INSTANTIATE_TEST_SUITE_P(HashCSR, HashGroupbyValidGroupsMemoryResourcesTest, ::testing::Bool());

TEST_F(HashGroupbyCompoundMemoryResourcesTest, StringMinMaxIntermediateIndices)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{0, 0, 1, 1, 2, 2};
  cudf::test::strings_column_wrapper values({"z", "a", "q", "r", "", ""}, {1, 1, 1, 1, 0, 0});
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests.front().values = values;
  requests.front().aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  requests.front().aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  cudf::test::fixed_width_column_wrapper<int32_t> expect_keys{0, 1, 2};
  cudf::test::strings_column_wrapper expect_min({"a", "q", ""}, {1, 1, 0});
  cudf::test::strings_column_wrapper expect_max({"z", "r", ""}, {1, 1, 0});
  test_groupby_memory_resources(
    cudf::table_view{{keys}}, requests, this->mr(), [&](auto const& result) {
      auto const actual = cudf::table_view{{result.first->view().column(0),
                                            *result.second.front().results[0],
                                            *result.second.front().results[1]}};
      auto const sorted = cudf::sort(actual);
      CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{expect_keys, expect_min, expect_max}},
                                    sorted->view());
    });
}
