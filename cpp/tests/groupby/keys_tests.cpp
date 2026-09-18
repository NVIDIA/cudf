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
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>

#include <cuda/iterator>

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

using groupby_key_shape_test = groupby_keys_test<int32_t>;

TEST_F(groupby_key_shape_test, NearlyDistinctSampleUnderestimatesPopulation)
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

TEST_F(groupby_key_shape_test, MixedGroupSizeReductions)
{
  std::vector<int32_t> const sizes{1, 32, 33, 1024, 1025, 262145};
  std::vector<int32_t> keys_data, values_data;
  std::vector<bool> validity;
  for (std::size_t group = 0; group < sizes.size(); ++group) {
    for (int32_t value = 1; value <= sizes[group]; ++value) {
      keys_data.push_back(static_cast<int32_t>(group));
      values_data.push_back(value);
      validity.push_back(sizes[group] != 1 && value == sizes[group]);
    }
  }
  auto const keys =
    cudf::test::fixed_width_column_wrapper<int32_t>(keys_data.begin(), keys_data.end());
  auto const values =
    cudf::test::fixed_width_column_wrapper<int32_t>(values_data.begin(), values_data.end());
  auto const nullable_values = cudf::test::fixed_width_column_wrapper<int32_t>(
    values_data.begin(), values_data.end(), validity.begin());
  std::vector<cudf::groupby::aggregation_request> requests(2);
  requests[0].values = values;
  requests[1].values = nullable_values;
  for (auto& request : requests) {
    request.aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
    request.aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
    request.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  }

  // Mixed sizes exercise direct and chunked reductions. Only the last input row is valid in
  // the nullable column, except for the all-null singleton; CSR row order is unspecified.
  cudf::groupby::groupby gb(cudf::table_view{{keys}});
  auto const [result_keys, results] = gb.aggregate(requests, cudf::test::get_default_stream());
  ASSERT_EQ(results.size(), 2);
  auto const expected_keys = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0},
    cuda::counting_iterator<int32_t>{static_cast<int32_t>(sizes.size())});
  for (std::size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(results[i].results.size(), 3);
    std::vector<int32_t> minima;
    std::vector<int64_t> sums;
    std::vector<bool> expected_validity;
    for (auto const size : sizes) {
      minima.push_back(i == 0 ? 1 : size);
      sums.push_back(i == 0 ? static_cast<int64_t>(size) * (size + 1) / 2 : size);
      expected_validity.push_back(i == 0 || size != 1);
    }
    auto const expected_min = cudf::test::fixed_width_column_wrapper<int32_t>(
      minima.begin(), minima.end(), expected_validity.begin());
    auto const expected_max = cudf::test::fixed_width_column_wrapper<int32_t>(
      sizes.begin(), sizes.end(), expected_validity.begin());
    auto const expected_sum = cudf::test::fixed_width_column_wrapper<int64_t>(
      sums.begin(), sums.end(), expected_validity.begin());
    auto const actual = cudf::table_view{{result_keys->view().column(0),
                                          *results[i].results[0],
                                          *results[i].results[1],
                                          *results[i].results[2]}};
    auto const sorted = cudf::sort(actual, {}, {}, cudf::test::get_default_stream());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(
      cudf::table_view{{expected_keys, expected_min, expected_max, expected_sum}}, sorted->view());
  }
}

TEST_F(groupby_key_shape_test, BatchedNullableSumUsesOutputAndTemporaryResources)
{
  auto const stream         = cudf::test::get_default_stream();
  constexpr int num_columns = 2;
  constexpr int num_rows    = 2'074;
  std::vector<int32_t> keys_data(num_rows);
  std::vector<std::vector<int32_t>> data(num_columns, std::vector<int32_t>(num_rows));
  std::vector<std::vector<bool>> valid(num_columns, std::vector<bool>(num_rows));
  std::vector<std::vector<int64_t>> sums(num_columns, std::vector<int64_t>(3));
  // Groups of 2,000, 73 and 1 rows exercise each reduction stage. The columns have different
  // all-null groups, so their partial values and output masks must remain independent.
  for (int row = 0; row < num_rows; ++row) {
    auto const group = row < 2'000 ? 0 : row < 2'073 ? 1 : 2;
    keys_data[row]   = group == 0 ? -7 : group == 1 ? 41 : 99;
    for (int column = 0; column < num_columns; ++column) {
      data[column][row]  = (column == 0 ? 200'000'000 : -300'000'000) + row % 17;
      valid[column][row] = group != 2 - 2 * column && row % 5 != column;
      if (valid[column][row]) { sums[column][group] += data[column][row]; }
    }
  }
  auto const keys =
    cudf::test::fixed_width_column_wrapper<int32_t>(keys_data.begin(), keys_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expect_keys{-7, 41, 99};
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> columns;
  std::vector<cudf::test::fixed_width_column_wrapper<int64_t>> expected;
  std::vector<cudf::groupby::aggregation_request> requests(num_columns);
  columns.reserve(num_columns);
  for (int column = 0; column < num_columns; ++column) {
    columns.emplace_back(data[column].begin(), data[column].end(), valid[column].begin());
    requests[column].values = columns.back();
    requests[column].aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    std::vector<bool> expected_valid{column != 1, true, column != 0};
    expected.emplace_back(sums[column].begin(), sums[column].end(), expected_valid.begin());
  }

  auto harness = cudf::test::memory_resource_test_harness{this->mr()};
  {
    auto result = [&] {
      cudf::test::scoped_current_device_resource temporary_scope{harness.temporary_mr()};
      cudf::groupby::groupby gb(cudf::table_view{{keys}});
      return gb.aggregate(requests, stream, harness.output_mr());
    }();
    ASSERT_EQ(result.second.size(), num_columns);
    auto output_bytes = result.first->alloc_size();
    for (auto const& request : result.second) {
      ASSERT_EQ(request.results.size(), 1);
      output_bytes += request.results.front()->alloc_size();
    }
    harness.expect_resource_usage(output_bytes,
                                  {cudf::test::output_allocation_expectation::EXACT,
                                   cudf::test::temporary_allocation_expectation::SOME},
                                  stream);
    for (int column = 0; column < num_columns; ++column) {
      auto const actual =
        cudf::table_view{{result.first->view().column(0), *result.second[column].results[0]}};
      auto const sorted = cudf::sort(actual, {}, {}, stream);
      CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{expect_keys, expected[column]}},
                                    sorted->view());
    }
  }
  harness.expect_no_live_allocations(stream);
}
