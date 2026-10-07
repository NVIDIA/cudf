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
#include <limits>
#include <type_traits>
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

template <typename T>
struct groupby_small_key_domain_test : public cudf::test::BaseFixture {};

using small_domain_types = cudf::test::Types<bool, int16_t, uint16_t>;
TYPED_TEST_SUITE(groupby_small_key_domain_test, small_domain_types);

TYPED_TEST(groupby_small_key_domain_test, CompleteNullableDomain)
{
  using K                            = TypeParam;
  constexpr cudf::size_type domain   = std::is_same_v<K, bool> ? 2 : 1 << 16;
  constexpr cudf::size_type states   = domain + 1;
  constexpr cudf::size_type repeats  = (1 << 21) / states + 1;
  constexpr cudf::size_type num_rows = states * repeats;
  static_assert(num_rows > (1 << 21));

  auto const key_at = [](cudf::size_type i) {
    auto const state = (i % states) % domain;
    return static_cast<K>(std::numeric_limits<K>::lowest() + state);
  };
  auto const valid_at = [](cudf::size_type i) { return i % states < domain; };
  auto const key_it   = cudf::detail::make_counting_transform_iterator(0, key_at);
  auto const valid_it = cudf::detail::make_counting_transform_iterator(0, valid_at);
  auto const keys = cudf::test::fixed_width_column_wrapper<K>(key_it, key_it + num_rows, valid_it);
  auto const counts = cuda::make_constant_iterator(repeats);

  // The full domain, including its null state, exceeds the sampling threshold. Both null
  // policies must retain all groups and their counts when sizing from the domain bound.
  for (auto const policy : {cudf::null_policy::INCLUDE, cudf::null_policy::EXCLUDE}) {
    SCOPED_TRACE(static_cast<int>(policy));
    auto const num_groups = domain + (policy == cudf::null_policy::INCLUDE);
    auto const expected_keys =
      cudf::test::fixed_width_column_wrapper<K>(key_it, key_it + num_groups, valid_it);
    auto const expected_counts =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>(counts, counts + num_groups);
    test_single_agg(
      keys,
      keys,
      expected_keys,
      expected_counts,
      cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE),
      force_use_sort_impl::NO,
      policy);
  }
}

TEST_F(groupby_key_shape_test, CompleteNullableByteKeyDomains)
{
  constexpr cudf::size_type states   = 257;
  constexpr cudf::size_type domain   = states * states;
  constexpr cudf::size_type repeats  = 32;
  constexpr cudf::size_type num_rows = domain * repeats;
  static_assert(num_rows > (1 << 21));
  auto const first = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<uint8_t>((i / states) % states); });
  auto const second = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<int8_t>(i % states - 128); });
  auto const first_valid = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return (i / states) % states < 256; });
  auto const second_valid =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % states < 256; });
  auto const keys_a =
    cudf::test::fixed_width_column_wrapper<uint8_t>(first, first + num_rows, first_valid);
  auto const keys_b =
    cudf::test::fixed_width_column_wrapper<int8_t>(second, second + num_rows, second_valid);
  auto const counts = cuda::make_constant_iterator(repeats);
  auto const stream = cudf::test::get_default_stream();

  for (auto const policy : {cudf::null_policy::INCLUDE, cudf::null_policy::EXCLUDE}) {
    SCOPED_TRACE(static_cast<int>(policy));
    auto const expected_states = policy == cudf::null_policy::INCLUDE ? states : 256;
    auto const num_groups      = expected_states * expected_states;
    auto const expected_row    = cudf::detail::make_counting_transform_iterator(
      0,
      [expected_states](auto i) { return (i / expected_states) * states + i % expected_states; });
    auto const expected_a = cudf::test::fixed_width_column_wrapper<uint8_t>(
      cuda::make_permutation_iterator(first, expected_row),
      cuda::make_permutation_iterator(first, expected_row + num_groups),
      cuda::make_permutation_iterator(first_valid, expected_row));
    auto const expected_b = cudf::test::fixed_width_column_wrapper<int8_t>(
      cuda::make_permutation_iterator(second, expected_row),
      cuda::make_permutation_iterator(second, expected_row + num_groups),
      cuda::make_permutation_iterator(second_valid, expected_row));
    auto const expected_counts =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>(counts, counts + num_groups);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = keys_a;
    requests[0].aggregations.push_back(
      cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
    cudf::groupby::groupby gb(cudf::table_view{{keys_a, keys_b}}, policy);
    auto const [keys, result] = gb.aggregate(requests, stream);
    auto const actual         = cudf::sort(
      cudf::table_view{{keys->view().column(0), keys->view().column(1), *result[0].results[0]}},
      {},
      {},
      stream);
    auto const expected =
      cudf::sort(cudf::table_view{{expected_a, expected_b, expected_counts}}, {}, {}, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), actual->view());
  }
}

TEST_F(groupby_key_shape_test, SmallKeyDomainFallback)
{
  constexpr cudf::size_type num_groups = 1 << 17;
  constexpr cudf::size_type repeats    = 17;
  constexpr cudf::size_type num_rows   = num_groups * repeats;
  static_assert(num_rows > (1 << 21));
  auto const first = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<uint8_t>(i % 256); });
  auto const second = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<uint8_t>((i / 256) % 256); });
  auto const third = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<uint8_t>((i % num_groups) / 65536); });
  auto const wide = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<int64_t>((i % num_groups) / 256); });
  auto const keys_a    = cudf::test::fixed_width_column_wrapper<uint8_t>(first, first + num_rows);
  auto const keys_b    = cudf::test::fixed_width_column_wrapper<uint8_t>(second, second + num_rows);
  auto const keys_c    = cudf::test::fixed_width_column_wrapper<uint8_t>(third, third + num_rows);
  auto const keys_wide = cudf::test::fixed_width_column_wrapper<int64_t>(wide, wide + num_rows);
  auto const expected_a =
    cudf::test::fixed_width_column_wrapper<uint8_t>(first, first + num_groups);
  auto const expected_b =
    cudf::test::fixed_width_column_wrapper<uint8_t>(second, second + num_groups);
  auto const expected_c =
    cudf::test::fixed_width_column_wrapper<uint8_t>(third, third + num_groups);
  auto const expected_wide =
    cudf::test::fixed_width_column_wrapper<int64_t>(wide, wide + num_groups);
  auto const counts = cuda::make_constant_iterator(repeats);
  auto const expected_counts =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(counts, counts + num_groups);
  auto const stream = cudf::test::get_default_stream();

  // An unsupported wide type and a product of three small domains both retain sampling.
  std::vector<cudf::table_view> key_tables{cudf::table_view{{keys_a, keys_wide}},
                                           cudf::table_view{{keys_a, keys_b, keys_c}}};
  std::vector<cudf::table_view> expected_tables{
    cudf::table_view{{expected_a, expected_wide, expected_counts}},
    cudf::table_view{{expected_a, expected_b, expected_c, expected_counts}}};
  for (std::size_t i = 0; i < key_tables.size(); ++i) {
    SCOPED_TRACE(i);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = keys_a;
    requests[0].aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
    cudf::groupby::groupby gb(key_tables[i]);
    auto const [keys, result] = gb.aggregate(requests, stream);
    auto const key_view       = keys->view();
    std::vector<cudf::column_view> actual_columns(key_view.begin(), key_view.end());
    actual_columns.push_back(result[0].results[0]->view());
    auto const actual   = cudf::sort(cudf::table_view{actual_columns}, {}, {}, stream);
    auto const expected = cudf::sort(expected_tables[i], {}, {}, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), actual->view());
  }
}

TEST_F(groupby_key_shape_test, NearlyDistinctSampleUnderestimatesPopulation)
{
  constexpr cudf::size_type num_rows    = 1 << 21;
  constexpr cudf::size_type stride      = 67;
  constexpr cudf::size_type sample_keys = 29'500;
  constexpr cudf::size_type num_samples = (num_rows + stride - 1) / stride;

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

struct groupby_minmax_fusion_test : public groupby_key_shape_test {
  void test_minmax(std::vector<int32_t> const& sizes)
  {
    auto const stream = cudf::test::get_default_stream();
    std::vector<int32_t> keys_data, values_data;
    std::vector<bool> validity;
    for (std::size_t group = 0; group < sizes.size(); ++group) {
      for (int32_t row = 0; row < sizes[group]; ++row) {
        keys_data.push_back(static_cast<int32_t>(group));
        values_data.push_back(row + 1);
        validity.push_back(group % 3 == 1 || (group % 3 == 2 && row == sizes[group] - 1));
      }
    }
    cudf::test::fixed_width_column_wrapper<int32_t> keys(keys_data.begin(), keys_data.end());
    cudf::test::fixed_width_column_wrapper<int32_t> values(values_data.begin(), values_data.end());
    cudf::test::fixed_width_column_wrapper<int32_t> nullable_values(
      values_data.begin(), values_data.end(), validity.begin());
    auto const group_ids = cuda::counting_iterator<int32_t>{0};
    cudf::test::fixed_width_column_wrapper<int32_t> expected_keys(
      group_ids, group_ids + static_cast<int32_t>(sizes.size()));

    for (bool nullable : {false, true}) {
      std::vector<int32_t> minima;
      std::vector<bool> expected_validity;
      for (std::size_t group = 0; group < sizes.size(); ++group) {
        minima.push_back(nullable && group % 3 == 2 ? sizes[group] : 1);
        expected_validity.push_back(!nullable || group % 3 != 0);
      }
      using wrapper     = cudf::test::fixed_width_column_wrapper<int32_t>;
      auto expected_min = nullable
                            ? wrapper(minima.begin(), minima.end(), expected_validity.begin())
                            : wrapper(minima.begin(), minima.end());
      auto expected_max = nullable ? wrapper(sizes.begin(), sizes.end(), expected_validity.begin())
                                   : wrapper(sizes.begin(), sizes.end());
      for (bool reverse : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "nullable=" << nullable << " reverse=" << reverse);
        std::vector<cudf::groupby::aggregation_request> requests(1);
        requests[0].values =
          nullable ? cudf::column_view(nullable_values) : cudf::column_view(values);
        auto& aggs = requests[0].aggregations;
        aggs.push_back(reverse ? cudf::make_max_aggregation<cudf::groupby_aggregation>()
                               : cudf::make_min_aggregation<cudf::groupby_aggregation>());
        aggs.push_back(reverse ? cudf::make_min_aggregation<cudf::groupby_aggregation>()
                               : cudf::make_max_aggregation<cudf::groupby_aggregation>());
        cudf::groupby::groupby gb(cudf::table_view{{keys}});
        auto const [result_keys, results] = gb.aggregate(requests, stream);
        ASSERT_EQ(results.size(), 1);
        ASSERT_EQ(results[0].results.size(), 2);
        auto const actual = cudf::table_view{{result_keys->view().column(0),
                                              *results[0].results[reverse ? 1 : 0],
                                              *results[0].results[reverse ? 0 : 1]}};
        auto const sorted = cudf::sort(actual, {}, {}, stream);
        CUDF_TEST_EXPECT_TABLES_EQUAL(
          (cudf::table_view{{expected_keys, expected_min, expected_max}}), sorted->view());
      }
    }
  }
};

TEST_F(groupby_minmax_fusion_test, FusedMinMaxWithoutPartialReductions)
{
  test_minmax({1, 2, 31, 32, 33, 1024});
}

TEST_F(groupby_minmax_fusion_test, MinMaxWithLongGroupFallback)
{
  test_minmax({1, 31, 32, 33, 1024, 1025, 4097});
}

TEST_F(groupby_minmax_fusion_test, InterleavedRequestsPreserveResultOrder)
{
  auto const stream = cudf::test::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int32_t> keys{2, 0, 1, 0, 2, 1, 0, 1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> values_a{3, 8, -2, 4, 1, 7, 6, 5, 9};
  cudf::test::fixed_width_column_wrapper<int32_t> values_b{10, 2, 6, -8, 15, 3, 0, 11, 14};
  std::vector<cudf::groupby::aggregation_request> requests(3);
  requests[0].values = values_a;
  requests[0].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  requests[1].values = values_b;
  requests[1].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  requests[1].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  requests[2].values = values_a;
  requests[2].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());

  // MIN(A), MIN(B), MAX(B), MAX(A) retains the two same-kind batches instead of splitting
  // them around a fused pair for B. A appears in two requests, so verify reconstruction too.
  cudf::groupby::groupby gb(cudf::table_view{{keys}});
  auto const [result_keys, results] = gb.aggregate(requests, stream);
  ASSERT_EQ(results.size(), 3);
  ASSERT_EQ(results[0].results.size(), 1);
  ASSERT_EQ(results[1].results.size(), 2);
  ASSERT_EQ(results[2].results.size(), 1);
  auto const actual = cudf::sort(cudf::table_view{{result_keys->view().column(0),
                                                   *results[0].results[0],
                                                   *results[1].results[0],
                                                   *results[1].results[1],
                                                   *results[2].results[0]}},
                                 {},
                                 {},
                                 stream);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_min_a{4, -2, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_min_b{-8, 3, 10};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_max_b{2, 11, 15};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_max_a{8, 7, 9};
  CUDF_TEST_EXPECT_TABLES_EQUAL(
    (cudf::table_view{
      {expected_keys, expected_min_a, expected_min_b, expected_max_b, expected_max_a}}),
    actual->view());
}

TEST_F(groupby_minmax_fusion_test, FloatingPointSpecialValuesMatchSeparateReductions)
{
  auto const stream  = cudf::test::get_default_stream();
  auto constexpr nan = std::numeric_limits<double>::quiet_NaN();
  auto constexpr inf = std::numeric_limits<double>::infinity();
  // Separate groupby calls can reduce rows in different orders. Exact comparisons must not mix
  // valid NaNs with other values, since the MIN/MAX operators are order-sensitive for NaNs.
  std::vector<std::vector<double>> const patterns{{nan, nan},
                                                  {-5.0, -inf, -3.0, 0.0, -0.0, 3.0, inf, 5.0},
                                                  {0.0, 0.0},
                                                  {-0.0, -0.0},
                                                  {0.0, -0.0},
                                                  {inf, -inf},
                                                  {2.0, inf, -inf}};
  std::vector<int32_t> keys_data;
  std::vector<double> values_data;
  std::vector<double> nullable_values_data;
  std::vector<bool> validity;
  for (std::size_t group = 0; group < patterns.size(); ++group) {
    // Repetition exercises both scalar and warp reductions without long-group partials.
    for (int repeat = 0; repeat < 5; ++repeat) {
      for (auto const value : patterns[group]) {
        keys_data.push_back(static_cast<int32_t>(group));
        values_data.push_back(value);
        validity.push_back(group != 5 && (group != 6 || value == 2.0));
        // Invalid NaN payloads must not affect the valid finite reduction results.
        nullable_values_data.push_back(validity.back() ? value : nan);
      }
    }
  }
  cudf::test::fixed_width_column_wrapper<int32_t> keys(keys_data.begin(), keys_data.end());
  cudf::test::fixed_width_column_wrapper<double> values(values_data.begin(), values_data.end());
  cudf::test::fixed_width_column_wrapper<double> nullable_values(
    nullable_values_data.begin(), nullable_values_data.end(), validity.begin());
  for (bool nullable : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "nullable=" << nullable);
    auto const aggregate = [&](bool minimum, bool maximum) {
      std::vector<cudf::groupby::aggregation_request> requests(1);
      requests[0].values =
        nullable ? cudf::column_view(nullable_values) : cudf::column_view(values);
      if (minimum) {
        requests[0].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
      }
      if (maximum) {
        requests[0].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
      }
      cudf::groupby::groupby gb(cudf::table_view{{keys}});
      auto const [result_keys, results] = gb.aggregate(requests, stream);
      std::vector<cudf::column_view> columns{result_keys->view().column(0)};
      for (auto const& result : results[0].results) {
        columns.push_back(result->view());
      }
      return cudf::sort(cudf::table_view{columns}, {}, {}, stream);
    };
    auto const fused   = aggregate(true, true);
    auto const minimum = aggregate(true, false);
    auto const maximum = aggregate(false, true);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(minimum->view().column(0), fused->view().column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(maximum->view().column(0), fused->view().column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(minimum->view().column(1), fused->view().column(1));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(maximum->view().column(1), fused->view().column(2));
    // Column equality treats signed zeros as equal. Uniform zero groups have unambiguous signs.
    for (auto const index : {1, 2}) {
      auto const fused_values =
        cudf::test::to_host<double>(fused->view().column(index), stream).first;
      auto const separate_values =
        cudf::test::to_host<double>((index == 1 ? minimum : maximum)->view().column(1), stream)
          .first;
      for (auto const group : {2, 3}) {
        EXPECT_EQ(std::signbit(separate_values[group]), std::signbit(fused_values[group]));
      }
    }
  }
}

TEST_F(groupby_minmax_fusion_test, MixedNanValuesComplete)
{
  auto const stream  = cudf::test::get_default_stream();
  auto constexpr nan = std::numeric_limits<double>::quiet_NaN();
  auto constexpr inf = std::numeric_limits<double>::infinity();
  std::vector<double> const pattern{nan, -inf, -3.0, 0.0, -0.0, 3.0, inf, nan};
  std::vector<int32_t> keys_data;
  std::vector<double> values_data;
  // Exercise scalar and warp reductions with mixed valid NaNs, without comparing order-sensitive
  // values. Like the existing MIN/MAX NaN tests, verify that aggregation completes successfully.
  for (int32_t group = 0; group < 2; ++group) {
    for (int repeat = 0; repeat < (group == 0 ? 1 : 5); ++repeat) {
      for (auto const value : pattern) {
        keys_data.push_back(group);
        values_data.push_back(value);
      }
    }
  }
  cudf::test::fixed_width_column_wrapper<int32_t> keys(keys_data.begin(), keys_data.end());
  cudf::test::fixed_width_column_wrapper<double> values(values_data.begin(), values_data.end());
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::groupby gb(cudf::table_view{{keys}});
  auto const [result_keys, results] = gb.aggregate(requests, stream);
  auto const sorted_keys            = cudf::sort(result_keys->view(), {}, {}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted_keys->view().column(0));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].results.size(), 2);
  for (auto const& result : results[0].results) {
    EXPECT_EQ(result->size(), 2);
    EXPECT_EQ(result->null_count(), 0);
  }
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

TEST_F(groupby_key_shape_test, OutputAndTemporaryResourcesForCompoundAndFusedAggregations)
{
  auto const stream = cudf::test::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int32_t> keys_a{{0, 0, 1, 1, 2, 2},
                                                         {true, true, true, true, false, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> keys_b{{0, 0, 0, 0, 0, 0},
                                                         {true, true, true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> values{1, 3, 2, 4, 5, 6};
  cudf::test::fixed_width_column_wrapper<int32_t> null_values{
    {1, 3, 2, 4, 5, 6}, {true, true, false, false, true, true}};
  cudf::test::structs_column_wrapper nested_keys{{keys_a, keys_b},
                                                 {true, true, true, true, false, true}};
  cudf::test::strings_column_wrapper string_keys{{"a", "a", "b", "b", "c", "c"},
                                                 {true, true, true, true, false, true}};
  cudf::test::structs_column_wrapper nested_string_keys{{string_keys, keys_a},
                                                        {true, false, true, true, true, true}};
  // A sliced nullable string child exercises null counting and nonempty-null sanitization.
  auto const sliced_nested_keys = cudf::slice(nested_string_keys, {1, 5}).front();
  std::vector<cudf::table_view> key_tables{cudf::table_view{{keys_a, keys_b}},
                                           cudf::table_view{{nested_keys}},
                                           cudf::table_view{{string_keys}},
                                           cudf::table_view{{nested_string_keys}},
                                           cudf::table_view{{sliced_nested_keys}}};
  for (auto const& keys : key_tables) {
    for (bool nullable : {false, true}) {
      for (int mode = 0; mode < 5; ++mode) {
        SCOPED_TRACE(::testing::Message() << "nullable=" << nullable << " mode=" << mode);
        auto const input = nullable ? cudf::column_view(null_values) : cudf::column_view(values);
        std::vector<cudf::groupby::aggregation_request> requests(1);
        requests[0].values = cudf::slice(input, {0, keys.num_rows()}).front();
        auto& aggs         = requests[0].aggregations;
        if (mode == 0) {
          aggs.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
          aggs.push_back(cudf::make_variance_aggregation<cudf::groupby_aggregation>());
          aggs.push_back(cudf::make_std_aggregation<cudf::groupby_aggregation>());
          aggs.push_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());
        } else if (mode == 1) {
          aggs.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
          aggs.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
          aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        } else if (mode == 2) {
          aggs.push_back(cudf::make_argmin_aggregation<cudf::groupby_aggregation>());
          aggs.push_back(cudf::make_argmax_aggregation<cudf::groupby_aggregation>());
          aggs.push_back(cudf::make_product_aggregation<cudf::groupby_aggregation>());
        } else if (mode == 3) {
          aggs.push_back(cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>());
        } else {
          aggs.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
          aggs.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
        }
        cudf::groupby::groupby expected_gb(keys);
        auto expected = expected_gb.aggregate(requests, stream, this->mr());
        auto harness  = cudf::test::memory_resource_test_harness{this->mr()};
        {
          auto result = [&] {
            cudf::test::scoped_current_device_resource temporary_scope{harness.temporary_mr()};
            cudf::groupby::groupby gb(keys);
            auto result = gb.aggregate(requests, stream, harness.output_mr());
            harness.synchronize(stream);
            return result;
          }();
          auto output_bytes = result.first->alloc_size();
          for (auto const& column : result.second[0].results) {
            output_bytes += column->alloc_size();
          }
          // The existing variance/std helpers may discard masks allocated on the output resource.
          auto const output_expectation =
            mode == 0 ? cudf::test::output_allocation_expectation::AT_LEAST_LIVE
                      : cudf::test::output_allocation_expectation::EXACT;
          harness.expect_resource_usage(
            output_bytes,
            {output_expectation, cudf::test::temporary_allocation_expectation::SOME},
            stream);
          auto actual_keys = result.first->view();
          std::vector<cudf::column_view> actual_views(actual_keys.begin(), actual_keys.end());
          auto expected_keys = expected.first->view();
          std::vector<cudf::column_view> expected_views(expected_keys.begin(), expected_keys.end());
          for (auto const& column : result.second[0].results) {
            actual_views.push_back(column->view());
          }
          for (auto const& column : expected.second[0].results) {
            expected_views.push_back(column->view());
          }
          auto actual_sorted   = cudf::sort(cudf::table_view{actual_views}, {}, {}, stream);
          auto expected_sorted = cudf::sort(cudf::table_view{expected_views}, {}, {}, stream);
          CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sorted->view(), actual_sorted->view());
        }
        harness.expect_no_live_allocations(stream);
      }
    }
  }
}

TEST_F(groupby_key_shape_test, FusedValidityAcrossReductionBoundaries)
{
  // Adjacent groups share mask words; sizes cross warp, block, and long-segment boundaries.
  std::vector<cudf::size_type> const sizes{1, 31, 32, 33, 255, 256, 257, 4097};
  constexpr cudf::size_type num_groups = 65;
  std::vector<int32_t> keys;
  std::vector<int64_t> values;
  std::vector<bool> valid;
  std::vector<cudf::size_type> minima, maxima;
  std::vector<int64_t> sums;
  std::vector<bool> group_valid;
  for (cudf::size_type group = 0; group < num_groups; ++group) {
    auto const size  = sizes[group % sizes.size()];
    auto const start = static_cast<cudf::size_type>(keys.size());
    int64_t sum{0};
    for (cudf::size_type row = 0; row < size; ++row) {
      // All-null, all-valid, and only-the-last-row-valid groups exercise partial merging.
      bool const is_valid = group % 3 == 1 || (group % 3 == 2 && row == size - 1);
      keys.push_back(group);
      values.push_back(row + 1);
      valid.push_back(is_valid);
      if (is_valid) { sum += row + 1; }
    }
    minima.push_back(group % 3 == 1 ? start : start + size - 1);
    maxima.push_back(start + size - 1);
    sums.push_back(sum);
    group_valid.push_back(group % 3 != 0);
  }
  cudf::test::fixed_width_column_wrapper<int32_t> key_column(keys.begin(), keys.end());
  cudf::test::fixed_width_column_wrapper<int64_t> value_column(
    values.begin(), values.end(), valid.begin());
  auto const group_ids = cuda::counting_iterator<cudf::size_type>{0};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys(group_ids, group_ids + num_groups);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_min(
    minima.begin(), minima.end(), group_valid.begin());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_max(
    maxima.begin(), maxima.end(), group_valid.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sum(sums.begin(), sums.end());
  auto const no_overflow = cuda::make_constant_iterator(false);
  cudf::test::fixed_width_column_wrapper<bool> expected_flags(no_overflow,
                                                              no_overflow + num_groups);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(expected_sum.release());
  children.push_back(expected_flags.release());
  auto [mask, null_count] =
    cudf::test::detail::make_null_mask(group_valid.begin(), group_valid.end());
  auto expected_overflow =
    cudf::create_structs_hierarchy(num_groups, std::move(children), null_count, std::move(mask));
  test_single_agg(key_column,
                  value_column,
                  expected_keys,
                  expected_min,
                  cudf::make_argmin_aggregation<cudf::groupby_aggregation>());
  test_single_agg(key_column,
                  value_column,
                  expected_keys,
                  expected_max,
                  cudf::make_argmax_aggregation<cudf::groupby_aggregation>());
  test_single_agg(key_column,
                  value_column,
                  expected_keys,
                  *expected_overflow,
                  cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>());
}
