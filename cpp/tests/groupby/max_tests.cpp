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
#include <cudf/dictionary/update_keys.hpp>

#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_max_test : public cudf::test::BaseFixture {};

using K = int32_t;
TYPED_TEST_SUITE(groupby_max_test, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(groupby_max_test, basic)
{
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({6, 9, 8});

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_max_test, empty_cols)
{
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_max_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5});

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_max_test, zero_valid_values)
{
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_max_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  //  { 0, 3,     1, 4, 5,   2, 8,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3, 5, 8, 0}, {1, 1, 1, 0});

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

struct groupby_max_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_max_string_test, basic)
{
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::strings_column_wrapper vals{
    "año", "bit", "₹1", "aaa", "zit", "bat", "aaa", "$1", "₹1", "wut"};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::strings_column_wrapper expect_vals({"año", "zit", "₹1"});

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TEST_F(groupby_max_string_test, zero_valid_values)
{
  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::strings_column_wrapper vals({"", "", ""}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::strings_column_wrapper expect_vals({""}, all_nulls());

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TEST_F(groupby_max_string_test, max_sorted_strings)
{
  // testcase replicated in issue #8717
  cudf::test::strings_column_wrapper keys(
    {"",   "",   "",   "",   "",   "",   "06", "06", "06", "06", "10", "10", "10", "10", "14", "14",
     "14", "14", "18", "18", "18", "18", "22", "22", "22", "22", "26", "26", "26", "26", "30", "30",
     "30", "30", "34", "34", "34", "34", "38", "38", "38", "38", "42", "42", "42", "42"},
    {false, false, false, false, false, false, true, true, true, true, true, true,
     true,  true,  true,  true,  true,  true,  true, true, true, true, true, true,
     true,  true,  true,  true,  true,  true,  true, true, true, true, true, true,
     true,  true,  true,  true,  true,  true,  true, true, true, true});
  cudf::test::strings_column_wrapper vals(
    {"", "", "",   "", "", "", "06", "", "", "", "10", "", "", "", "14", "",
     "", "", "18", "", "", "", "22", "", "", "", "26", "", "", "", "30", "",
     "", "", "34", "", "", "", "38", "", "", "", "42", "", "", ""},
    {false, false, false, false, false, false, true, false, false, false, true, false,
     false, false, true,  false, false, false, true, false, false, false, true, false,
     false, false, true,  false, false, false, true, false, false, false, true, false,
     false, false, true,  false, false, false, true, false, false, false});
  cudf::test::strings_column_wrapper expect_keys(
    {"06", "10", "14", "18", "22", "26", "30", "34", "38", "42", ""},
    {true, true, true, true, true, true, true, true, true, true, false});
  cudf::test::strings_column_wrapper expect_vals(
    {"06", "10", "14", "18", "22", "26", "30", "34", "38", "42", ""},
    {true, true, true, true, true, true, true, true, true, true, false});

  // cudf::test::fixed_width_column_wrapper<size_type> expect_argmax(
  // {6, 10, 14, 18, 22, 26, 30, 34, 38, 42, -1},
  // {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::NO,
                  cudf::null_policy::INCLUDE,
                  cudf::sorted::YES);
}

struct groupby_dictionary_max_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_max_test, basic)
{
  using V = std::string;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{     1,     2,    3,     1,     2,     2,     1,    3,    3,    2 };
  cudf::test::dictionary_column_wrapper<V>  vals{ "año", "bit", "₹1", "aaa", "zit", "bat", "aaa", "$1", "₹1", "wut"};
  cudf::test::fixed_width_column_wrapper<K> expect_keys   {     1,     2,    3 };
  cudf::test::dictionary_column_wrapper<V>  expect_vals_w({ "año", "zit", "₹1" });
  // clang-format on

  auto expect_vals = cudf::dictionary::set_keys(expect_vals_w, vals.keys());

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals->view(),
                  cudf::make_max_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals->view(),
                  cudf::make_max_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

TEST_F(groupby_dictionary_max_test, fixed_width)
{
  using V = int64_t;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{     1,     2,    3,     1,     2,     2,     1,    3,    3,    2 };
  cudf::test::dictionary_column_wrapper<V>  vals{ 0xABC, 0xBBB, 0xF1, 0xAAA, 0xFFF, 0xBAA, 0xAAA, 0x01, 0xF1, 0xEEE};
  cudf::test::fixed_width_column_wrapper<K> expect_keys    {     1,     2,    3 };
  cudf::test::fixed_width_column_wrapper<V>  expect_vals_w({ 0xABC, 0xFFF, 0xF1 });
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals_w,
                  cudf::make_max_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals_w,
                  cudf::make_max_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

template <typename T>
struct GroupByMaxFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupByMaxFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(GroupByMaxFixedPointTest, GroupBySortMaxDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using K          = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    // clang-format on

    auto const expect_keys     = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_max = fp_wrapper{{6, 9, 8}, scale};

    auto agg3 = cudf::make_max_aggregation<cudf::groupby_aggregation>();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_max, std::move(agg3), force_use_sort_impl::YES);
  }
}

TYPED_TEST(GroupByMaxFixedPointTest, GroupByHashMaxDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using K          = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    // clang-format on

    auto const expect_keys     = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_max = fp_wrapper{{6, 9, 8}, scale};

    auto agg7 = cudf::make_max_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals_max, std::move(agg7));
  }
}

struct groupby_max_struct_test : public cudf::test::BaseFixture {};

TEST_F(groupby_max_struct_test, basic)
{
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto const expect_vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{"año", "zit", "₹1"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 5, 3};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_max_struct_test, slice_input)
{
  constexpr int32_t dont_care{1};
  auto const keys_original = cudf::test::fixed_width_column_wrapper<int32_t>{
    dont_care, dont_care, 1, 2, 3, 1, 2, 2, 1, 3, 3, 2, dont_care};
  auto const vals_original = [] {
    auto child1 = cudf::test::strings_column_wrapper{"dont_care",
                                                     "dont_care",
                                                     "año",
                                                     "bit",
                                                     "₹1",
                                                     "aaa",
                                                     "zit",
                                                     "bat",
                                                     "aab",
                                                     "$1",
                                                     "€1",
                                                     "wut",
                                                     "dont_care"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      dont_care, dont_care, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, dont_care};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto const keys        = cudf::slice(keys_original, {2, 12})[0];
  auto const vals        = cudf::slice(vals_original, {2, 12})[0];
  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto const expect_vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{"año", "zit", "₹1"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 5, 3};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_max_struct_test, null_keys_and_values)
{
  constexpr int32_t null{0};
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{
    {1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4}, null_at(7)};
  auto const vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{
      "año", "bit", "₹1", "aaa", "zit", "" /*NULL*/, "" /*NULL*/, "$1", "€1", "wut", "" /*NULL*/};
    auto child2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{9, 8, 7, 6, 5, null, null, 2, 1, 0, null};
    return cudf::test::structs_column_wrapper{{child1, child2}, nulls_at({5, 6, 10})};
  }();

  auto const expect_keys =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4}, no_nulls()};
  auto const expect_vals = [] {
    auto child1 = cudf::test::strings_column_wrapper{"año", "zit", "₹1", "" /*NULL*/};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{9, 5, 7, null};
    return cudf::test::structs_column_wrapper{{child1, child2}, null_at(3)};
  }();

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_max_struct_test, values_with_null_child)
{
  constexpr int32_t null{0};
  {
    auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{1, 1};
    auto const vals = [] {
      auto child1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 1};
      auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{{-1, null}, null_at(1)};
      return cudf::test::structs_column_wrapper{child1, child2};
    }();

    auto const expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>{1};
    auto const expect_vals = [] {
      auto child1 = cudf::test::fixed_width_column_wrapper<int32_t>{1};
      auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{-1};
      return cudf::test::structs_column_wrapper{child1, child2};
    }();

    auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
  }

  {
    auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{1, 1};
    auto const vals = [] {
      auto child1 = cudf::test::fixed_width_column_wrapper<int32_t>{{-1, null}, null_at(1)};
      auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{{null, null}, nulls_at({0, 1})};
      return cudf::test::structs_column_wrapper{child1, child2};
    }();

    auto const expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>{1};
    auto const expect_vals = [] {
      auto child1 = cudf::test::fixed_width_column_wrapper<int32_t>{-1};
      auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{{null}, null_at(0)};
      return cudf::test::structs_column_wrapper{child1, child2};
    }();

    auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
  }
}

struct groupby_max_list_test : public cudf::test::BaseFixture {};

TEST_F(groupby_max_list_test, basic)
{
  using lists = cudf::test::lists_column_wrapper<int32_t>;

  auto const keys        = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 1, 2};
  auto const vals        = lists{{1, 2}, {3, 4}, {5, 6, 7}, {0, 8}, {9, 10}};
  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};
  auto const expect_vals = lists{{1, 2}, {9, 10}, {5, 6, 7}};

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_max_aggregation<cudf::groupby_aggregation>());
}

TEST_F(groupby_max_list_test, slice_input)
{
  using lists = cudf::test::lists_column_wrapper<int32_t>;
  constexpr int32_t dont_care{1};

  auto const keys_original =
    cudf::test::fixed_width_column_wrapper<int32_t>{dont_care, 1, 2, 3, 1, 2, dont_care};
  auto const vals_original =
    lists{{1, 2, 3, 4, 5} /*dont care*/, {1, 2}, {3, 4}, {5, 6, 7}, {0, 8}, {9, 10}};
  auto const keys = cudf::slice(keys_original, {1, 6})[0];
  auto const vals = cudf::slice(vals_original, {1, 6})[0];

  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};
  auto const expect_vals = lists{{1, 2}, {9, 10}, {5, 6, 7}};

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_max_aggregation<cudf::groupby_aggregation>());
}

TEST_F(groupby_max_list_test, null_keys_and_values)
{
  using lists = cudf::test::lists_column_wrapper<int32_t>;
  constexpr int32_t null{0};

  auto const keys =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, null, 1, 2}, null_at(3)};
  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3}, no_nulls()};

  // Null list element.
  {
    auto const vals = lists{{{} /*null*/, {1, 2}, {3, 4}, {5, 6, 7}, {0, 8}, {9, 10}}, null_at(0)};
    auto const expect_vals = lists{{0, 8}, {9, 10}, {3, 4}};
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    expect_vals,
                    cudf::make_max_aggregation<cudf::groupby_aggregation>());
  }

  // Null child element.
  {
    auto const vals        = lists{lists{{0, null}, null_at(1)},
                            lists{1, 2},
                            lists{3, 4},
                            lists{5, 6, 7},
                            lists{0, 8},
                            lists{9, 10}};
    auto const expect_vals = lists{{0, 8}, {9, 10}, {3, 4}};
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    expect_vals,
                    cudf::make_max_aggregation<cudf::groupby_aggregation>());
  }
}

template <typename V>
struct groupby_max_floating_point_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_max_floating_point_test, cudf::test::FloatingPointTypes);

TYPED_TEST(groupby_max_floating_point_test, values_with_infinity)
{
  using T          = TypeParam;
  using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;
  using floats_col = cudf::test::fixed_width_column_wrapper<T, int32_t>;

  auto constexpr inf = std::numeric_limits<T>::infinity();

  auto const keys = int32s_col{1, 2, 1, 2};
  auto const vals = floats_col{static_cast<T>(1), static_cast<T>(1), inf, static_cast<T>(2)};

  auto const expected_keys = int32s_col{1, 2};
  auto const expected_vals = floats_col{inf, static_cast<T>(2)};

  // Related issue: https://github.com/NVIDIA/cudf/issues/11352
  // The issue only occurs in sort-based cudf::aggregation.
  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  test_single_agg(
    keys, vals, expected_keys, expected_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_max_floating_point_test, values_with_nan)
{
  using T          = TypeParam;
  using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;
  using floats_col = cudf::test::fixed_width_column_wrapper<T, int32_t>;

  auto constexpr nan = std::numeric_limits<T>::quiet_NaN();

  auto const keys = int32s_col{1, 1};
  auto const vals = floats_col{nan, nan};

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());

  // Without properly handling NaN, this will hang forever in hash-based aggregate (which is the
  // default back-end for min/max in groupby context).
  // This test is just to verify that the aggregate operation does not hang.
  auto gb_obj       = cudf::groupby::groupby(cudf::table_view({keys}));
  auto const result = gb_obj.aggregate(requests);

  EXPECT_EQ(result.first->num_rows(), 1);
}

// Test the shared memory kernel in hash-based groupby aggregations.
struct groupby_max_hash_based_shmem_kernel_test : cudf::test::BaseFixture {};

TEST_F(groupby_max_hash_based_shmem_kernel_test, all_unique_keys)
{
  auto constexpr block_size      = 128;
  auto constexpr size_multiplier = 4;

  // Since the maximum number of thread blocks (128 threads each) can be 100+ depending on GPU
  // architecture, while the number of rows is just 128*4, each thread block will process exactly
  // 128 rows. Because all thread blocks encounter no more than 128 distinct keys, they will
  // run using the shared memory kernel.
  std::vector<int> h_keys(block_size * size_multiplier);
  std::iota(h_keys.begin(), h_keys.end(), 0);

  cudf::test::fixed_width_column_wrapper<int> keys(h_keys.begin(), h_keys.end());
  cudf::test::fixed_width_column_wrapper<int> expect_keys(h_keys.begin(), h_keys.end());

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  // Keys are the same as values.
  test_single_agg(keys, keys, expect_keys, expect_keys, std::move(agg));
}

TEST_F(groupby_max_hash_based_shmem_kernel_test, repeated_keys)
{
  auto constexpr block_size      = 128;
  auto constexpr size_multiplier = 10000;

  // Repeat the keys for every 128 rows, thus we will have only 128 distinct keys in total.
  // Because all thread blocks encounter no more than 128 distinct keys, they will run using the
  // shared memory kernel.
  std::vector<int> h_keys(block_size * size_multiplier);
  auto it = h_keys.begin();
  while (it != h_keys.end()) {
    std::iota(it, it + block_size, 0);
    it += block_size;
  }

  cudf::test::fixed_width_column_wrapper<int> expect_keys(h_keys.begin(),
                                                          h_keys.begin() + block_size);
  std::ranges::shuffle(h_keys.begin(), h_keys.end(), std::default_random_engine());
  cudf::test::fixed_width_column_wrapper<int> keys(h_keys.begin(), h_keys.end());

  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  // Keys are the same as values.
  test_single_agg(keys, keys, expect_keys, expect_keys, std::move(agg));
}
