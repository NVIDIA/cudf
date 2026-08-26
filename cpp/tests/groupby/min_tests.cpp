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

using namespace cudf::test::iterators;

template <typename V>
struct groupby_min_test : public cudf::test::BaseFixture {};

using K = int32_t;
TYPED_TEST_SUITE(groupby_min_test, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(groupby_min_test, basic)
{
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0, 1, 2});

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_min_test, empty_cols)
{
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_min_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5});

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_min_test, zero_valid_values)
{
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_min_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  //  { 3, 6,     1, 4, 9,   2, 8,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({3, 1, 2, 0}, {1, 1, 1, 0});

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

struct groupby_min_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_min_string_test, basic)
{
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::strings_column_wrapper vals{
    "año", "bit", "₹1", "aaa", "zit", "bat", "aaa", "$1", "₹1", "wut"};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::strings_column_wrapper expect_vals({"aaa", "bat", "$1"});

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TEST_F(groupby_min_string_test, zero_valid_values)
{
  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::strings_column_wrapper vals({"", "", ""}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::strings_column_wrapper expect_vals({""}, all_nulls());

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));

  auto agg2 = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg2), force_use_sort_impl::YES);
}

TEST_F(groupby_min_string_test, min_sorted_strings)
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

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  std::move(agg),
                  force_use_sort_impl::NO,
                  cudf::null_policy::INCLUDE,
                  cudf::sorted::YES);
}

struct groupby_dictionary_min_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_min_test, basic)
{
  using V = std::string;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{     1,     2,    3,     1,     2,     2,     1,    3,    3,    2 };
  cudf::test::dictionary_column_wrapper<V>  vals{ "año", "bit", "₹1", "aaa", "zit", "bat", "aaa", "$1", "₹1", "wut"};
  cudf::test::fixed_width_column_wrapper<K> expect_keys   {     1,     2,    3 };
  cudf::test::dictionary_column_wrapper<V>  expect_vals_w({ "aaa", "bat", "$1" });
  // clang-format on

  auto expect_vals = cudf::dictionary::set_keys(expect_vals_w, vals.keys());

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals->view(),
                  cudf::make_min_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals->view(),
                  cudf::make_min_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

TEST_F(groupby_dictionary_min_test, fixed_width)
{
  using V = int64_t;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{     1,     2,    3,     1,     2,     2,     1,    3,    3,    2 };
  cudf::test::dictionary_column_wrapper<V>  vals{ 0xABC, 0xBBB, 0xF1, 0xAAA, 0xFFF, 0xBAA, 0xAAA, 0x01, 0xF1, 0xEEE};
  cudf::test::fixed_width_column_wrapper<K> expect_keys    {     1,     2,    3 };
  cudf::test::fixed_width_column_wrapper<V>  expect_vals_w({ 0xAAA, 0xBAA, 0x01 });
  // clang-format on

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals_w,
                  cudf::make_min_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals_w,
                  cudf::make_min_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

template <typename T>
struct GroupByMinFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupByMinFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(GroupByMinFixedPointTest, GroupBySortMinDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    // clang-format on

    auto const expect_keys     = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_min = fp_wrapper{{0, 1, 2}, scale};

    auto agg2 = cudf::make_min_aggregation<cudf::groupby_aggregation>();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_min, std::move(agg2), force_use_sort_impl::YES);
  }
}

TYPED_TEST(GroupByMinFixedPointTest, GroupByHashMinDecimalAsValue)
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
    auto const expect_vals_min = fp_wrapper{{0, 1, 2}, scale};

    auto agg6 = cudf::make_min_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals_min, std::move(agg6));
  }
}

struct groupby_min_struct_test : public cudf::test::BaseFixture {};

TEST_F(groupby_min_struct_test, basic)
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
    auto child1 = cudf::test::strings_column_wrapper{"aaa", "bat", "$1"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{4, 6, 8};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_min_struct_test, slice_input)
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
    auto child1 = cudf::test::strings_column_wrapper{"aaa", "bat", "$1"};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{4, 6, 8};
    return cudf::test::structs_column_wrapper{{child1, child2}};
  }();

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_min_struct_test, null_keys_and_values)
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
    auto child1 = cudf::test::strings_column_wrapper{"aaa", "bit", "€1", "" /*NULL*/};
    auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{6, 8, 1, null};
    return cudf::test::structs_column_wrapper{{child1, child2}, null_at(3)};
  }();

  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_min_struct_test, values_with_null_child)
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
      auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{{null}, null_at(0)};
      return cudf::test::structs_column_wrapper{child1, child2};
    }();

    auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
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
      auto child1 = cudf::test::fixed_width_column_wrapper<int32_t>{{null}, null_at(0)};
      auto child2 = cudf::test::fixed_width_column_wrapper<int32_t>{{null}, null_at(0)};
      return cudf::test::structs_column_wrapper{child1, child2};
    }();

    auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
  }
}

struct groupby_min_list_test : public cudf::test::BaseFixture {};

TEST_F(groupby_min_list_test, basic)
{
  using lists = cudf::test::lists_column_wrapper<int32_t>;

  auto const keys        = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 1, 2};
  auto const vals        = lists{{1, 2}, {3, 4}, {5, 6, 7}, {0, 8}, {9, 10}};
  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};
  auto const expect_vals = lists{{0, 8}, {3, 4}, {5, 6, 7}};

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_min_aggregation<cudf::groupby_aggregation>());
}

TEST_F(groupby_min_list_test, slice_input)
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
  auto const expect_vals = lists{{0, 8}, {3, 4}, {5, 6, 7}};

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_min_aggregation<cudf::groupby_aggregation>());
}

TEST_F(groupby_min_list_test, null_keys_and_values)
{
  using lists = cudf::test::lists_column_wrapper<int32_t>;
  constexpr int32_t null{0};

  auto const keys =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, null, 1, 2}, null_at(3)};
  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3}, no_nulls()};

  // Null list element.
  {
    auto const vals = lists{{{} /*null*/, {1, 2}, {3, 4}, {5, 6, 7}, {0, 8}, {9, 10}}, null_at(0)};
    auto const expect_vals = lists{{0, 8}, {1, 2}, {3, 4}};
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    expect_vals,
                    cudf::make_min_aggregation<cudf::groupby_aggregation>());
  }

  // Null child element.
  {
    auto const vals        = lists{lists{{0, null}, null_at(1)},
                            lists{1, 2},
                            lists{3, 4},
                            lists{5, 6, 7},
                            lists{0, 8},
                            lists{9, 10}};
    auto const expect_vals = lists{lists{{0, null}, null_at(1)}, {1, 2}, {3, 4}};
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    expect_vals,
                    cudf::make_min_aggregation<cudf::groupby_aggregation>());
  }
}

template <typename V>
struct groupby_min_floating_point_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_min_floating_point_test, cudf::test::FloatingPointTypes);

TYPED_TEST(groupby_min_floating_point_test, values_with_infinity)
{
  using T          = TypeParam;
  using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;
  using floats_col = cudf::test::fixed_width_column_wrapper<T, int32_t>;

  auto constexpr inf = std::numeric_limits<T>::infinity();

  auto const keys = int32s_col{1, 2, 1, 2};
  auto const vals = floats_col{static_cast<T>(1), static_cast<T>(1), -inf, static_cast<T>(2)};

  auto const expected_keys = int32s_col{1, 2};
  auto const expected_vals = floats_col{-inf, static_cast<T>(1)};

  // Related issue: https://github.com/NVIDIA/cudf/issues/11352
  // The issue only occurs in sort-based aggregation.
  auto agg = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  test_single_agg(
    keys, vals, expected_keys, expected_vals, std::move(agg), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_min_floating_point_test, values_with_nan)
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
  requests[0].aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());

  // Without properly handling NaN, this will hang forever in hash-based aggregate (which is the
  // default back-end for min/max in groupby context).
  // This test is just to verify that the aggregate operation does not hang.
  auto gb_obj       = cudf::groupby::groupby(cudf::table_view({keys}));
  auto const result = gb_obj.aggregate(requests);

  EXPECT_EQ(result.first->num_rows(), 1);
}
