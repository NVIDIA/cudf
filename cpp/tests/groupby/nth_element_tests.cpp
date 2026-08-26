/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

using namespace cudf::test::iterators;

template <typename V>
struct groupby_nth_element_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_nth_element_test, cudf::test::AllTypes);

TYPED_TEST(groupby_nth_element_test, basic)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  // keys                                                 {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  // vals                                                 {0, 3, 6, 1, 4, 5, 9, 2, 7, 8};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  // groupby.first()
  auto agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0);
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals0({0, 1, 2});
  test_single_agg(keys, vals, expect_keys, expect_vals0, std::move(agg));

  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(1);
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals1({3, 4, 7});
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg));

  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(2);
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals2({6, 5, 8});
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, empty_cols)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, basic_out_of_bounds)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 3, 2, 2, 9});

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  auto agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(3);
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals({0, 9, 0}, {0, 1, 0});
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, negative)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  // keys                                                 {1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  // vals                                                 {0, 3, 6, 1, 4, 5, 9, 2, 7, 8};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  // groupby.last()
  auto agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-1);
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals0({6, 9, 8});
  test_single_agg(keys, vals, expect_keys, expect_vals0, std::move(agg));

  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-2);
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals1({3, 5, 7});
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg));

  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-3);
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals2({0, 4, 2});
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, negative_out_of_bounds)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 3, 2, 2, 9});

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  auto agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-4);
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals({0, 1, 0}, {0, 1, 0});
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, zero_valid_keys)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V, int32_t> vals({3, 4, 5});

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, zero_valid_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V, int32_t> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals({3}, all_nulls());

  auto agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, null_keys_and_values)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                          {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  // keys   {1, 1, 1   2,2,2,2    3,3,    4}
  // vals   {-,3,6,    1,4,-,9,   2,8,    -}
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals({-1, 1, 2, -1}, {0, 1, 1, 0});

  auto agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, null_keys_and_values_out_of_bounds)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V, int32_t> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                          {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});
  // {1, 1, 1    2, 2, 2,    3, 3,   4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  // {-,3,6,     1,4,-,9,    2,8,    -}
  //  value,     null,       out,    out
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals({6, -1, -1, -1}, {1, 0, 0, 0});

  auto agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(2);
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, exclude_nulls)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 3, 1, 2, 2, 1, 3, 3, 2, 4, 4, 2},
    {true, true, true, true, true, true, true, true, false, true, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<V, int32_t> vals(
    {0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 4, 4, 2}, {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0});

  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  // keys    {1, 1, 1    2, 2, 2, 2      3, 3, 3    4}
  // vals    {-, 3, 6    1, 4, -, 9, -   2, 2, 8,   4,-}
  //       0  null,      value,          value,     null
  //       1  value,     value,          value,     null
  //       2  value,     null,           value,     out
  // null_policy::INCLUDE
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_nuls0({-1, 1, 2, 4}, {0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_nuls1({3, 4, 2, -1}, {1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_nuls2({6, -1, 8, -1}, {1, 0, 1, 0});

  // null_policy::EXCLUDE
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals0({3, 1, 2, 4});
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals1({6, 4, 2, -1}, {1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals2({-1, 9, 8, -1}, {0, 1, 1, 0});

  auto agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls0, std::move(agg));
  agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(1, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls1, std::move(agg));
  agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(2, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls2, std::move(agg));

  agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals0, std::move(agg));
  agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(1, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg));
  agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(2, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg));
}

TYPED_TEST(groupby_nth_element_test, exclude_nulls_negative_index)
{
  using K = int32_t;
  using V = TypeParam;
  using R = V;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 3, 1, 2, 2, 1, 3, 3, 2, 4, 4, 2},
    {true, true, true, true, true, true, true, true, false, true, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<V, int32_t> vals(
    {0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 4, 4, 2}, {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0});

  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4}, no_nulls());
  // keys    {1, 1, 1    2, 2, 2,        3, 3,       4}
  // vals    {-, 3, 6    1, 4, -, 9, -   2, 2, 8,    4,-}
  //       0  null,      value,          value,      value
  //       1  value,     value,          value,      null
  //       2  value,     null,           value,      out
  //       3  out,       value,          out,        out
  //       4  out,       null,           out,        out

  // null_policy::INCLUDE
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_nuls0({6, -1, 8, -1}, {1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_nuls1({3, 9, 2, 4});
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_nuls2({-1, -1, 2, -1}, {0, 0, 1, 0});

  // null_policy::EXCLUDE
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals0({6, 9, 8, 4});
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals1({3, 4, 2, -1}, {1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<R, int32_t> expect_vals2({-1, 1, 2, -1}, {0, 1, 1, 0});

  auto agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-1, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls0, std::move(agg));
  agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-2, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls1, std::move(agg));
  agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-3, cudf::null_policy::INCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_nuls2, std::move(agg));

  agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-1, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals0, std::move(agg));
  agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-2, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg));
  agg =
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-3, cudf::null_policy::EXCLUDE);
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg));
}

struct groupby_nth_element_string_test : public cudf::test::BaseFixture {};

TEST_F(groupby_nth_element_string_test, basic_string)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::strings_column_wrapper vals{"ABCD", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  // groupby.first()
  auto agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0);
  cudf::test::strings_column_wrapper expect_vals0{"ABCD", "1", "2"};
  test_single_agg(keys, vals, expect_keys, expect_vals0, std::move(agg));

  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(1);
  cudf::test::strings_column_wrapper expect_vals1{"3", "4", "7"};
  test_single_agg(keys, vals, expect_keys, expect_vals1, std::move(agg));

  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(2);
  cudf::test::strings_column_wrapper expect_vals2{"6", "5", "8"};
  test_single_agg(keys, vals, expect_keys, expect_vals2, std::move(agg));

  //+ve out of bounds
  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(3);
  cudf::test::strings_column_wrapper expect_vals3{{"", "9", ""}, {false, true, false}};
  test_single_agg(keys, vals, expect_keys, expect_vals3, std::move(agg));

  // groupby.last()
  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-1);
  cudf::test::strings_column_wrapper expect_vals4{"6", "9", "8"};
  test_single_agg(keys, vals, expect_keys, expect_vals4, std::move(agg));

  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-2);
  cudf::test::strings_column_wrapper expect_vals5{"3", "5", "7"};
  test_single_agg(keys, vals, expect_keys, expect_vals5, std::move(agg));

  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-3);
  cudf::test::strings_column_wrapper expect_vals6{"ABCD", "4", "2"};
  test_single_agg(keys, vals, expect_keys, expect_vals6, std::move(agg));

  //-ve out of bounds
  agg = cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(-4);
  cudf::test::strings_column_wrapper expect_vals7{{"", "1", ""}, {false, true, false}};
  test_single_agg(keys, vals, expect_keys, expect_vals7, std::move(agg));
}

TEST_F(groupby_nth_element_string_test, dictionary)
{
  using K = int32_t;
  using V = std::string;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::dictionary_column_wrapper<V> vals{"AB", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::dictionary_column_wrapper<V> expect_vals_w{"6", "5", "8"};

  auto expect_vals = cudf::dictionary::set_keys(expect_vals_w, vals.keys());

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals->view(),
                  cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(2));
}

template <typename T>
struct groupby_nth_element_lists_test : cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_nth_element_lists_test, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(groupby_nth_element_lists_test, Basics)
{
  using K = int32_t;
  using V = TypeParam;

  using lists = cudf::test::lists_column_wrapper<V, int32_t>;

  auto keys   = cudf::test::fixed_width_column_wrapper<K, int32_t>{1, 1, 2, 2, 3, 3};
  auto values = lists{{1, 2}, {3, 4}, {5, 6, 7}, lists{}, {9, 10}, {11}};

  auto expected_keys   = cudf::test::fixed_width_column_wrapper<K, int32_t>{1, 2, 3};
  auto expected_values = lists{{1, 2}, {5, 6, 7}, {9, 10}};

  test_single_agg(keys,
                  values,
                  expected_keys,
                  expected_values,
                  cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
}

TYPED_TEST(groupby_nth_element_lists_test, EmptyInput)
{
  using K = int32_t;
  using V = TypeParam;

  using lists = cudf::test::lists_column_wrapper<V, int32_t>;

  auto keys   = cudf::test::fixed_width_column_wrapper<K, int32_t>{};
  auto values = lists{};

  auto expected_keys   = cudf::test::fixed_width_column_wrapper<K, int32_t>{};
  auto expected_values = lists{};

  test_single_agg(keys,
                  values,
                  expected_keys,
                  expected_values,
                  cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(2));
}

struct groupby_nth_element_structs_test : cudf::test::BaseFixture {};

TEST_F(groupby_nth_element_structs_test, Basics)
{
  using structs = cudf::test::structs_column_wrapper;
  using ints    = cudf::test::fixed_width_column_wrapper<int>;
  using doubles = cudf::test::fixed_width_column_wrapper<double>;
  using strings = cudf::test::strings_column_wrapper;

  auto keys   = ints{0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  auto child0 = ints{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto child1 = doubles{0.1, 1.2, 2.3, 3.4, 4.51, 5.3e4, 6.3231, -0.07, 832.1, 9.999};
  auto child2 = strings{"", "a", "b", "c", "d", "e", "f", "g", "HH", "JJJ"};
  auto values = structs{{child0, child1, child2},
                        {true, false, true, false, true, true, true, true, false, true}};

  auto expected_keys = ints{0, 1, 2, 3};
  auto expected_ch0  = ints{1, 4, 7, 0};
  auto expected_ch1  = doubles{1.2, 4.51, -0.07, 0.0};
  auto expected_ch2  = strings{"a", "d", "g", ""};
  auto expected_values =
    structs{{expected_ch0, expected_ch1, expected_ch2}, {false, true, true, false}};
  test_single_agg(keys,
                  values,
                  expected_keys,
                  expected_values,
                  cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(1));

  expected_keys   = ints{0, 1, 2, 3};
  expected_ch0    = ints{0, 4, 6, 9};
  expected_ch1    = doubles{0.1, 4.51, 6.3231, 9.999};
  expected_ch2    = strings{"", "d", "f", "JJJ"};
  expected_values = structs{{expected_ch0, expected_ch1, expected_ch2}, {true, true, true, true}};
  test_single_agg(
    keys,
    values,
    expected_keys,
    expected_values,
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0, cudf::null_policy::EXCLUDE));
}

TEST_F(groupby_nth_element_structs_test, NestedStructs)
{
  using structs = cudf::test::structs_column_wrapper;
  using ints    = cudf::test::fixed_width_column_wrapper<int>;
  using doubles = cudf::test::fixed_width_column_wrapper<double>;
  using lists   = cudf::test::lists_column_wrapper<int>;

  auto keys             = ints{0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  auto child0           = ints{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto child0_of_child1 = ints{0, -1, -2, -3, -4, -5, -6, -7, -8, -9};
  auto child1_of_child1 = doubles{0.1, 1.2, 2.3, 3.4, 4.51, 5.3e4, 6.3231, -0.07, 832.1, 9.999};
  auto child1           = structs{child0_of_child1, child1_of_child1};
  auto child2           = lists{{0}, {1, 2, 3}, {}, {4}, {5, 6}, {}, {}, {7}, {8, 9}, {}};
  auto values           = structs{{child0, child1, child2},
                                  {true, false, true, false, true, true, true, true, false, true}};

  auto expected_keys       = ints{0, 1, 2, 3};
  auto expected_ch0        = ints{1, 4, 7, 0};
  auto expected_ch0_of_ch1 = ints{-1, -4, -7, 0};
  auto expected_ch1_of_ch1 = doubles{1.2, 4.51, -0.07, 0.0};
  auto expected_ch1        = structs{expected_ch0_of_ch1, expected_ch1_of_ch1};
  auto expected_ch2        = lists{{1, 2, 3}, {5, 6}, {7}, {}};
  auto expected_values =
    structs{{expected_ch0, expected_ch1, expected_ch2}, {false, true, true, false}};
  test_single_agg(keys,
                  values,
                  expected_keys,
                  expected_values,
                  cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(1));

  expected_keys       = ints{0, 1, 2, 3};
  expected_ch0        = ints{0, 4, 6, 9};
  expected_ch0_of_ch1 = ints{0, -4, -6, -9};
  expected_ch1_of_ch1 = doubles{0.1, 4.51, 6.3231, 9.999};
  expected_ch1        = structs{expected_ch0_of_ch1, expected_ch1_of_ch1};
  expected_ch2        = lists{{0}, {5, 6}, {}, {}};
  expected_values = structs{{expected_ch0, expected_ch1, expected_ch2}, {true, true, true, true}};
  test_single_agg(
    keys,
    values,
    expected_keys,
    expected_values,
    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0, cudf::null_policy::EXCLUDE));
}

TEST_F(groupby_nth_element_structs_test, EmptyInput)
{
  using structs = cudf::test::structs_column_wrapper;
  using ints    = cudf::test::fixed_width_column_wrapper<int>;
  using doubles = cudf::test::fixed_width_column_wrapper<double>;
  using strings = cudf::test::strings_column_wrapper;

  auto keys   = ints{};
  auto child0 = ints{};
  auto child1 = doubles{};
  auto child2 = strings{};
  auto values = structs{{child0, child1, child2}};

  auto expected_keys   = ints{};
  auto expected_ch0    = ints{};
  auto expected_ch1    = doubles{};
  auto expected_ch2    = strings{};
  auto expected_values = structs{{expected_ch0, expected_ch1, expected_ch2}};
  test_single_agg(keys,
                  values,
                  expected_keys,
                  expected_values,
                  cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
}
