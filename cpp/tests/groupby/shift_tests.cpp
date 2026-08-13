/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar_factories.hpp>

template <typename T>
struct groupby_shift_fixed_width_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_shift_fixed_width_test, cudf::test::FixedWidthTypes);

template <typename K, typename V>
void test_groupby_shift_fixed_width_single(
  cudf::test::fixed_width_column_wrapper<K> const& key,
  cudf::test::fixed_width_column_wrapper<V> const& value,
  cudf::size_type offset,
  cudf::scalar const& fill_value,
  cudf::test::fixed_width_column_wrapper<V> const& expected)
{
  cudf::groupby::groupby gb_obj(cudf::table_view({key}));
  std::vector<cudf::size_type> offsets{offset};
  auto got = gb_obj.shift(cudf::table_view{{value}}, offsets, {fill_value});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL((*got.second).view().column(0), expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithoutNull_NullScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> expected({-1, -1, 3, 5, -1, -1, 4},
                                                     {0, 0, 1, 1, 0, 0, 1});
  cudf::size_type offset = 2;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithNull_NullScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> expected({-1, -1, -1, -1, -1, -1, -1},
                                                     {0, 0, 0, 0, 0, 0, 0});
  cudf::size_type offset = 2;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithoutNull_ValidScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5});
  cudf::test::fixed_width_column_wrapper<V> expected({42, 42, 42, 3, 5, 8, 9, 42, 42, 42, 4, 6, 7});
  cudf::size_type offset = 3;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ForwardShiftWithNull_ValidScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5},
                                                {1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1});
  cudf::test::fixed_width_column_wrapper<V> expected(
    {42, 42, 42, 3, 5, -1, -1, 42, 42, 42, -1, -1, 7}, {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1});
  cudf::size_type offset = 3;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithoutNull_NullScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> expected({5, 8, 9, -1, 6, 7, -1},
                                                     {1, 1, 1, 0, 1, 1, 0});
  cudf::size_type offset = -1;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithNull_NullScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<V> expected({-1, 8, 9, -1, 6, 7, -1},
                                                     {0, 1, 1, 0, 1, 1, 0});
  cudf::size_type offset = -1;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithoutNull_ValidScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> expected(
    {3, 5, 42, 42, 42, 42, 42, 4, 42, 42, 42, 42, 42});
  cudf::size_type offset = -5;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, BackwardShiftWithNull_ValidScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val({3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5},
                                                {1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1});
  cudf::test::fixed_width_column_wrapper<V> expected({5, -1, -1, -1, 3, 5, 42, -1, 7, 0, 2, -1, 42},
                                                     {1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1});
  cudf::size_type offset = -1;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ZeroShiftNullScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> expected({3, 5, 8, 9, 4, 6, 7});
  cudf::size_type offset = 0;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, ZeroShiftValidScalar)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> expected({3, 5, 8, 9, 1, 3, 5, 4, 6, 7, 0, 2, 4});
  cudf::size_type offset = 0;
  auto slr =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, VeryLargeForwardOffset)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> expected(
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  cudf::size_type offset = 1024;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, *slr, expected);
}

TYPED_TEST(groupby_shift_fixed_width_test, VeryLargeBackwardOffset)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
  cudf::test::fixed_width_column_wrapper<V> val{3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> expected(
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  cudf::size_type offset = -1024;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_fixed_width_single<K, V>(key, val, offset, *slr, expected);
}

struct groupby_shift_string_test : public cudf::test::BaseFixture {};

template <typename K>
void test_groupby_shift_string_single(cudf::test::fixed_width_column_wrapper<K> const& key,
                                      cudf::test::strings_column_wrapper const& value,
                                      cudf::size_type offset,
                                      cudf::scalar const& fill_value,
                                      cudf::test::strings_column_wrapper const& expected)
{
  cudf::groupby::groupby gb_obj(cudf::table_view({key}));
  std::vector<cudf::size_type> offsets{offset};
  auto got = gb_obj.shift(cudf::table_view{{value}}, offsets, {fill_value});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL((*got.second).view().column(0), expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithoutNull_NullScalar)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"", "a", "cc", "f", "", "bb", "d"},
                                              {false, true, true, true, false, true, true});
  cudf::size_type offset = 1;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithNull_NullScalar)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"},
                                         {true, false, true, true, false, false, false});
  cudf::test::strings_column_wrapper expected({"", "", "a", "cc", "", "", ""},
                                              {false, false, true, true, false, false, false});
  cudf::size_type offset = 2;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithoutNull_ValidScalar)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"42", "42", "a", "cc", "42", "42", "bb"});

  cudf::size_type offset = 2;
  auto slr               = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ForwardShiftWithNull_ValidScalar)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"},
                                         {true, true, false, false, true, false, true});
  cudf::test::strings_column_wrapper expected({"42", "a", "", "", "42", "bb", ""},
                                              {true, true, false, false, true, true, false});

  cudf::size_type offset = 1;
  auto slr               = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithoutNull_NullScalar)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"gg", "", "", "", "", "", ""},
                                              {true, false, false, false, false, false, false});

  cudf::size_type offset = -3;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithNull_NullScalar)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"},
                                         {true, false, true, true, false, false, false});
  cudf::test::strings_column_wrapper expected({"cc", "", "", "", "d", "", ""},
                                              {true, false, false, false, true, false, false});

  cudf::size_type offset = -1;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithoutNull_ValidScalar)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"42", "42", "42", "42", "42", "42", "42"});

  cudf::size_type offset = -4;
  auto slr               = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, BackwardShiftWithNull_ValidScalar)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val({"a", "bb", "cc", "d", "eee", "f", "gg"},
                                         {true, true, false, false, true, false, true});
  cudf::test::strings_column_wrapper expected({"", "gg", "42", "42", "eee", "42", "42"},
                                              {false, true, true, true, true, true, true});

  cudf::size_type offset = -2;
  auto slr               = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ZeroShiftNullScalar)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"a", "cc", "f", "gg", "bb", "d", "eee"});

  cudf::size_type offset = 0;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type());

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, ZeroShiftValidScalar)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"a", "cc", "f", "gg", "bb", "d", "eee"});

  cudf::size_type offset = 0;
  auto slr               = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, VeryLargeForwardOffset)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"42", "42", "42", "42", "42", "42", "42"});

  cudf::size_type offset = 1024;
  auto slr               = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

TEST_F(groupby_shift_string_test, VeryLargeBackwardOffset)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper val{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::strings_column_wrapper expected({"42", "42", "42", "42", "42", "42", "42"});

  cudf::size_type offset = -1024;
  auto slr               = cudf::make_string_scalar("42");

  test_groupby_shift_string_single(key, val, offset, *slr, expected);
}

template <typename T>
struct groupby_shift_mixed_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_shift_mixed_test, cudf::test::FixedWidthTypes);

template <typename K>
void test_groupby_shift_multi(cudf::test::fixed_width_column_wrapper<K> const& key,
                              cudf::table_view const& value,
                              std::vector<cudf::size_type> offsets,
                              std::vector<std::reference_wrapper<cudf::scalar const>> fill_values,
                              cudf::table_view const& expected)
{
  cudf::groupby::groupby gb_obj(cudf::table_view({key}));
  auto got = gb_obj.shift(value, offsets, fill_values);
  CUDF_TEST_EXPECT_TABLES_EQUAL((*got.second).view(), expected);
}

TYPED_TEST(groupby_shift_mixed_test, NoFill)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper v1{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::fixed_width_column_wrapper<TypeParam> v2{1, 2, 3, 4, 5, 6, 7};
  cudf::table_view value{{v1, v2}};

  cudf::test::strings_column_wrapper e1({"", "", "a", "cc", "", "", "bb"},
                                        {false, false, true, true, false, false, true});
  cudf::test::fixed_width_column_wrapper<TypeParam> e2({-1, 1, 3, 6, -1, 2, 4},
                                                       {0, 1, 1, 1, 0, 1, 1});
  cudf::table_view expected{{e1, e2}};

  std::vector<cudf::size_type> offset{2, 1};
  auto slr1 = cudf::make_default_constructed_scalar(cudf::column_view(v1).type());
  auto slr2 = cudf::make_default_constructed_scalar(cudf::column_view(v2).type());
  std::vector<std::reference_wrapper<cudf::scalar const>> fill_values{*slr1, *slr2};

  test_groupby_shift_multi(key, value, offset, fill_values, expected);
}

TYPED_TEST(groupby_shift_mixed_test, Fill)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::strings_column_wrapper v1{"a", "bb", "cc", "d", "eee", "f", "gg"};
  cudf::test::fixed_width_column_wrapper<TypeParam> v2{1, 2, 3, 4, 5, 6, 7};
  cudf::table_view value{{v1, v2}};

  cudf::test::strings_column_wrapper e1({"cc", "f", "gg", "42", "d", "eee", "42"});
  cudf::test::fixed_width_column_wrapper<TypeParam> e2({6, 7, 42, 42, 5, 42, 42});
  cudf::table_view expected{{e1, e2}};

  std::vector<cudf::size_type> offset{-1, -2};

  auto slr1 = cudf::make_string_scalar("42");
  auto slr2 =
    cudf::scalar_type_t<TypeParam>(cudf::test::make_type_param_scalar<TypeParam>(42), true);
  std::vector<std::reference_wrapper<cudf::scalar const>> fill_values{*slr1, slr2};

  test_groupby_shift_multi(key, value, offset, fill_values, expected);
}

struct groupby_shift_fixed_point_type_test : public cudf::test::BaseFixture {};

TEST_F(groupby_shift_fixed_point_type_test, Matching)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{2, 3, 4, 4, 3, 2, 2, 4};
  cudf::test::fixed_point_column_wrapper<int32_t> v1{{10, 10, 40, 40, 20, 20, 30, 40},
                                                     numeric::scale_type{-1}};
  cudf::test::fixed_point_column_wrapper<int64_t> v2{{5, 5, 8, 8, 6, 7, 9, 7},
                                                     numeric::scale_type{3}};
  cudf::table_view value{{v1, v2}};

  std::vector<cudf::size_type> offset{-3, 1};
  auto slr1 = cudf::make_fixed_point_scalar<numeric::decimal32>(-42, numeric::scale_type{-1});
  auto slr2 = cudf::make_fixed_point_scalar<numeric::decimal64>(42, numeric::scale_type{3});
  std::vector<std::reference_wrapper<cudf::scalar const>> fill_values{*slr1, *slr2};

  cudf::test::fixed_point_column_wrapper<int32_t> e1{{-42, -42, -42, -42, -42, -42, -42, -42},
                                                     numeric::scale_type{-1}};
  cudf::test::fixed_point_column_wrapper<int64_t> e2{{42, 5, 7, 42, 5, 42, 8, 8},
                                                     numeric::scale_type{3}};
  cudf::table_view expected{{e1, e2}};

  test_groupby_shift_multi(key, value, offset, fill_values, expected);
}

TEST_F(groupby_shift_fixed_point_type_test, MismatchScaleType)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{2, 3, 4, 4, 3, 2, 2, 4};
  cudf::test::fixed_point_column_wrapper<int32_t> v1{{10, 10, 40, 40, 20, 20, 30, 40},
                                                     numeric::scale_type{-1}};

  std::vector<cudf::size_type> offset{-3};
  auto slr1 = cudf::make_fixed_point_scalar<numeric::decimal32>(-42, numeric::scale_type{-4});

  cudf::test::fixed_point_column_wrapper<int32_t> stub{{-42, -42, -42, -42, -42, -42, -42, -42},
                                                       numeric::scale_type{-1}};

  EXPECT_THROW(test_groupby_shift_multi(
                 key, cudf::table_view{{v1}}, offset, {*slr1}, cudf::table_view{{stub}}),
               cudf::data_type_error);
}

TEST_F(groupby_shift_fixed_point_type_test, MismatchRepType)
{
  using K = int32_t;
  cudf::test::fixed_width_column_wrapper<K> key{2, 3, 4, 4, 3, 2, 2, 4};
  cudf::test::fixed_point_column_wrapper<int64_t> v1{{10, 10, 40, 40, 20, 20, 30, 40},
                                                     numeric::scale_type{-1}};

  std::vector<cudf::size_type> offset{-3};
  auto slr1 = cudf::make_fixed_point_scalar<numeric::decimal32>(-42, numeric::scale_type{-1});

  cudf::test::fixed_point_column_wrapper<int32_t> stub{{-42, -42, -42, -42, -42, -42, -42, -42},
                                                       numeric::scale_type{-1}};

  EXPECT_THROW(test_groupby_shift_multi(
                 key, cudf::table_view{{v1}}, offset, {*slr1}, cudf::table_view{{stub}}),
               cudf::data_type_error);
}
