/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

template <typename T>
struct ColumnTypeCheckTestTyped : public cudf::test::BaseFixture {};

struct ColumnTypeCheckTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ColumnTypeCheckTestTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(ColumnTypeCheckTestTyped, SameFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> lhs{1, 1}, rhs{2};
  EXPECT_TRUE(cudf::have_same_types(lhs, rhs));
}

TEST_F(ColumnTypeCheckTest, SameString)
{
  cudf::test::strings_column_wrapper lhs{{'a', 'a'}}, rhs{{'b'}};
  EXPECT_TRUE(cudf::have_same_types(lhs, rhs));

  cudf::test::strings_column_wrapper lhs2{}, rhs2{{'b'}};
  EXPECT_TRUE(cudf::have_same_types(lhs2, rhs2));

  cudf::test::strings_column_wrapper lhs3{}, rhs3{};
  EXPECT_TRUE(cudf::have_same_types(lhs3, rhs3));
}

TEST_F(ColumnTypeCheckTest, SameList)
{
  using LCW = cudf::test::lists_column_wrapper<int32_t>;

  LCW lhs{}, rhs{};
  EXPECT_TRUE(cudf::have_same_types(lhs, rhs));

  LCW lhs2{{1, 2, 3}}, rhs2{{4, 5}};
  EXPECT_TRUE(cudf::have_same_types(lhs2, rhs2));

  LCW lhs3{{LCW{1}, LCW{2, 3}}}, rhs3{{LCW{4, 5}}};
  EXPECT_TRUE(cudf::have_same_types(lhs3, rhs3));

  LCW lhs4{{LCW{1}, LCW{}, LCW{2, 3}}}, rhs4{{LCW{4, 5}, LCW{}}};
  EXPECT_TRUE(cudf::have_same_types(lhs4, rhs4));
}

TYPED_TEST(ColumnTypeCheckTestTyped, SameDictionary)
{
  using DCW = cudf::test::dictionary_column_wrapper<TypeParam>;
  DCW lhs{1, 1, 2, 3}, rhs{5, 5};
  EXPECT_TRUE(cudf::have_same_types(lhs, rhs));

  DCW lhs2{}, rhs2{};
  EXPECT_TRUE(cudf::have_same_types(lhs2, rhs2));
}

TEST_F(ColumnTypeCheckTest, SameStruct)
{
  using SCW      = cudf::test::structs_column_wrapper;
  using FCW      = cudf::test::fixed_width_column_wrapper<int32_t>;
  using StringCW = cudf::test::strings_column_wrapper;
  using LCW      = cudf::test::lists_column_wrapper<int32_t>;
  using DCW      = cudf::test::dictionary_column_wrapper<int32_t>;

  FCW lf1{1, 2, 3}, rf1{0, 1};
  StringCW lf2{"a", "bb", ""}, rf2{"cc", "d"};
  LCW lf3{LCW{1, 2}, LCW{}, LCW{4}}, rf3{LCW{1}, LCW{2}};
  DCW lf4{5, 5, 5}, rf4{9, 9};

  SCW lhs{lf1, lf2, lf3, lf4}, rhs{rf1, rf2, rf3, rf4};
  EXPECT_TRUE(cudf::have_same_types(lhs, rhs));
}

TEST_F(ColumnTypeCheckTest, DifferentBasics)
{
  cudf::test::fixed_width_column_wrapper<int32_t> lhs1{1, 1};
  cudf::test::strings_column_wrapper rhs1{"a", "bb"};

  EXPECT_FALSE(cudf::have_same_types(lhs1, rhs1));

  cudf::test::lists_column_wrapper<cudf::string_view> lhs2{{"hello"}, {"world", "!"}};
  cudf::test::strings_column_wrapper rhs2{"", "kk"};

  EXPECT_FALSE(cudf::have_same_types(lhs2, rhs2));

  cudf::test::fixed_width_column_wrapper<int32_t> lhs3{1, 1};
  cudf::test::dictionary_column_wrapper<int32_t> rhs3{2, 2};

  EXPECT_FALSE(cudf::have_same_types(lhs3, rhs3));

  cudf::test::lists_column_wrapper<int32_t> lhs4{{8, 8, 8}, {10, 10}};
  cudf::test::structs_column_wrapper rhs4{rhs2, rhs3};

  EXPECT_FALSE(cudf::have_same_types(lhs4, rhs4));
}

TEST_F(ColumnTypeCheckTest, DifferentFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<int32_t> lhs1{1, 1};
  cudf::test::fixed_width_column_wrapper<int64_t> rhs1{2};

  EXPECT_FALSE(cudf::have_same_types(lhs1, rhs1));

  cudf::test::fixed_width_column_wrapper<float> lhs2{1, 1};
  cudf::test::fixed_width_column_wrapper<double> rhs2{2};

  EXPECT_FALSE(cudf::have_same_types(lhs2, rhs2));

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms> lhs3{1, 1};
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us> rhs3{2};

  EXPECT_FALSE(cudf::have_same_types(lhs3, rhs3));

  cudf::test::fixed_width_column_wrapper<cudf::duration_D> lhs4{};
  cudf::test::fixed_width_column_wrapper<cudf::duration_us> rhs4{42};

  EXPECT_FALSE(cudf::have_same_types(lhs4, rhs4));

  // Same rep, different scale
  cudf::test::fixed_point_column_wrapper<int32_t> lhs5({10000}, numeric::scale_type{-3});
  cudf::test::fixed_point_column_wrapper<int32_t> rhs5({10000}, numeric::scale_type{0});

  EXPECT_FALSE(cudf::have_same_types(lhs5, rhs5));
  EXPECT_TRUE(cudf::column_types_equivalent(lhs5, rhs5));

  // Different rep, same scale
  cudf::test::fixed_point_column_wrapper<int32_t> lhs6({10000}, numeric::scale_type{-1});
  cudf::test::fixed_point_column_wrapper<int64_t> rhs6({4200}, numeric::scale_type{-1});

  EXPECT_FALSE(cudf::have_same_types(lhs6, rhs6));
}

TEST_F(ColumnTypeCheckTest, DifferentDictionary)
{
  cudf::test::dictionary_column_wrapper<int32_t, uint32_t> lhs1{1, 1, 1, 2, 2, 3};
  cudf::test::dictionary_column_wrapper<int64_t, uint32_t> rhs1{0, 0, 42, 42};

  EXPECT_FALSE(cudf::have_same_types(lhs1, rhs1));

  cudf::test::dictionary_column_wrapper<double, uint32_t> lhs2{3.14, 3.14, 5.00};
  cudf::test::dictionary_column_wrapper<int64_t, uint32_t> rhs2{0, 0, 42, 42};

  EXPECT_FALSE(cudf::have_same_types(lhs2, rhs2));

  cudf::test::dictionary_column_wrapper<int32_t, uint32_t> lhs3{1, 1, 1, 2, 2, 3};
  cudf::test::dictionary_column_wrapper<cudf::duration_s, uint32_t> rhs3{8, 8};

  EXPECT_FALSE(cudf::have_same_types(lhs3, rhs3));

  cudf::test::dictionary_column_wrapper<int32_t, uint32_t> lhs4{1, 1, 2, 3}, rhs4{};
  EXPECT_FALSE(cudf::have_same_types(lhs4, rhs4));
}

TEST_F(ColumnTypeCheckTest, DifferentLists)
{
  using LCW_i = cudf::test::lists_column_wrapper<int32_t>;
  using LCW_f = cudf::test::lists_column_wrapper<float>;

  // Different nested level
  LCW_i lhs1{LCW_i{1, 1, 2, 3}, LCW_i{}, LCW_i{42, 42}};
  LCW_i rhs1{LCW_i{LCW_i{8, 8, 8}, LCW_i{9, 9}}, LCW_i{LCW_i{42, 42}}};

  EXPECT_FALSE(cudf::have_same_types(lhs1, rhs1));

  // Different base column type
  LCW_i lhs2{LCW_i{1, 1, 2, 3}, LCW_i{}, LCW_i{42, 42}};
  LCW_f rhs2{LCW_f{9.0, 9.1}, LCW_f{3.14}, LCW_f{}};

  EXPECT_FALSE(cudf::have_same_types(lhs2, rhs2));
}

TEST_F(ColumnTypeCheckTest, DifferentStructs)
{
  cudf::test::fixed_width_column_wrapper<int32_t> lf1{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<int64_t> rf1{2, 2};

  cudf::test::structs_column_wrapper lhs1{lf1};
  cudf::test::structs_column_wrapper rhs1{rf1};

  EXPECT_FALSE(cudf::have_same_types(lhs1, rhs1));

  cudf::test::fixed_width_column_wrapper<int32_t> lf2{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> rf2{2, 2};

  cudf::test::strings_column_wrapper lf3{"a", "b", "c"};

  cudf::test::structs_column_wrapper lhs2{lf2, lf3};
  cudf::test::structs_column_wrapper rhs2{rf2};

  EXPECT_FALSE(cudf::have_same_types(lhs2, rhs2));
}

TYPED_TEST(ColumnTypeCheckTestTyped, AllTypesEqual)
{
  {
    // An empty table
    cudf::table_view tbl{};
    EXPECT_TRUE(cudf::all_have_same_types(tbl.begin(), tbl.end()));
  }

  {
    // A table with one column
    cudf::test::fixed_width_column_wrapper<TypeParam> col1{1, 2, 3};
    cudf::table_view tbl{{col1}};
    EXPECT_TRUE(cudf::all_have_same_types(tbl.begin(), tbl.end()));
  }

  {
    // A table with all the same types
    cudf::test::fixed_width_column_wrapper<TypeParam> col1{1, 2, 3};
    cudf::test::fixed_width_column_wrapper<TypeParam> col2{4, 5, 6};
    cudf::test::fixed_width_column_wrapper<TypeParam> col3{7, 8, 9};
    cudf::table_view tbl{{col1, col2, col3}};
    EXPECT_TRUE(cudf::all_have_same_types(tbl.begin(), tbl.end()));
  }
}

TEST_F(ColumnTypeCheckTest, AllTypesNotEqual)
{
  // A table with different types
  cudf::test::fixed_width_column_wrapper<int> col1{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> col2{3.14, 1.57, 2.71};
  cudf::table_view tbl{{col1, col2}};
  EXPECT_FALSE(cudf::all_have_same_types(tbl.begin(), tbl.end()));
}
