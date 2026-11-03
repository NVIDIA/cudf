/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

template <typename T>
class EncodeNumericTests : public cudf::test::BaseFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(EncodeNumericTests, NumericTypesNotBool);

TYPED_TEST(EncodeNumericTests, SingleNullEncode)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input({1}, {0});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect({0});
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TYPED_TEST(EncodeNumericTests, EmptyEncode)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input({});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect({});
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TYPED_TEST(EncodeNumericTests, SimpleNoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 2, 3, 2, 3, 2, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{{0, 1, 2, 1, 2, 1, 0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_keys{{1, 2, 3}};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TYPED_TEST(EncodeNumericTests, SimpleWithNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 2, 3, 2, 3, 2, 1},
                                                          {1, 1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{{0, 1, 2, 3, 2, 1, 0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_keys{{1, 2, 3, 0}, {1, 1, 1, 0}};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TYPED_TEST(EncodeNumericTests, UnorderedWithNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{2, 1, 5, 1, 1, 3, 2},
                                                          {0, 1, 1, 1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{{4, 0, 3, 0, 4, 2, 1}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_keys{{1, 2, 3, 5, 0}, {1, 1, 1, 1, 0}};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

struct EncodeStringTest : public cudf::test::BaseFixture {};

TEST_F(EncodeStringTest, SimpleNoNulls)
{
  cudf::test::strings_column_wrapper input{"a", "b", "c", "d", "a"};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{0, 1, 2, 3, 0};
  cudf::test::strings_column_wrapper expect_keys{"a", "b", "c", "d"};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TEST_F(EncodeStringTest, SimpleWithNulls)
{
  cudf::test::strings_column_wrapper input{{"a", "", "c", "d", ""}, {1, 0, 1, 1, 0}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{0, 3, 1, 2, 3};
  cudf::test::strings_column_wrapper expect_keys{{"a", "c", "d", ""}, {1, 1, 1, 0}};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TEST_F(EncodeStringTest, UnorderedWithNulls)
{
  cudf::test::strings_column_wrapper input{{"ef", "", "c", "d", "", "a"}, {1, 0, 1, 1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect{3, 4, 1, 2, 4, 0};
  cudf::test::strings_column_wrapper expect_keys{{"a", "c", "d", "ef", ""}, {1, 1, 1, 1, 0}};
  auto const result = cudf::encode(cudf::table_view({input}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->view().column(0), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

TYPED_TEST(EncodeNumericTests, TableEncodeWithNulls)
{
  auto col_1 = cudf::test::fixed_width_column_wrapper<TypeParam>({1, 0, 2, 0, 1}, {1, 0, 1, 0, 1});
  auto col_2 = cudf::test::fixed_width_column_wrapper<TypeParam>({1, 3, 2, 0, 1}, {1, 1, 1, 0, 1});
  auto input = cudf::table_view({col_1, col_2});

  auto expect_keys_col1 =
    cudf::test::fixed_width_column_wrapper<TypeParam>({1, 2, 0, 0}, {1, 1, 0, 0});
  auto expect_keys_col2 =
    cudf::test::fixed_width_column_wrapper<TypeParam>({1, 2, 3, 0}, {1, 1, 1, 0});
  auto expect_keys = cudf::table_view({expect_keys_col1, expect_keys_col2});
  auto expect      = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 2, 1, 3, 0});

  auto const result = cudf::encode(input);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result.first->view(), expect_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second->view(), expect);
}

CUDF_TEST_PROGRAM_MAIN()
