/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>

#include <string>

template <typename T>
struct EmptyLikeTest : public cudf::test::BaseFixture {};

using numeric_types = cudf::test::NumericTypes;

TYPED_TEST_SUITE(EmptyLikeTest, numeric_types);

TYPED_TEST(EmptyLikeTest, ColumnNumericTests)
{
  cudf::size_type size   = 10;
  cudf::mask_state state = cudf::mask_state::ALL_VALID;
  auto input    = make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()}, size, state);
  auto expected = make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()}, 0);
  auto got      = cudf::empty_like(input->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *got);
}

struct EmptyLikeStringTest : public EmptyLikeTest<std::string> {};

void check_empty_string_columns(cudf::column_view lhs, cudf::column_view rhs)
{
  EXPECT_EQ(lhs.type(), rhs.type());
  EXPECT_EQ(lhs.size(), 0);
  EXPECT_EQ(lhs.null_count(), 0);
  EXPECT_EQ(lhs.nullable(), false);
  EXPECT_EQ(lhs.has_nulls(), false);
  // An empty column is not required to have children
}

TEST_F(EmptyLikeStringTest, ColumnStringTest)
{
  std::vector<char const*> h_strings{"the quick brown fox jumps over the lazy dog",
                                     "thé result does not include the value in the sum in",
                                     "",
                                     nullptr,
                                     "absent stop words"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));

  auto got = cudf::empty_like(strings);
  check_empty_string_columns(got->view(), strings);
}

template <typename T>
struct EmptyLikeScalarTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(EmptyLikeScalarTest, cudf::test::FixedWidthTypes);

TYPED_TEST(EmptyLikeScalarTest, FixedWidth)
{
  // make a column
  auto input = make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<TypeParam>()}, 1, rmm::device_buffer{}, 0);
  // get a scalar out of it
  std::unique_ptr<cudf::scalar> sc = cudf::get_element(*input, 0);

  // empty_like(column) -> column
  auto expected = cudf::empty_like(*input);
  // empty_like(scalar) -> column
  auto result = cudf::empty_like(*sc);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *result);
}

struct EmptyLikeScalarStringTest : public EmptyLikeScalarTest<std::string> {};

TEST_F(EmptyLikeScalarStringTest, String)
{
  // make a column
  cudf::test::strings_column_wrapper input{"abc"};

  // get a scalar out of it
  std::unique_ptr<cudf::scalar> sc = cudf::get_element(input, 0);

  // empty_like(column) -> column
  auto expected = cudf::empty_like(input);
  // empty_like(scalar) -> column
  auto result = cudf::empty_like(*sc);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *result);
}

struct EmptyLikeScalarListTest : public EmptyLikeScalarTest<cudf::list_view> {};

TEST_F(EmptyLikeScalarListTest, List)
{
  // make a column
  cudf::test::lists_column_wrapper<cudf::string_view> input{{{"abc", "def"}, {"h", "ijk"}},
                                                            {{"123", "456"}, {"78"}}};
  // get a scalar out of it
  std::unique_ptr<cudf::scalar> sc = cudf::get_element(input, 0);

  // empty_like(column) -> column
  auto expected = cudf::empty_like(input);
  // empty_like(scalar) -> column
  auto result = cudf::empty_like(*sc);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *result);
}

struct EmptyLikeScalarStructTest : public EmptyLikeScalarTest<cudf::struct_view> {};

TEST_F(EmptyLikeScalarStructTest, Struct)
{
  cudf::test::lists_column_wrapper<cudf::string_view> col0{{{"abc", "def"}, {"h", "ijk"}}};
  cudf::test::strings_column_wrapper col1{"abc"};
  cudf::test::fixed_width_column_wrapper<float> col2{1.0f};
  // scalar. TODO:  make cudf::get_element() work for struct scalars
  cudf::table_view tbl({col0, col1, col2});
  cudf::struct_scalar sc(tbl);
  // column
  cudf::test::structs_column_wrapper input({col0, col1, col2});

  // empty_like(column) -> column
  auto expected = cudf::empty_like(input);
  // empty_like(scalar) -> column
  auto result = cudf::empty_like(sc);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *result);
}

std::unique_ptr<cudf::table> create_table(cudf::size_type size, cudf::mask_state state)
{
  auto num_column_1 = make_numeric_column(cudf::data_type{cudf::type_id::INT64}, size, state);
  auto num_column_2 = make_numeric_column(cudf::data_type{cudf::type_id::INT32}, size, state);
  auto num_column_3 = make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64}, size, state);
  auto num_column_4 = make_numeric_column(cudf::data_type{cudf::type_id::FLOAT32}, size, state);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(num_column_1));
  columns.push_back(std::move(num_column_2));
  columns.push_back(std::move(num_column_3));
  columns.push_back(std::move(num_column_4));

  return std::make_unique<cudf::table>(std::move(columns));
}

struct EmptyLikeTableTest : public cudf::test::BaseFixture {};

TEST_F(EmptyLikeTableTest, TableTest)
{
  cudf::mask_state state = cudf::mask_state::ALL_VALID;
  cudf::size_type size   = 10;
  auto input             = create_table(size, state);
  auto expected          = create_table(0, cudf::mask_state::ALL_VALID);
  auto got               = cudf::empty_like(input->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(got->view(), expected->view());
}

template <typename T>
struct AllocateLikeTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(AllocateLikeTest, numeric_types);

TYPED_TEST(AllocateLikeTest, ColumnNumericTestSameSize)
{
  // For same size as input
  cudf::size_type size = 10;

  auto input = make_numeric_column(
    cudf::data_type{cudf::type_to_id<TypeParam>()}, size, cudf::mask_state::UNALLOCATED);
  auto got = cudf::allocate_like(input->view());
  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(*input, *got);

  input = make_numeric_column(
    cudf::data_type{cudf::type_to_id<TypeParam>()}, size, cudf::mask_state::ALL_VALID);
  got = cudf::allocate_like(input->view());
  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(*input, *got);
}

TYPED_TEST(AllocateLikeTest, ColumnNumericTestSpecifiedSize)
{
  // For different size as input
  cudf::size_type size           = 10;
  cudf::size_type specified_size = 5;

  auto state = cudf::mask_state::UNALLOCATED;
  auto input = make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()}, size, state);
  auto expected =
    make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()}, specified_size, state);
  auto got = cudf::allocate_like(input->view(), specified_size);
  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(*expected, *got);

  input = make_numeric_column(
    cudf::data_type{cudf::type_to_id<TypeParam>()}, size, cudf::mask_state::ALL_VALID);
  got = cudf::allocate_like(input->view(), specified_size);
  // Can't use CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL because the sizes of
  // the two columns are different.
  EXPECT_EQ(input->type(), got->type());
  EXPECT_EQ(specified_size, got->size());
  EXPECT_EQ(0, got->null_count());
  EXPECT_EQ(input->nullable(), got->nullable());
  EXPECT_EQ(input->num_children(), got->num_children());
}

CUDF_TEST_PROGRAM_MAIN()
