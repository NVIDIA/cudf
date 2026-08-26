/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/hashing.hpp>
#include <cudf/utilities/error.hpp>

class SHA224HashTest : public cudf::test::BaseFixture {};

TEST_F(SHA224HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hashing::sha224(empty_table);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hashing::sha224(empty_table);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA224HashTest, MultiValue)
{
  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "0",
     "A 56 character string to test message padding algorithm.",
     "A 63 character string to test message padding algorithm, again.",
     "A 64 character string to test message padding algorithm, again!!",
     "A 128 character string to test message padding algorithm for exact multiples of 64 bytes. "
     "Let's be very sure that this works!!!!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)",
     "Multi-byte characters: é¼³⅝"});

  /*
  These outputs can be generated with shell:
  ```
  echo -n "input string" | sha224sum
  ```
  Or with Python:
  ```
  import hashlib
  print(hashlib.sha224("input string".encode()).hexdigest())
  ```
  */
  cudf::test::strings_column_wrapper const sha224_string_results1(
    {"d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f",
     "dfd5f9139a820075df69d7895015360b76d0360f3d4b77a845689614",
     "5d1ed8373987e403482cefe1662a63fa3076c0a5331d141f41654bbe",
     "0662c91000b99de7a20c89097dd62f59120398d52499497489ccff95",
     "f9ea303770699483f3e53263b32a3b3c876d1b8808ce84df4b8ca1c4",
     "5a2db111de8110fda407037b587af6bb05eec55761696b90bca05729",
     "2da6cd4bdaa0a99fd7236cd5507c52e12328e71192e83b32d2f110f9",
     "e7d0adb165079efc6c6343112f8b154aa3644ca6326f658aaa0f8e4a",
     "309cc09eaa051beea7d0b0159daca9b4e8a533cb554e8f382c82709e",
     "6c728722ae8eafd058672bd92958199ff3a5a129e8c076752f7650f8"});

  cudf::test::strings_column_wrapper const sha224_string_results2(
    {"d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f",
     "5538ae2b02d4ae0b7090dc908ca69cd11a2ffad43c7435f1dbad5e6a",
     "8e1955a473a149368dc0a931f99379b44b0bb752f206dbdf68629232",
     "8581001e08295b7884428c022378cfdd643c977aefe4512f0252dc30",
     "d5854dfe3c32996345b103a6a16c7bdfa924723d620b150737e77370",
     "040e548e5e6e7affc27b39521c303a8030c119b9ba13d9739cf2c649",
     "dd56deac5f2caa579a440ee814fc04a3afaf805d567087ac3317beb3",
     "14fb559f6309604bedd89183f585f3b433932b5b0e675848feebf8ec",
     "d219eefea538491efcb69bc5bbef4177ad991d1b6e1367b5981b8c31",
     "5d5c2eace7ee553fe5cd25c8a8916e1eda81a5a5ca36a6338118a661"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, -1, 100, -100, limits::min(), limits::max(), 42, 1, 2, 3});

  cudf::test::fixed_width_column_wrapper<bool> const bools_col({0, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  // Test string inputs against known outputs
  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha224_string_output1 = cudf::hashing::sha224(string_input1);
  auto const sha224_string_output2 = cudf::hashing::sha224(string_input2);
  EXPECT_EQ(string_input1.num_rows(), sha224_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha224_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha224_string_output1->view(), sha224_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha224_string_output2->view(), sha224_string_results2);

  // Test non-string inputs for self-consistency
  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col});
  auto const sha224_output1 = cudf::hashing::sha224(input1);
  auto const sha224_output2 = cudf::hashing::sha224(input2);
  EXPECT_EQ(input1.num_rows(), sha224_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha224_output1->view(), sha224_output2->view());
}

TEST_F(SHA224HashTest, EmptyNullEquivalence)
{
  // Test that empty strings hash the same as nulls
  cudf::test::strings_column_wrapper const strings_col1({"", ""}, {true, false});
  cudf::test::strings_column_wrapper const strings_col2({"", ""}, {false, true});

  auto const input1 = cudf::table_view({strings_col1});
  auto const input2 = cudf::table_view({strings_col2});

  auto const output1 = cudf::hashing::sha224(input1);
  auto const output2 = cudf::hashing::sha224(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(SHA224HashTest, ListsUnsupported)
{
  cudf::test::lists_column_wrapper<cudf::string_view> strings_list_col(
    {{""},
     {"", "Some inputs"},
     {"All ", "work ", "and", " no", " play ", "makes Jack", " a dull boy"},
     {R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`)", "{|}~"}});

  auto const input = cudf::table_view({strings_list_col});

  EXPECT_THROW(cudf::hashing::sha224(input), cudf::data_type_error);
}

TEST_F(SHA224HashTest, StructsUnsupported)
{
  auto child_col   = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
  auto struct_col  = cudf::test::structs_column_wrapper{{child_col}};
  auto const input = cudf::table_view({struct_col});

  EXPECT_THROW(cudf::hashing::sha224(input), cudf::data_type_error);
}

template <typename T>
class SHA224HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA224HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA224HashTestTyped, NoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::sha224(input);
  auto const output2 = cudf::hashing::sha224(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA224HashTestTyped, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::sha224(input);
  auto const output2 = cudf::hashing::sha224(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA224HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA224HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA224HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  cudf::test::fixed_width_column_wrapper<T> const col1(
    {T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  cudf::test::fixed_width_column_wrapper<T> const col2(
    {T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hashing::sha224(input1);
  auto const output2 = cudf::hashing::sha224(input2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}
