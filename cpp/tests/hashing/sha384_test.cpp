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

class SHA384HashTest : public cudf::test::BaseFixture {};

TEST_F(SHA384HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hashing::sha384(empty_table);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hashing::sha384(empty_table);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA384HashTest, MultiValue)
{
  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "0",
     "A 56 character string to test message padding algorithm.",
     "A 63 character string to test message padding algorithm, again.",
     "A 64 character string to test message padding algorithm, again!!",
     "A 256 character string to test message padding algorithm for exact multiples of 64 bytes, "
     "again. I ran this through python for a quick length check to ensure this string is exactly "
     "256 chars, but feel free to count it yourself manually to be extra certain.",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)",
     "Multi-byte characters: é¼³⅝"});

  /*
  These outputs can be generated with shell:
  ```
  echo -n "input string" | sha384sum
  ```
  Or with Python:
  ```
  import hashlib
  print(hashlib.sha384("input string".encode()).hexdigest())
  ```
  */
  cudf::test::strings_column_wrapper const sha384_string_results1(
    {"38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b"
     "95b",
     "5f91550edb03f0bb8917da57f0f8818976f5da971307b7ee4886bb951c4891a1f16f840dae8f655aa5df718884ebc"
     "15b",
     "982000cce895dc439edbcb7ba5b908cb5b7e939fe913d58506a486735a914b0dfbcebb02c33c428287baa0bfc7fe0"
     "948",
     "c3ea54e4d6d97c2a84dac9ac48ed9dd1a49118be880d8466044720cfdcd23427bf556f12204bb34ede29dbf207033"
     "78c",
     "5d7a853a18138fa90feac07c896dfca65a0f1eb2ed40f1fd7be6238dd7ef429bb1aeb0236735500eb954c9b4ba923"
     "254",
     "26e1c7410c4fc28ce814e80abf8bf73868c3907319c59887cfaf6f621c3202b94552852e7ec40b55912cab635e917"
     "4c8",
     "c72bcaf3a4b01986711cd5d2614aa8f9d7fad61455613eac4561b1468f9a25dd26566c8ad1190dec7567be4f6fc1d"
     "b29",
     "281826f23bebb3f835d2f15edcb0cdb3078ae2d7dc516f3a366af172dff4db6dd5833bc1e5ee411d52c598773e939"
     "7b6",
     "3a9d1a870a5f6a4c04df1daf1808163d33852897ebc757a5b028a1214fbc758485a392159b11bc360cfadc79f9512"
     "822",
     "f6d9687e48ef1f69f7523c2a06c338e2b2e6cb251823d46bfa7f9ba65a071693919726b85f6dd77726a73c57a0e3a"
     "4a5"});

  cudf::test::strings_column_wrapper const sha384_string_results2(
    {"38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b"
     "95b",
     "34ae2cd40efabf896d8d4173e500278d10671b2d914efb5480e8349190bc7e8e1d532ad568d00a8295ea536a9b42b"
     "bc6",
     "e80c25efd8032ea94dad1509a68f9bf745ce1184b8a148714c28c7e0fae1100ab14057417394f83118eaa151e014d"
     "917",
     "69eaddc4ef2ed967fc6a86d3ed3777b2c2015df4cf8bbbf65681556f451a4a0ae805a89c2d56641b4422b5f248c56"
     "77d",
     "112a6f9c74741d490747db90f5e901a88b7a32f637c030d6d96e5f89a70a5f1ee209e018648842c0e1d32002f95fd"
     "d07",
     "a52942e39227bb9b4a3a0a22e170f3e86828a6c45b6b4e49ce41d06ebe879a6c12687b198a3addd90277f3ec1fa1c"
     "ba8",
     "dc6f24bb0eb2c96fb53c52c402f073de089f3aeae9594be0c4f4cb31b13bd48769b80aa97d83a25ece1edf0c83373"
     "f56",
     "781a33adfdcdcbb514318728c074fbb59d44002995825642e0c9bfef8a2ccf3fb637b39ff3dd265df8cd93c86e945"
     "ce9",
     "d2efb1591c4503f23c34ddb4da6bb1017d3d4d7c9f23ee6aa52e71c98d41060bc35eb22f41b6130d5c42a6e717fb3"
     "edf",
     "46e493cdd8b1e43ce2e90b6934a39e724949a1f8ea6709e09dbc68172089de864873ee7e10decdff98b44fbce2ba8"
     "146"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, -1, 100, -100, limits::min(), limits::max(), 42, 1, 2, 3});

  cudf::test::fixed_width_column_wrapper<bool> const bools_col({0, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  // Test string inputs against known outputs
  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha384_string_output1 = cudf::hashing::sha384(string_input1);
  auto const sha384_string_output2 = cudf::hashing::sha384(string_input2);
  EXPECT_EQ(string_input1.num_rows(), sha384_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha384_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha384_string_output1->view(), sha384_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha384_string_output2->view(), sha384_string_results2);

  // Test non-string inputs for self-consistency
  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col});
  auto const sha384_output1 = cudf::hashing::sha384(input1);
  auto const sha384_output2 = cudf::hashing::sha384(input2);
  EXPECT_EQ(input1.num_rows(), sha384_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha384_output1->view(), sha384_output2->view());
}

TEST_F(SHA384HashTest, EmptyNullEquivalence)
{
  // Test that empty strings hash the same as nulls
  cudf::test::strings_column_wrapper const strings_col1({"", ""}, {true, false});
  cudf::test::strings_column_wrapper const strings_col2({"", ""}, {false, true});

  auto const input1 = cudf::table_view({strings_col1});
  auto const input2 = cudf::table_view({strings_col2});

  auto const output1 = cudf::hashing::sha384(input1);
  auto const output2 = cudf::hashing::sha384(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(SHA384HashTest, ListsUnsupported)
{
  cudf::test::lists_column_wrapper<cudf::string_view> strings_list_col(
    {{""},
     {"", "Some inputs"},
     {"All ", "work ", "and", " no", " play ", "makes Jack", " a dull boy"},
     {R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`)", "{|}~"}});

  auto const input = cudf::table_view({strings_list_col});

  EXPECT_THROW(cudf::hashing::sha384(input), cudf::data_type_error);
}

TEST_F(SHA384HashTest, StructsUnsupported)
{
  auto child_col   = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
  auto struct_col  = cudf::test::structs_column_wrapper{{child_col}};
  auto const input = cudf::table_view({struct_col});

  EXPECT_THROW(cudf::hashing::sha384(input), cudf::data_type_error);
}

template <typename T>
class SHA384HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA384HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA384HashTestTyped, NoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::sha384(input);
  auto const output2 = cudf::hashing::sha384(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA384HashTestTyped, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::sha384(input);
  auto const output2 = cudf::hashing::sha384(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA384HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA384HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA384HashTestFloatTyped, TestExtremes)
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

  auto const output1 = cudf::hashing::sha384(input1);
  auto const output2 = cudf::hashing::sha384(input2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}
