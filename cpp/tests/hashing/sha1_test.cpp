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

class SHA1HashTest : public cudf::test::BaseFixture {};

TEST_F(SHA1HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hashing::sha1(empty_table);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hashing::sha1(empty_table);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA1HashTest, MultiValue)
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
  echo -n "input string" | sha1sum
  ```
  Or with Python:
  ```
  import hashlib
  print(hashlib.sha1("input string".encode()).hexdigest())
  ```
  */
  cudf::test::strings_column_wrapper const sha1_string_results1(
    {"da39a3ee5e6b4b0d3255bfef95601890afd80709",
     "b6589fc6ab0dc82cf12099d1c2d40ab994e8410c",
     "cb73203438ab46ea54491c53e288a2703c440c4a",
     "c595ebd13a785c1c2659e010a42e2ff9987ef51f",
     "4ffaf61804c55b8c2171be548bef2e1d0baca17a",
     "a5a365142798fac9acb8d8d1c57a281d2f7d7f93",
     "595965dd18f38087186162c788485fe249242131",
     "a62ca720fbab830c8890044eacbeac216f1ca2e4",
     "11e16c52273b5669a41d17ec7c187475193f88b3",
     "0826be2f7b9340eed269c7f9f3f3662c0a3ece68"});

  cudf::test::strings_column_wrapper const sha1_string_results2(
    {"da39a3ee5e6b4b0d3255bfef95601890afd80709",
     "fb96549631c835eb239cd614cc6b5cb7d295121a",
     "e3977ee0ea7f238134ec93c79988fa84b7c5d79e",
     "f6f75b6fa3c3d8d86b44fcb2c98c9ad4b37dcdd0",
     "c7abd431a775c604edf41a62f7f215e7258dc16a",
     "408e7160bf55fae2a8cfb3ff2f6f830513a30a8a",
     "153fdf20d2bd8ae76241197314d6e0be7fe10f50",
     "8c3656f7cb37898f9296c1965000d6da13fed64e",
     "b4a848399375ec842c2cb445d98b5f80a4dce94f",
     "106a56e997aa6a149cc5091750574a25c324fa65"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, -1, 100, -100, limits::min(), limits::max(), 42, 1, 2, 3});

  cudf::test::fixed_width_column_wrapper<bool> const bools_col({0, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  // Test string inputs against known outputs
  auto const string_input1       = cudf::table_view({strings_col});
  auto const string_input2       = cudf::table_view({strings_col, strings_col});
  auto const sha1_string_output1 = cudf::hashing::sha1(string_input1);
  auto const sha1_string_output2 = cudf::hashing::sha1(string_input2);
  EXPECT_EQ(string_input1.num_rows(), sha1_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha1_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha1_string_output1->view(), sha1_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha1_string_output2->view(), sha1_string_results2);

  // Test non-string inputs for self-consistency
  auto const input1       = cudf::table_view({strings_col, ints_col, bools_col});
  auto const input2       = cudf::table_view({strings_col, ints_col, bools_col});
  auto const sha1_output1 = cudf::hashing::sha1(input1);
  auto const sha1_output2 = cudf::hashing::sha1(input2);
  EXPECT_EQ(input1.num_rows(), sha1_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha1_output1->view(), sha1_output2->view());
}

TEST_F(SHA1HashTest, EmptyNullEquivalence)
{
  // Test that empty strings hash the same as nulls
  cudf::test::strings_column_wrapper const strings_col1({"", ""}, {true, false});
  cudf::test::strings_column_wrapper const strings_col2({"", ""}, {false, true});

  auto const input1 = cudf::table_view({strings_col1});
  auto const input2 = cudf::table_view({strings_col2});

  auto const output1 = cudf::hashing::sha1(input1);
  auto const output2 = cudf::hashing::sha1(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(SHA1HashTest, ListsUnsupported)
{
  cudf::test::lists_column_wrapper<cudf::string_view> strings_list_col(
    {{""},
     {"", "Some inputs"},
     {"All ", "work ", "and", " no", " play ", "makes Jack", " a dull boy"},
     {R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`)", "{|}~"}});

  auto const input = cudf::table_view({strings_list_col});

  EXPECT_THROW(cudf::hashing::sha1(input), cudf::data_type_error);
}

TEST_F(SHA1HashTest, StructsUnsupported)
{
  auto child_col   = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
  auto struct_col  = cudf::test::structs_column_wrapper{{child_col}};
  auto const input = cudf::table_view({struct_col});

  EXPECT_THROW(cudf::hashing::sha1(input), cudf::data_type_error);
}

template <typename T>
class SHA1HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA1HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA1HashTestTyped, NoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::sha1(input);
  auto const output2 = cudf::hashing::sha1(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA1HashTestTyped, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::sha1(input);
  auto const output2 = cudf::hashing::sha1(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA1HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA1HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA1HashTestFloatTyped, TestExtremes)
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

  auto const output1 = cudf::hashing::sha1(input1);
  auto const output2 = cudf::hashing::sha1(input2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}
