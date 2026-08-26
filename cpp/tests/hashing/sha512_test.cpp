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

class SHA512HashTest : public cudf::test::BaseFixture {};

TEST_F(SHA512HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hashing::sha512(empty_table);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hashing::sha512(empty_table);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA512HashTest, MultiValue)
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
  echo -n "input string" | sha512sum
  ```
  Or with Python:
  ```
  import hashlib
  print(hashlib.sha512("input string".encode()).hexdigest())
  ```
  */
  cudf::test::strings_column_wrapper const sha512_string_results1(
    {"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877ee"
     "c2f63b931bd47417a81a538327af927da3e",
     "31bca02094eb78126a517b206a88c73cfa9ec6f704c7030d18212cace820f025f00bf0ea68dbf3f3a5436ca63b53b"
     "f7bf80ad8d5de7d8359d0b7fed9dbc3ab99",
     "1d8b355dbe0c4ad81c9815a1490f0b6a6fa710e42ca60767ffd6d845acd116defe307c9496a80c4a67653873af6ed"
     "83e2e04c2102f55f9cd402677b246832e4c",
     "8ac8ae9de5597aa630f071f81fcb94dc93b6a8f92d8f2cdd5a469764a5daf6ef387b6465ae097dcd6e0c64286260d"
     "cc3d2c789d2cf5960df648c78a765e6c27c",
     "9c436e24be60e17425a1a829642d97e7180b57485cf95db007cf5b32bbae1f2325b6874b3377e37806b15b739bffa"
     "412ea6d095b726487d70e7b50e92d56c750",
     "c0b72e552c4026f6f3797281a92a124e4c83d1a901f58c7139dad5a5160e319324e002e7c8c5e64eabdb713b6006d"
     "d06e7755244aff3861fd52f4ccb3c79b321",
     "6a25ca1f20f6e79faea2a0770075e4262beb66b40f59c22d3e8abdb6188ef8d8914faf5dbf6df76165bb61b81dfda"
     "46643f0d6366a39f7bd3d270312f9d3cf87",
     "bae9eb4b5c05a4c5f85750b70b2f0ce78e387f992f0927a017eb40bd180a13004f6252a6bbf9816f195fb7d86668c"
     "393dc0985aaf7168f48e8b905f3b9b02df2",
     "05a4ca1c523dcab32edb7d8793934a4cdf41a9062b229d711f5326e297bda83fa965118b9d7636172b43688e8e149"
     "008b3f967f1a969962b7e959af894a8a315",
     "1a15d73f16820b25f2af1c824a00a6ab18fe3eb91adaae31f441f4eca7ca11baf56d2f56e4f600781bf3637a49a4f"
     "bdbd5d7e0d8e894c51144e28eed59b3721a"});

  cudf::test::strings_column_wrapper const sha512_string_results2(
    {"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877ee"
     "c2f63b931bd47417a81a538327af927da3e",
     "8ab3361c051a97ddc3c665d29f2762f8ac4240d08995f8724b6d07d8cbedd32c28f589ccdae514f20a6c8eea6f755"
     "408dd3dd6837d66932ca2352eaeab594427",
     "338b22eb841420affff9904f903ed14c91bf8f4d1b10f25c145a31018367607a2cf562121ba7eaa2d08db3382cc82"
     "149805198c1fa3e7dc714fc2782e0f6ebd8",
     "d3045ecde16ea036d2f2ff3fa685beb46d5fcb73de71f0aee653265f18b22e4c131255e6eb5ad3be2f32914408ec6"
     "67911b49d951714decbdbfca1957be8ba10",
     "da7706221f8861ef522ab9555f57306382fb18c337536545d839e431dede4ff9f9affafb82ab5588734a8fc6631e6"
     "a0cd864634b62e24a42755c863c5d5c5848",
     "7d7498241ad7aee2845a1f79d44115511853f7d1490da871f0aaeba05a4299dc256f267c0203db1d175bee97e3969"
     "ba6f250a0d08ac037210aeced85f83a8412",
     "04dadc8fdf205fe535c8eb38f20882fc2a0e308081052d7588e74f6620aa207749039468c126db7407050def80415"
     "1d037cb188d5d4d459015032972a9e9f001",
     "aae2e742074847889a029a8d3170f9e17177d48ec0b9dabe572aa68dd3001af0c512f164ba84aa75b13950948170a"
     "0912912d16c98d2f05cb633c0d5b6a9105e",
     "77f46e99a7a51ac04b4380ebca70c0782381629f711169a3b9dad3fc9aa6221a9c0cdaa9b9ea4329773e773e2987c"
     "d1eebe0661386909684927d67819a2cf736",
     "023f99dea2a46cb4f0672645c4123697a57e2911c1889bcb5339383f81d78e0efbcca11568621b732e7ac13bef576"
     "a79f0dfb0a1db2a2ede8a14e860e3a9f1bc"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, -1, 100, -100, limits::min(), limits::max(), 42, 1, 2, 3});

  cudf::test::fixed_width_column_wrapper<bool> const bools_col({0, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  // Test string inputs against known outputs
  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha512_string_output1 = cudf::hashing::sha512(string_input1);
  auto const sha512_string_output2 = cudf::hashing::sha512(string_input2);
  EXPECT_EQ(string_input1.num_rows(), sha512_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha512_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha512_string_output1->view(), sha512_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha512_string_output2->view(), sha512_string_results2);

  // Test non-string inputs for self-consistency
  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col});
  auto const sha512_output1 = cudf::hashing::sha512(input1);
  auto const sha512_output2 = cudf::hashing::sha512(input2);
  EXPECT_EQ(input1.num_rows(), sha512_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha512_output1->view(), sha512_output2->view());
}

TEST_F(SHA512HashTest, EmptyNullEquivalence)
{
  // Test that empty strings hash the same as nulls
  cudf::test::strings_column_wrapper const strings_col1({"", ""}, {true, false});
  cudf::test::strings_column_wrapper const strings_col2({"", ""}, {false, true});

  auto const input1 = cudf::table_view({strings_col1});
  auto const input2 = cudf::table_view({strings_col2});

  auto const output1 = cudf::hashing::sha512(input1);
  auto const output2 = cudf::hashing::sha512(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(SHA512HashTest, ListsUnsupported)
{
  cudf::test::lists_column_wrapper<cudf::string_view> strings_list_col(
    {{""},
     {"", "Some inputs"},
     {"All ", "work ", "and", " no", " play ", "makes Jack", " a dull boy"},
     {R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`)", "{|}~"}});

  auto const input = cudf::table_view({strings_list_col});

  EXPECT_THROW(cudf::hashing::sha512(input), cudf::data_type_error);
}

TEST_F(SHA512HashTest, StructsUnsupported)
{
  auto child_col   = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
  auto struct_col  = cudf::test::structs_column_wrapper{{child_col}};
  auto const input = cudf::table_view({struct_col});

  EXPECT_THROW(cudf::hashing::sha512(input), cudf::data_type_error);
}

template <typename T>
class SHA512HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA512HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA512HashTestTyped, NoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::sha512(input);
  auto const output2 = cudf::hashing::sha512(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA512HashTestTyped, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::sha512(input);
  auto const output2 = cudf::hashing::sha512(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA512HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA512HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA512HashTestFloatTyped, TestExtremes)
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

  auto const output1 = cudf::hashing::sha512(input1);
  auto const output2 = cudf::hashing::sha512(input2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}
