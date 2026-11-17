/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>

#include <vector>

struct DictionaryDecodeTest : public cudf::test::BaseFixtureWithHarness {};

TEST_F(DictionaryDecodeTest, StringColumn)
{
  auto const stream = this->stream();
  auto const mr     = this->resources();

  std::vector<char const*> h_strings{"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), stream, mr);

  auto dictionary =
    cudf::dictionary::encode(strings, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto output =
    cudf::dictionary::decode(cudf::dictionary_column_view(dictionary->view()), stream, mr);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    strings, *output, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TEST_F(DictionaryDecodeTest, FloatColumn)
{
  auto const stream = this->stream();
  auto const mr     = this->resources();

  cudf::test::fixed_width_column_wrapper<float> input{
    {4.25, 7.125, 0.5, -11.75, 7.125, 0.5}, stream, mr};

  auto dictionary =
    cudf::dictionary::encode(input, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto output =
    cudf::dictionary::decode(cudf::dictionary_column_view(dictionary->view()), stream, mr);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    input, *output, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TEST_F(DictionaryDecodeTest, ColumnWithNull)
{
  auto const stream = this->stream();
  auto const mr     = this->resources();

  cudf::test::fixed_width_column_wrapper<int64_t> input{
    {444, 0, 333, 111, 222, 222, 222, 444, 000},
    {true, true, true, true, true, false, true, true, true},
    stream,
    mr};

  auto dictionary =
    cudf::dictionary::encode(input, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto output =
    cudf::dictionary::decode(cudf::dictionary_column_view(dictionary->view()), stream, mr);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    input, *output, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TEST_F(DictionaryDecodeTest, EmptyColumn)
{
  auto const stream = this->stream();
  auto const mr     = this->resources();

  cudf::test::fixed_width_column_wrapper<int16_t> input{};
  auto dictionary =
    cudf::dictionary::encode(input, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto output =
    cudf::dictionary::decode(cudf::dictionary_column_view(dictionary->view()), stream, mr);

  // check empty
  EXPECT_EQ(output->size(), 0);
  EXPECT_EQ(output->type().id(), cudf::type_id::EMPTY);
}
