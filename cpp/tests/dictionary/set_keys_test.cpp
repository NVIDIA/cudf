/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <vector>

struct DictionarySetKeysTest : public cudf::test::BaseFixture {};

TEST_F(DictionarySetKeysTest, StringsKeys)
{
  cudf::test::strings_column_wrapper strings{
    "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  auto dictionary = cudf::dictionary::encode(strings);

  cudf::test::strings_column_wrapper new_keys{"fff", "eee", "ccc", "aaa"};
  auto result = cudf::dictionary::set_keys(dictionary->view(), new_keys);
  // ensure the keys match exactly (no sorting and deterministic)
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::dictionary_column_view(result->view()).keys(), new_keys);

  std::vector<char const*> h_expected{
    "eee", "aaa", nullptr, nullptr, "ccc", "ccc", "ccc", "eee", "aaa"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
}

TEST_F(DictionarySetKeysTest, FloatKeys)
{
  cudf::test::fixed_width_column_wrapper<float> input{4.25, 7.125, 0.5, -11.75, 7.125, 0.5};
  auto dictionary = cudf::dictionary::encode(input);

  cudf::test::fixed_width_column_wrapper<float> new_keys{0.5, 1.0, 4.25, 7.125};
  auto result = cudf::dictionary::set_keys(dictionary->view(), new_keys);
  // ensure the keys match exactly (no sorting and deterministic)
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::dictionary_column_view(result->view()).keys(), new_keys);

  cudf::test::fixed_width_column_wrapper<float> expected{{4.25, 7.125, 0.5, 0., 7.125, 0.5},
                                                         {true, true, true, false, true, true}};
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
}

TEST_F(DictionarySetKeysTest, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{
    {444, 0, 333, 111, 222, 222, 222, 444, 0},
    {true, true, true, true, true, false, true, true, true}};
  auto dictionary = cudf::dictionary::encode(input);

  cudf::test::fixed_width_column_wrapper<int64_t> new_keys{0, 222, 333, 444};
  auto result = cudf::dictionary::set_keys(dictionary->view(), new_keys);
  // ensure the keys match exactly (no sorting and deterministic)
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::dictionary_column_view(result->view()).keys(), new_keys);

  cudf::test::fixed_width_column_wrapper<int64_t> expected{
    {444, 0, 333, 111, 222, 222, 222, 444, 0},
    {true, true, true, false, true, false, true, true, true}};
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
}

TEST_F(DictionarySetKeysTest, Errors)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{1, 2, 3};
  auto dictionary = cudf::dictionary::encode(input);

  cudf::test::fixed_width_column_wrapper<float> new_keys{1.0, 2.0, 3.0};
  EXPECT_THROW(cudf::dictionary::set_keys(dictionary->view(), new_keys), cudf::data_type_error);
  cudf::test::fixed_width_column_wrapper<int64_t> null_keys{{1, 2, 3}, {true, false, true}};
  EXPECT_THROW(cudf::dictionary::set_keys(dictionary->view(), null_keys), std::invalid_argument);
}

TEST_F(DictionarySetKeysTest, MatchDictionaries)
{
  cudf::test::dictionary_column_wrapper<int32_t> col1{55, 0, 44, 11, 22, 22, 22, 55, 0};
  cudf::test::dictionary_column_wrapper<int32_t> col2{11, 0, 33, 11, 44, 55, 66, 55, 0};

  auto input = std::vector<cudf::dictionary_column_view>(
    {cudf::dictionary_column_view(col1), cudf::dictionary_column_view(col2)});

  auto results = cudf::dictionary::match_dictionaries(input);
  auto keys1   = cudf::dictionary_column_view(results[0]->view()).keys();
  auto keys2   = cudf::dictionary_column_view(results[1]->view()).keys();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(keys1, keys2);

  auto result1   = cudf::dictionary::decode(cudf::dictionary_column_view(results[0]->view()));
  auto expected1 = cudf::dictionary::decode(cudf::dictionary_column_view(col1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result1->view(), expected1->view());

  auto result2   = cudf::dictionary::decode(cudf::dictionary_column_view(results[1]->view()));
  auto expected2 = cudf::dictionary::decode(cudf::dictionary_column_view(col2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result2->view(), expected2->view());
}

TEST_F(DictionarySetKeysTest, DuplicateKeys)
{
  auto input = cudf::test::strings_column_wrapper{
    "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  auto dictionary = cudf::dictionary::encode(input);

  auto new_keys = cudf::test::strings_column_wrapper{"fff", "eee", "ccc", "aaa", "ccc"};
  auto result   = cudf::dictionary::set_keys(dictionary->view(), new_keys);
  EXPECT_TRUE(result->null_count() > 0);  // duplicate keys should result in nulls

  auto expected = cudf::test::strings_column_wrapper(
    {"eee", "aaa", "", "", "ccc", "ccc", "ccc", "eee", "aaa"}, {1, 1, 0, 0, 1, 1, 1, 1, 1});
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
}
