/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/utilities/error.hpp>

struct DictionaryAddKeysTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryAddKeysTest, Strings)
{
  auto input = cudf::test::strings_column_wrapper(
    {"fff", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "fff", "aaa"});
  cudf::test::strings_column_wrapper new_keys({"ddd", "bbb", "eee"});

  auto dictionary = cudf::dictionary::encode(input);
  auto result =
    cudf::dictionary::add_keys(cudf::dictionary_column_view(dictionary->view()), new_keys);

  cudf::dictionary_column_view view(result->view());
  auto decode = cudf::dictionary::decode(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, decode->view());
}

TEST_F(DictionaryAddKeysTest, Floats)
{
  cudf::test::fixed_width_column_wrapper<float> input{4.25, 7.125, 0.5, -11.75, 7.125, 0.5};
  cudf::test::fixed_width_column_wrapper<float> new_keys{4.25, -11.75, 5.0};

  auto dictionary = cudf::dictionary::encode(input);
  auto result =
    cudf::dictionary::add_keys(cudf::dictionary_column_view(dictionary->view()), new_keys);
  cudf::dictionary_column_view view(result->view());
  auto decode = cudf::dictionary::decode(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, decode->view());
}

TEST_F(DictionaryAddKeysTest, WithNull)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{{555, 0, 333, 111, 222, 222, 222, 555, 0},
                                                        {1, 1, 1, 0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> new_keys{0, 111, 444, 777};

  auto dictionary = cudf::dictionary::encode(input);
  auto result =
    cudf::dictionary::add_keys(cudf::dictionary_column_view(dictionary->view()), new_keys);
  cudf::dictionary_column_view view(result->view());
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), input);  // new keys should not change anything
}

TEST_F(DictionaryAddKeysTest, DuplicateKeys)
{
  auto input = cudf::test::dictionary_column_wrapper<std::string>(
    {"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"});

  auto const dup_keys  = cudf::test::strings_column_wrapper{"aaa", "ccc", "eee", "ccc"};
  auto const with_dups = cudf::dictionary::set_keys(input, dup_keys);  // force duplicate keys

  // add_keys with a key already present
  auto const added_keys  = cudf::test::strings_column_wrapper{"ccc", "fff"};
  auto const result      = cudf::dictionary::add_keys(with_dups->view(), added_keys);
  auto const result_view = cudf::dictionary_column_view(result->view());

  // old 4 keys + "fff" only — "ccc" must not have been added a third time
  EXPECT_EQ(result_view.keys_size(), 5);

  // indices are unchanged so decode should match what set_keys produced
  auto expected_decoded = cudf::test::strings_column_wrapper(
    {"eee", "aaa", "", "", "ccc", "ccc", "ccc", "eee", "aaa"}, {1, 1, 0, 0, 1, 1, 1, 1, 1});
  auto const decoded = cudf::dictionary::decode(result_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected_decoded);
}

TEST_F(DictionaryAddKeysTest, Errors)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{1, 2, 3};
  auto dictionary = cudf::dictionary::encode(input);

  cudf::test::fixed_width_column_wrapper<float> new_keys{1.0, 2.0, 3.0};
  EXPECT_THROW(cudf::dictionary::add_keys(dictionary->view(), new_keys), cudf::data_type_error);
  cudf::test::fixed_width_column_wrapper<int64_t> null_keys{{1, 2, 3}, {1, 0, 1}};
  EXPECT_THROW(cudf::dictionary::add_keys(dictionary->view(), null_keys), std::invalid_argument);
}

CUDF_TEST_PROGRAM_MAIN()
