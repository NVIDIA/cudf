/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <vector>

struct DictionaryRemoveKeysTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryRemoveKeysTest, StringsColumn)
{
  cudf::test::strings_column_wrapper strings{
    "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  cudf::test::strings_column_wrapper del_keys{"ddd", "bbb", "fff"};

  auto const dictionary = cudf::dictionary::encode(strings);
  // remove keys
  {
    auto const result =
      cudf::dictionary::remove_keys(cudf::dictionary_column_view(dictionary->view()), del_keys);
    std::vector<char const*> h_expected{
      "eee", "aaa", nullptr, nullptr, "ccc", "ccc", "ccc", "eee", "aaa"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    auto const decoded = cudf::dictionary::decode(result->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
  // remove_unused_keys
  {
    cudf::test::fixed_width_column_wrapper<int32_t> gather_map{0, 4, 3, 1};
    auto const table_result =
      cudf::gather(cudf::table_view{{dictionary->view()}}, gather_map)->release();
    auto const result  = cudf::dictionary::remove_unused_keys(table_result.front()->view());
    auto const decoded = cudf::dictionary::decode(result->view());
    cudf::test::strings_column_wrapper expected{"eee", "ccc", "bbb", "aaa"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
}

TEST_F(DictionaryRemoveKeysTest, FloatColumn)
{
  cudf::test::fixed_width_column_wrapper<float> input{4.25, 7.125, 0.5, -11.75, 7.125, 0.5};
  cudf::test::fixed_width_column_wrapper<float> del_keys{4.25, -11.75, 5.0};

  auto const dictionary = cudf::dictionary::encode(input);

  {
    auto const result =
      cudf::dictionary::remove_keys(cudf::dictionary_column_view(dictionary->view()), del_keys);
    auto const decoded = cudf::dictionary::decode(result->view());
    cudf::test::fixed_width_column_wrapper<float> expected{{0., 7.125, 0.5, 0., 7.125, 0.5},
                                                           {false, true, true, false, true, true}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> gather_map{0, 2, 3, 1};
    auto const table_result =
      cudf::gather(cudf::table_view{{dictionary->view()}}, gather_map)->release();
    auto const result  = cudf::dictionary::remove_unused_keys(table_result.front()->view());
    auto const decoded = cudf::dictionary::decode(result->view());
    cudf::test::fixed_width_column_wrapper<float> expected{{4.25, 0.5, -11.75, 7.125}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
}

TEST_F(DictionaryRemoveKeysTest, WithNull)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{
    {444, 0, 333, 111, 222, 222, 222, 444, 0},
    {true, true, true, true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<int64_t> del_keys{0, 111, 777};

  auto const dictionary = cudf::dictionary::encode(input);
  {
    auto const result =
      cudf::dictionary::remove_keys(cudf::dictionary_column_view(dictionary->view()), del_keys);
    auto const decoded = cudf::dictionary::decode(result->view());
    cudf::test::fixed_width_column_wrapper<int64_t> expected{
      {444, 0, 333, 0, 222, 0, 222, 444, 0},
      {true, false, true, false, true, false, true, true, false}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> gather_map{0, 2, 3, 1};
    auto const table_result =
      cudf::gather(cudf::table_view{{dictionary->view()}}, gather_map)->release();
    auto const result  = cudf::dictionary::remove_unused_keys(table_result.front()->view());
    auto const decoded = cudf::dictionary::decode(result->view());
    cudf::test::fixed_width_column_wrapper<int64_t> expected{{444, 333, 111, 0}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
}

TEST_F(DictionaryRemoveKeysTest, DuplicateKeys)
{
  auto input = cudf::test::dictionary_column_wrapper<std::string>(
    {"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"});

  // set_keys with a duplicate "ccc" — position 3 is unreferenced
  auto dup_keys        = cudf::test::strings_column_wrapper{"aaa", "ccc", "eee", "ccc"};
  auto const with_dups = cudf::dictionary::set_keys(input, dup_keys);

  // remove "ccc" — both the indexed occurrence at position 1 and the unreferenced
  // duplicate at position 3 must be dropped
  auto del_keys = cudf::test::strings_column_wrapper{"ccc"};
  auto const result =
    cudf::dictionary::remove_keys(cudf::dictionary_column_view(with_dups->view()), del_keys);

  EXPECT_EQ(cudf::dictionary_column_view(result->view()).keys_size(), 2);

  auto expected = cudf::test::strings_column_wrapper(
    {"eee", "aaa", "", "", "", "", "", "eee", "aaa"}, {1, 1, 0, 0, 0, 0, 0, 1, 1});
  auto const decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
}

TEST_F(DictionaryRemoveKeysTest, RemoveUnusedAfterDuplicateSetKeys)
{
  auto input = cudf::test::dictionary_column_wrapper<std::string>(
    {"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"});

  // force duplicate keys
  auto new_keys        = cudf::test::strings_column_wrapper{"aaa", "ccc", "eee", "ccc"};
  auto const with_dups = cudf::dictionary::set_keys(input, new_keys);
  cudf::dictionary_column_view dups_view(with_dups->view());

  // keys are stored in the order given, duplicate included
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dups_view.keys(), new_keys);

  // remove_unused_keys should drop the unreferenced second "ccc" at position 3
  auto const cleaned      = cudf::dictionary::remove_unused_keys(dups_view);
  auto const cleaned_view = cudf::dictionary_column_view(cleaned->view());

  EXPECT_EQ(cleaned_view.keys_size(), 3);

  auto expected_decoded = cudf::test::strings_column_wrapper(
    {"eee", "aaa", "", "", "ccc", "ccc", "ccc", "eee", "aaa"}, {1, 1, 0, 0, 1, 1, 1, 1, 1});
  auto const decoded = cudf::dictionary::decode(cleaned_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected_decoded);
}

TEST_F(DictionaryRemoveKeysTest, Errors)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{1, 2, 3};
  auto const dictionary = cudf::dictionary::encode(input);

  cudf::test::fixed_width_column_wrapper<float> del_keys{1.0, 2.0, 3.0};
  EXPECT_THROW(cudf::dictionary::remove_keys(dictionary->view(), del_keys), cudf::data_type_error);
  cudf::test::fixed_width_column_wrapper<int64_t> null_keys{{1, 2, 3}, {true, false, true}};
  EXPECT_THROW(cudf::dictionary::remove_keys(dictionary->view(), null_keys), cudf::logic_error);
}

TEST_F(DictionaryRemoveKeysTest, RemoveDuplicateKeysStrings)
{
  // Build a dictionary with intentionally duplicate keys via set_keys.
  // keys = ["aaa", "ccc", "eee", "ccc"] (position 1 and 3 are both "ccc")
  // indices point into that key set
  auto input =
    cudf::test::dictionary_column_wrapper<std::string>({"eee", "aaa", "ccc", "ccc", "eee", "aaa"});
  auto dup_keys        = cudf::test::strings_column_wrapper{"aaa", "ccc", "eee", "ccc"};
  auto const with_dups = cudf::dictionary::set_keys(input, dup_keys);
  cudf::dictionary_column_view dv(with_dups->view());
  EXPECT_EQ(dv.keys_size(), 4);  // confirm we have duplicates

  auto const result = cudf::dictionary::remove_duplicate_keys(dv);
  cudf::dictionary_column_view rv(result->view());

  EXPECT_EQ(rv.keys_size(), 3);
  cudf::test::strings_column_wrapper expected_decoded{"eee", "aaa", "ccc", "ccc", "eee", "aaa"};
  auto const decoded = cudf::dictionary::decode(rv);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(decoded->view(), expected_decoded);
}

TEST_F(DictionaryRemoveKeysTest, RemoveDuplicateKeysIntegers)
{
  // keys = [10, 20, 10, 30] (10 appears at positions 0 and 2)
  // All indices pointing to position 2 must be remapped to position 0 after dedup
  cudf::test::fixed_width_column_wrapper<int32_t> keys_col{10, 20, 10, 30};
  cudf::test::fixed_width_column_wrapper<int32_t> indices_col{2, 0, 3, 1, 2, 0};
  auto with_dups = cudf::make_dictionary_column(
    keys_col.release(), indices_col.release(), rmm::device_buffer{}, 0);
  cudf::dictionary_column_view dv(with_dups->view());
  EXPECT_EQ(dv.keys_size(), 4);

  auto const result = cudf::dictionary::remove_duplicate_keys(dv);
  cudf::dictionary_column_view rv(result->view());

  EXPECT_EQ(rv.keys_size(), 3);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_decoded{10, 10, 30, 20, 10, 10};
  auto const decoded = cudf::dictionary::decode(rv);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(decoded->view(), expected_decoded);
}

TEST_F(DictionaryRemoveKeysTest, RemoveDuplicateKeysNoDuplicates)
{
  // When no duplicates exist the output should be equivalent to the input
  auto input        = cudf::test::dictionary_column_wrapper<std::string>({"b", "a", "c", "a", "b"});
  auto const result = cudf::dictionary::remove_duplicate_keys(cudf::dictionary_column_view(input));
  auto const decoded = cudf::dictionary::decode(cudf::dictionary_column_view(result->view()));
  cudf::test::strings_column_wrapper expected{"b", "a", "c", "a", "b"};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(decoded->view(), expected);
}

TEST_F(DictionaryRemoveKeysTest, RemoveDuplicateKeysEmpty)
{
  auto input        = cudf::test::dictionary_column_wrapper<int32_t>({});
  auto const result = cudf::dictionary::remove_duplicate_keys(cudf::dictionary_column_view(input));
  EXPECT_EQ(result->size(), 0);
  EXPECT_EQ(cudf::dictionary_column_view(result->view()).keys_size(), 0);
}

TEST_F(DictionaryRemoveKeysTest, RemoveDuplicateKeysSliced)
{
  // Build a dictionary with duplicate keys then slice it to exercise offset handling.
  // keys = ["aaa", "ccc", "eee", "ccc"] — "ccc" appears at positions 1 and 3
  auto input =
    cudf::test::dictionary_column_wrapper<std::string>({"eee", "aaa", "ccc", "ccc", "eee", "aaa"});
  auto dup_keys        = cudf::test::strings_column_wrapper{"aaa", "ccc", "eee", "ccc"};
  auto const with_dups = cudf::dictionary::set_keys(input, dup_keys);

  // Slice to rows [1, 4) → logical values {"aaa", "ccc", "ccc"}
  auto sliced       = cudf::slice(with_dups->view(), {1, 4}).front();
  auto const result = cudf::dictionary::remove_duplicate_keys(cudf::dictionary_column_view(sliced));
  cudf::dictionary_column_view rv(result->view());

  EXPECT_EQ(rv.keys_size(), 3);
  cudf::test::strings_column_wrapper expected_decoded{"aaa", "ccc", "ccc"};
  auto const decoded = cudf::dictionary::decode(rv);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(decoded->view(), expected_decoded);
}

TEST_F(DictionaryRemoveKeysTest, RemoveDuplicateKeysWithNullRows)
{
  // Null rows in the parent must be preserved; no new nulls are introduced
  auto input = cudf::test::dictionary_column_wrapper<std::string>(
    {"eee", "aaa", "ccc", "ccc", "eee"}, {true, true, false, true, true});
  auto dup_keys        = cudf::test::strings_column_wrapper{"aaa", "ccc", "eee", "ccc"};
  auto const with_dups = cudf::dictionary::set_keys(input, dup_keys);
  cudf::dictionary_column_view dv(with_dups->view());

  auto const result = cudf::dictionary::remove_duplicate_keys(dv);
  cudf::dictionary_column_view rv(result->view());
  EXPECT_EQ(rv.keys_size(), 3);

  cudf::test::strings_column_wrapper expected_decoded({"eee", "aaa", "", "ccc", "eee"},
                                                      {true, true, false, true, true});
  auto const decoded = cudf::dictionary::decode(rv);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected_decoded);
}
