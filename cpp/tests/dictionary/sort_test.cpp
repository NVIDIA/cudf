/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/sorting.hpp>

#include <vector>

struct DictionarySortTest : public cudf::test::BaseFixture {};

TEST_F(DictionarySortTest, SortStrings)
{
  auto h_input =
    std::vector<std::string>({"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"});
  auto d_input = cudf::test::strings_column_wrapper(h_input.begin(), h_input.end());

  auto dict_input = cudf::dictionary::encode(d_input);

  auto result = cudf::sort(cudf::table_view{{dict_input->view()}},
                           std::vector<cudf::order>{cudf::order::ASCENDING})
                  ->release();

  std::sort(h_input.begin(), h_input.end());
  auto result_decoded = cudf::dictionary::decode(result.front()->view());
  auto expected       = cudf::test::strings_column_wrapper(h_input.begin(), h_input.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_decoded->view());
}

TEST_F(DictionarySortTest, SortFloat)
{
  auto h_input = std::vector<double>({1.25, -5.75, 8.125, 1e9, 9.7});
  auto d_input = cudf::test::fixed_width_column_wrapper<double>(h_input.begin(), h_input.end());

  auto dict_input = cudf::dictionary::encode(d_input);

  auto result = cudf::sort(cudf::table_view{{dict_input->view()}},
                           std::vector<cudf::order>{cudf::order::ASCENDING})
                  ->release();

  std::sort(h_input.begin(), h_input.end());
  auto result_decoded = cudf::dictionary::decode(result.front()->view());
  auto expected = cudf::test::fixed_width_column_wrapper<double>(h_input.begin(), h_input.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_decoded->view());
}

TEST_F(DictionarySortTest, DuplicateKeys)
{
  auto input =
    cudf::test::dictionary_column_wrapper<std::string>({"eee", "aaa", "ccc", "ccc", "eee", "aaa"});

  // force duplicate keys
  auto dup_keys  = cudf::test::strings_column_wrapper{"aaa", "ccc", "eee", "ccc"};
  auto with_dups = cudf::dictionary::set_keys(input, dup_keys);

  auto result = cudf::sort(cudf::table_view{{with_dups->view()}},
                           std::vector<cudf::order>{cudf::order::ASCENDING})
                  ->release();

  auto result_decoded = cudf::dictionary::decode(result.front()->view());
  auto expected = cudf::test::strings_column_wrapper({"aaa", "aaa", "ccc", "ccc", "eee", "eee"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result_decoded->view());
}

TEST_F(DictionarySortTest, AllNullsNoKeys)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 0}, {false, false, false});
  auto dict_input = cudf::dictionary::encode(input);
  EXPECT_EQ(cudf::dictionary_column_view(dict_input->view()).keys().size(), 0);

  auto result   = cudf::sorted_order(cudf::table_view{{dict_input->view()}});
  auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 1, 2});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

  auto sorted         = cudf::sort(cudf::table_view{{dict_input->view()}})->release();
  auto result_decoded = cudf::dictionary::decode(sorted.front()->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result_decoded->view());
}

TEST_F(DictionarySortTest, Sliced)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
    {50, 30, 0, 10, 40, 0, 20, 60}, {true, true, false, true, true, false, true, true});
  auto dict_input = cudf::dictionary::encode(input);

  // compare against the sorted order of the equivalent non-dictionary column
  for (auto const& bounds :
       std::vector<std::vector<cudf::size_type>>{{1, 5}, {2, 7}, {3, 5}, {3, 8}, {4, 4}, {0, 8}}) {
    auto const sliced      = cudf::slice(input, bounds).front();
    auto const dict_sliced = cudf::slice(dict_input->view(), bounds).front();
    for (auto const null_order : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
      for (auto const column_order : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
        auto const expected =
          cudf::stable_sorted_order(cudf::table_view{{sliced}}, {column_order}, {null_order});
        auto const result =
          cudf::stable_sorted_order(cudf::table_view{{dict_sliced}}, {column_order}, {null_order});
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view());
      }
    }
  }
}
