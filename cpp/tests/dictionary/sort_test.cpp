/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

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
