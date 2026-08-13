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

#include <vector>

struct DictionaryGatherTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryGatherTest, Gather)
{
  cudf::test::strings_column_wrapper strings{
    "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  auto dictionary = cudf::dictionary::encode(strings);
  cudf::dictionary_column_view view(dictionary->view());

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map{0, 4, 3, 1};
  auto table_result = cudf::gather(cudf::table_view{{view.parent()}}, gather_map)->release();
  auto result       = cudf::dictionary_column_view(table_result.front()->view());

  cudf::test::strings_column_wrapper expected{"eee", "ccc", "bbb", "aaa"};
  auto decoded = cudf::dictionary::decode(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
}

TEST_F(DictionaryGatherTest, GatherWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int64_t> data{{1, 5, 5, 3, 7, 1},
                                                       {false, true, false, true, true, true}};

  auto dictionary = cudf::dictionary::encode(data);
  cudf::dictionary_column_view view(dictionary->view());

  cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{4, 1, 2, 4}};
  auto table_result = cudf::gather(cudf::table_view{{dictionary->view()}}, gather_map);
  auto result       = cudf::dictionary_column_view(table_result->view().column(0));

  cudf::test::fixed_width_column_wrapper<int64_t> expected{{7, 5, 5, 7}, {true, true, false, true}};
  auto result_decoded = cudf::dictionary::decode(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_decoded->view());
}

TEST_F(DictionaryGatherTest, DuplicateKeys)
{
  auto keys   = cudf::test::strings_column_wrapper({"pear", "apple", "fruit", "pear"});
  auto values = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 1, 2, 3, 0});

  auto dictionary =
    cudf::make_dictionary_column(keys.release(), values.release(), rmm::device_buffer{}, 0);
  auto view = cudf::dictionary_column_view(dictionary->view());

  // decoded values are: {apple, fruit, pear, apple, fruit, pear, pear}
  // where row 6 is the first 'pear' key and rows 2 and 5 are the second 'pear' key
  {
    // gather_map avoids any 'pear' rows
    auto gather_map   = cudf::test::fixed_width_column_wrapper<int32_t>({0, 4, 3, 1});
    auto table_result = cudf::gather(cudf::table_view{{view.parent()}}, gather_map)->release();
    auto result       = cudf::dictionary_column_view(table_result.front()->view());

    auto expected = cudf::test::strings_column_wrapper({"apple", "fruit", "apple", "fruit"});
    auto decoded  = cudf::dictionary::decode(result);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
  }

  {
    // gather_map includes the first 'pear' row (index 6)
    auto gather_map   = cudf::test::fixed_width_column_wrapper<int32_t>({0, 6, 3, 1});
    auto table_result = cudf::gather(cudf::table_view{{view.parent()}}, gather_map)->release();
    auto result       = cudf::dictionary_column_view(table_result.front()->view());

    auto expected = cudf::test::strings_column_wrapper({"apple", "pear", "apple", "fruit"});
    auto decoded  = cudf::dictionary::decode(result);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
  }

  {
    // gather_map includes the second 'pear' row (index 2)
    auto gather_map   = cudf::test::fixed_width_column_wrapper<int32_t>({0, 2, 4, 3});
    auto table_result = cudf::gather(cudf::table_view{{view.parent()}}, gather_map)->release();
    auto result       = cudf::dictionary_column_view(table_result.front()->view());

    auto expected = cudf::test::strings_column_wrapper({"apple", "pear", "fruit", "apple"});
    auto decoded  = cudf::dictionary::decode(result);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
  }

  {
    // gather_map includes both the first and second 'pear' rows (indices 6 and 2)
    auto gather_map   = cudf::test::fixed_width_column_wrapper<int32_t>({6, 0, 2, 1});
    auto table_result = cudf::gather(cudf::table_view{{view.parent()}}, gather_map)->release();
    auto result       = cudf::dictionary_column_view(table_result.front()->view());

    auto expected = cudf::test::strings_column_wrapper({"pear", "apple", "pear", "fruit"});
    auto decoded  = cudf::dictionary::decode(result);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
  }
}
