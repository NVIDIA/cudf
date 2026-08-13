/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>

#include <vector>

struct DictionaryScatterTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryScatterTest, Scatter)
{
  cudf::test::strings_column_wrapper strings_source{"xxx", "bbb", "aaa", "ccc"};
  auto source = cudf::dictionary::encode(strings_source);
  cudf::test::strings_column_wrapper strings_target{
    "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  auto target = cudf::dictionary::encode(strings_target);

  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map{0, 2, 3, 7};
  auto table_result =
    cudf::scatter(
      cudf::table_view{{source->view()}}, scatter_map, cudf::table_view{{target->view()}})
      ->release();
  auto decoded =
    cudf::dictionary::decode(cudf::dictionary_column_view(table_result.front()->view()));
  cudf::test::strings_column_wrapper expected{
    "xxx", "aaa", "bbb", "aaa", "ccc", "ccc", "ccc", "ccc", "aaa"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());

  // empty map test
  cudf::test::fixed_width_column_wrapper<int32_t> empty_map{};
  table_result =
    cudf::scatter(cudf::table_view{{source->view()}}, empty_map, cudf::table_view{{target->view()}})
      ->release();
  decoded = cudf::dictionary::decode(cudf::dictionary_column_view(table_result.front()->view()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strings_target, decoded->view());

  // empty target test
  cudf::test::strings_column_wrapper empty_target;
  auto empty_dictionary = cudf::dictionary::encode(empty_target);
  table_result          = cudf::scatter(cudf::table_view{{empty_dictionary->view()}},
                               empty_map,
                               cudf::table_view{{empty_dictionary->view()}})
                   ->release();
  decoded = cudf::dictionary::decode(cudf::dictionary_column_view(table_result.front()->view()));
  EXPECT_EQ(0, decoded->size());
}

TEST_F(DictionaryScatterTest, ScatterScalar)
{
  cudf::test::strings_column_wrapper strings_target{
    "eee", "aaa", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"};
  auto target = cudf::dictionary::encode(strings_target);
  std::vector<std::reference_wrapper<cudf::scalar const>> source;

  cudf::string_scalar const source_scalar                       = cudf::string_scalar("bbb");
  std::reference_wrapper<cudf::string_scalar const> slr_wrapper = std::ref(source_scalar);
  source.emplace_back(slr_wrapper);
  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map{0, 2, 3, 7};

  auto table_result =
    cudf::scatter(source, cudf::column_view{scatter_map}, cudf::table_view({target->view()}))
      ->release();
  auto decoded =
    cudf::dictionary::decode(cudf::dictionary_column_view(table_result.front()->view()));

  cudf::test::strings_column_wrapper expected{
    "bbb", "aaa", "bbb", "bbb", "ccc", "ccc", "eee", "bbb"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
}

TEST_F(DictionaryScatterTest, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<int64_t> data_source{{1, 5, 7, 9},
                                                              {false, true, true, true}};
  auto source = cudf::dictionary::encode(data_source);
  cudf::test::fixed_width_column_wrapper<int64_t> data_target{
    {1, 5, 5, 3, 7, 1, 4, 2}, {false, true, false, true, true, true, true, true}};
  auto target = cudf::dictionary::encode(data_target);

  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map{7, 2, 3, 1};
  auto table_result =
    cudf::scatter(
      cudf::table_view{{source->view()}}, scatter_map, cudf::table_view{{target->view()}})
      ->release();
  auto decoded =
    cudf::dictionary::decode(cudf::dictionary_column_view(table_result.front()->view()));

  cudf::test::fixed_width_column_wrapper<int64_t> expected{
    {1, 9, 5, 7, 7, 1, 4, 1}, {false, true, true, true, true, true, true, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
}

TEST_F(DictionaryScatterTest, ScalarWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int64_t> data_target{
    {1, 5, 5, 3, 7, 1, 4, 2}, {false, true, false, true, true, true, true, true}};
  auto target = cudf::dictionary::encode(data_target);
  std::vector<std::reference_wrapper<cudf::scalar const>> source;
  cudf::numeric_scalar<int64_t> const source_slr = cudf::test::make_type_param_scalar<int64_t>(100);
  std::reference_wrapper<cudf::numeric_scalar<int64_t> const> slr_wrapper = std::ref(source_slr);
  source.emplace_back(slr_wrapper);

  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map{7, 2, 3, 1, -3};
  auto table_result =
    cudf::scatter(source, scatter_map, cudf::table_view{{target->view()}})->release();

  auto decoded =
    cudf::dictionary::decode(cudf::dictionary_column_view(table_result.front()->view()));

  cudf::test::fixed_width_column_wrapper<int64_t> expected{
    {1, 100, 100, 100, 7, 100, 4, 100}, {false, true, true, true, true, true, true, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
}

TEST_F(DictionaryScatterTest, Error)
{
  cudf::test::strings_column_wrapper strings_source{"this string intentionally left blank"};
  auto source = cudf::dictionary::encode(strings_source);
  cudf::test::fixed_width_column_wrapper<int64_t> integers_target({1, 2, 3});
  auto target = cudf::dictionary::encode(integers_target);
  cudf::test::fixed_width_column_wrapper<int32_t> scatter_map({0});
  EXPECT_THROW(
    cudf::scatter(
      cudf::table_view{{source->view()}}, scatter_map, cudf::table_view{{target->view()}}),
    cudf::data_type_error);
}
