/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

#include <vector>

struct StringsAttributesTest : public cudf::test::BaseFixture {};

TEST_F(StringsAttributesTest, CodePoints)
{
  std::vector<char const*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::code_points(strings_view);

    cudf::test::fixed_width_column_wrapper<int32_t> expected{
      101, 101, 101, 98, 98, 97, 97, 98, 98, 98, 50089, 50089, 50089};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsAttributesTest, ZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  cudf::column_view expected_column(cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);

  auto results = cudf::strings::count_bytes(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected_column);
  results = cudf::strings::count_characters(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected_column);
  results = cudf::strings::code_points(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected_column);
}

TEST_F(StringsAttributesTest, StringsLengths)
{
  std::vector<char const*> h_strings{
    "eee", "bb", nullptr, "", "aa", "ééé", "something a bit longer than 32 bytes"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::count_characters(strings_view);
    std::vector<int32_t> h_expected{3, 2, 0, 0, 2, 3, 36};
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::count_bytes(strings_view);
    std::vector<int32_t> h_expected{3, 2, 0, 0, 2, 6, 36};
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsAttributesTest, StringsLengthsLong)
{
  std::vector<std::string> h_strings(
    40000, "something a bit longer than 32 bytes ééé ééé ééé ééé ééé ééé ééé");
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::count_characters(strings_view);
  std::vector<int32_t> h_expected(h_strings.size(), 64);
  cudf::test::fixed_width_column_wrapper<int32_t> expected(h_expected.begin(), h_expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}
