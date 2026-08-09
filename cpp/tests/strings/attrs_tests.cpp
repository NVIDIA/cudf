/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsAttributesTest : public cudf::test::BaseFixture {};

TEST_F(StringsAttributesTest, CodePoints)
{
  std::vector<char const*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
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
  // The two counts are counts of rows' worth of characters and so are size_type wide, while a code
  // point is a Unicode scalar value, which is 32 bits whatever size_type is.
  cudf::column_view expected_counts(
    cudf::data_type{cudf::type_to_id<cudf::size_type>()}, 0, nullptr, nullptr, 0);
  cudf::column_view expected_code_points(
    cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);

  auto results = cudf::strings::count_bytes(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected_counts);
  results = cudf::strings::count_characters(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected_counts);
  results = cudf::strings::code_points(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected_code_points);
}

TEST_F(StringsAttributesTest, StringsLengths)
{
  std::vector<char const*> h_strings{
    "eee", "bb", nullptr, "", "aa", "ééé", "something a bit longer than 32 bytes"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::count_characters(strings_view);
    std::vector<int32_t> h_expected{3, 2, 0, 0, 2, 3, 36};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::count_bytes(strings_view);
    std::vector<int32_t> h_expected{3, 2, 0, 0, 2, 6, 36};
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

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
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(h_expected.begin(),
                                                                   h_expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}
