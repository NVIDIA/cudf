/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>

#include <cuda/iterator>

#include <vector>

struct StringsStripTest : public cudf::test::BaseFixture {};

TEST_F(StringsStripTest, StripLeft)
{
  std::vector<char const*> h_strings{"  aBc  ", "   ", nullptr, "aaaa ", "b", "\tccc ddd"};
  std::vector<char const*> h_expected{"aBc  ", "", nullptr, "aaaa ", "b", "ccc ddd"};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::strip(strings_view, cudf::strings::side_type::LEFT);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsStripTest, StripRight)
{
  std::vector<char const*> h_strings{"  aBc  ", "   ", nullptr, "aaaa ", "b", "\tccc ddd"};
  std::vector<char const*> h_expected{"  aBc", "", nullptr, "", "b", "\tccc ddd"};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results =
    cudf::strings::strip(strings_view, cudf::strings::side_type::RIGHT, cudf::string_scalar(" a"));

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsStripTest, StripBoth)
{
  std::vector<char const*> h_strings{"  aBc  ", "   ", nullptr, "ééé ", "b", " ccc dddé"};
  std::vector<char const*> h_expected{"aBc", "", nullptr, "", "b", "ccc ddd"};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results =
    cudf::strings::strip(strings_view, cudf::strings::side_type::BOTH, cudf::string_scalar(" é"));

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsStripTest, EmptyStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::strip(strings_view);
  auto view         = results->view();
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsStripTest, AllEmptyStrings)
{
  auto input =
    cudf::test::strings_column_wrapper({"", "", "", "", "", ""}, {true, true, false, true, true});
  auto results =
    cudf::strings::strip(cudf::strings_column_view(input), cudf::strings::side_type::BOTH);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);
}

TEST_F(StringsStripTest, InvalidParameter)
{
  std::vector<char const*> h_strings{"string left intentionally blank"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_view = cudf::strings_column_view(strings);
  EXPECT_THROW(cudf::strings::strip(
                 strings_view, cudf::strings::side_type::BOTH, cudf::string_scalar("", false)),
               cudf::logic_error);
}
