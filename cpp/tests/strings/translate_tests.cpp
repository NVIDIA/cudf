/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/translate.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <vector>

struct StringsTranslateTest : public cudf::test::BaseFixture {};

std::pair<cudf::char_utf8, cudf::char_utf8> make_entry(char const* from, char const* to)
{
  cudf::char_utf8 in  = 0;
  cudf::char_utf8 out = 0;
  cudf::strings::detail::to_char_utf8(from, in);
  if (to) cudf::strings::detail::to_char_utf8(to, out);
  return std::pair(in, out);
}

TEST_F(StringsTranslateTest, Translate)
{
  std::vector<char const*> h_strings{"eee ddd", "bb cc", nullptr, "", "aa", "débd"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> translate_table{
    make_entry("b", nullptr), make_entry("a", "A"), make_entry("é", "E"), make_entry("e", "_")};
  auto results = cudf::strings::translate(strings_view, translate_table);

  std::vector<char const*> h_expected{"___ ddd", " cc", nullptr, "", "AA", "dEd"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsTranslateTest, ZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> translate_table;
  auto results = cudf::strings::translate(strings_view, translate_table);
  cudf::test::expect_column_empty(results->view());
  results = cudf::strings::filter_characters(strings_view, translate_table);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsTranslateTest, FilterCharacters)
{
  std::vector<char const*> h_strings{"eee ddd", "bb cc", nullptr, "", "12309", "débd"};
  auto validity =
    cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> filter_table{
    make_entry("a", "c"), make_entry("é", "ú"), make_entry("0", "9")};
  {
    auto results = cudf::strings::filter_characters(strings_view, filter_table);
    cudf::test::strings_column_wrapper expected({"", "bbcc", "", "", "12309", "éb"}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::filter_characters(
      strings_view, filter_table, cudf::strings::filter_type::REMOVE);
    cudf::test::strings_column_wrapper expected({"eee ddd", " ", "", "", "", "dd"}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::filter_characters(
      strings_view, filter_table, cudf::strings::filter_type::KEEP, cudf::string_scalar("_"));
    cudf::test::strings_column_wrapper expected({"_______", "bb_cc", "", "", "12309", "_éb_"},
                                                validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::filter_characters(
      strings_view, filter_table, cudf::strings::filter_type::REMOVE, cudf::string_scalar("++"));
    cudf::test::strings_column_wrapper expected(
      {"eee ddd", "++++ ++++", "", "", "++++++++++", "d++++d"}, validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsTranslateTest, ErrorTest)
{
  cudf::test::strings_column_wrapper h_strings({"string left intentionally blank"});
  auto strings_view = cudf::strings_column_view(h_strings);
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> filter_table;
  EXPECT_THROW(
    cudf::strings::filter_characters(
      strings_view, filter_table, cudf::strings::filter_type::KEEP, cudf::string_scalar("", false)),
    cudf::logic_error);
}
