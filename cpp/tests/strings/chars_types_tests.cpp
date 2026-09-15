/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

#include <array>
#include <vector>

struct StringsCharsTest : public cudf::test::BaseFixture {};

class CharsTypes : public StringsCharsTest,
                   public testing::WithParamInterface<cudf::strings::string_character_types> {};

TEST_P(CharsTypes, AllTypes)
{
  std::vector<char const*> h_strings{"Héllo",
                                     "thesé",
                                     nullptr,
                                     "HERE",
                                     "tést strings",
                                     "",
                                     "1.75",
                                     "-34",
                                     "+9.8",
                                     "17¼",
                                     "x³",
                                     "2³",
                                     " 12⅝",
                                     "1234567890",
                                     "de",
                                     "\t\r\n\f "};

  std::array expecteds{false, false, false, false, false, false, false, false,
                       false, false, false, false, false, true,  false, false,  // decimal
                       false, false, false, false, false, false, false, false,
                       false, true,  false, true,  false, true,  false, false,  // numeric
                       false, false, false, false, false, false, false, false,
                       false, false, false, true,  false, true,  false, false,  // digit
                       true,  true,  false, true,  false, false, false, false,
                       false, false, false, false, false, false, true,  false,  // alpha
                       false, false, false, false, false, false, false, false,
                       false, false, false, false, false, false, false, true,  // space
                       false, false, false, true,  false, false, false, false,
                       false, false, false, false, false, false, false, false,  // upper
                       false, true,  false, false, false, false, false, false,
                       false, false, false, false, false, false, true,  false};  // lower

  auto is_parm = GetParam();

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::all_characters_of_type(strings_view, is_parm);

  int x             = static_cast<int>(is_parm);
  int index         = 0;
  int strings_count = static_cast<int>(h_strings.size());
  while (x >>= 1)
    ++index;
  bool* sub_expected = &expecteds[index * strings_count];

  cudf::test::fixed_width_column_wrapper<bool> expected(
    sub_expected,
    sub_expected + strings_count,
    cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

INSTANTIATE_TEST_CASE_P(StringsCharsTest,
                        CharsTypes,
                        testing::ValuesIn(std::array<cudf::strings::string_character_types, 7>{
                          cudf::strings::string_character_types::DECIMAL,
                          cudf::strings::string_character_types::NUMERIC,
                          cudf::strings::string_character_types::DIGIT,
                          cudf::strings::string_character_types::ALPHA,
                          cudf::strings::string_character_types::SPACE,
                          cudf::strings::string_character_types::UPPER,
                          cudf::strings::string_character_types::LOWER}));

TEST_F(StringsCharsTest, LowerUpper)
{
  cudf::test::strings_column_wrapper strings({"a1", "A1", "a!", "A!", "!1", "aA"});
  auto strings_view = cudf::strings_column_view(strings);
  auto verify_types =
    cudf::strings::string_character_types::LOWER | cudf::strings::string_character_types::UPPER;
  {
    auto results = cudf::strings::all_characters_of_type(
      strings_view, cudf::strings::string_character_types::LOWER, verify_types);
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 0, 1, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::all_characters_of_type(
      strings_view, cudf::strings::string_character_types::UPPER, verify_types);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsCharsTest, Alphanumeric)
{
  std::vector<char const*> h_strings{"Héllo",
                                     "thesé",
                                     nullptr,
                                     "HERE",
                                     "tést strings",
                                     "",
                                     "1.75",
                                     "-34",
                                     "+9.8",
                                     "17¼",
                                     "x³",
                                     "2³",
                                     " 12⅝",
                                     "1234567890",
                                     "de",
                                     "\t\r\n\f "};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::all_characters_of_type(
    strings_view, cudf::strings::string_character_types::ALPHANUM);

  std::vector<bool> h_expected{true,
                               true,
                               false,
                               true,
                               false,
                               false,
                               false,
                               false,
                               false,
                               true,
                               true,
                               true,
                               false,
                               true,
                               true,
                               false};
  cudf::test::fixed_width_column_wrapper<bool> expected(
    h_expected.begin(), h_expected.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCharsTest, AlphaNumericSpace)
{
  std::vector<char const*> h_strings{"Héllo",
                                     "thesé",
                                     nullptr,
                                     "HERE",
                                     "tést strings",
                                     "",
                                     "1.75",
                                     "-34",
                                     "+9.8",
                                     "17¼",
                                     "x³",
                                     "2³",
                                     " 12⅝",
                                     "1234567890",
                                     "de",
                                     "\t\r\n\f "};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto types =
    cudf::strings::string_character_types::ALPHANUM | cudf::strings::string_character_types::SPACE;
  auto results = cudf::strings::all_characters_of_type(
    strings_view, (cudf::strings::string_character_types)types);

  std::vector<bool> h_expected{true,
                               true,
                               false,
                               true,
                               true,
                               false,
                               false,
                               false,
                               false,
                               true,
                               true,
                               true,
                               true,
                               true,
                               true,
                               true};
  cudf::test::fixed_width_column_wrapper<bool> expected(
    h_expected.begin(), h_expected.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCharsTest, Numerics)
{
  std::vector<char const*> h_strings{"Héllo",
                                     "thesé",
                                     nullptr,
                                     "HERE",
                                     "tést strings",
                                     "",
                                     "1.75",
                                     "-34",
                                     "+9.8",
                                     "17¼",
                                     "x³",
                                     "2³",
                                     " 12⅝",
                                     "1234567890",
                                     "de",
                                     "\t\r\n\f "};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto types = cudf::strings::string_character_types::DIGIT |
               cudf::strings::string_character_types::DECIMAL |
               cudf::strings::string_character_types::NUMERIC;
  auto results = cudf::strings::all_characters_of_type(
    strings_view, (cudf::strings::string_character_types)types);

  std::vector<bool> h_expected{false,
                               false,
                               false,
                               false,
                               false,
                               false,
                               false,
                               false,
                               false,
                               true,
                               false,
                               true,
                               false,
                               true,
                               false,
                               false};
  cudf::test::fixed_width_column_wrapper<bool> expected(
    h_expected.begin(), h_expected.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCharsTest, EmptyStrings)
{
  cudf::test::strings_column_wrapper strings({"", "", ""});
  auto strings_view = cudf::strings_column_view(strings);
  cudf::test::fixed_width_column_wrapper<bool> expected({0, 0, 0});
  auto results = cudf::strings::all_characters_of_type(
    strings_view, cudf::strings::string_character_types::ALPHANUM);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCharsTest, FilterCharTypes)
{
  // The example strings are based on issue 5520
  cudf::test::strings_column_wrapper strings(
    {"abc£def", "01234 56789", "℉℧ is not alphanumeric", "but Αγγλικά is", ""});
  auto results =
    cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                             cudf::strings::string_character_types::ALL_TYPES,
                                             cudf::string_scalar(" "),
                                             cudf::strings::string_character_types::ALPHANUM);
  {
    cudf::test::strings_column_wrapper expected(
      {"abc def", "01234 56789", "   is not alphanumeric", "but Αγγλικά is", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  results = cudf::strings::filter_characters_of_type(
    cudf::strings_column_view(strings), cudf::strings::string_character_types::ALPHANUM);
  {
    cudf::test::strings_column_wrapper expected({"£", " ", "℉℧   ", "  ", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  results = cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                                     cudf::strings::string_character_types::SPACE);
  {
    cudf::test::strings_column_wrapper expected(
      {"abc£def", "0123456789", "℉℧isnotalphanumeric", "butΑγγλικάis", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  results =
    cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                             cudf::strings::string_character_types::ALL_TYPES,
                                             cudf::string_scalar("+"),
                                             cudf::strings::string_character_types::SPACE);
  {
    cudf::test::strings_column_wrapper expected(
      {"+++++++", "+++++ +++++", "++ ++ +++ ++++++++++++", "+++ +++++++ ++", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  results = cudf::strings::filter_characters_of_type(
    cudf::strings_column_view(strings), cudf::strings::string_character_types::NUMERIC);
  {
    cudf::test::strings_column_wrapper expected(
      {"abc£def", " ", "℉℧ is not alphanumeric", "but Αγγλικά is", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  results =
    cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                             cudf::strings::string_character_types::ALL_TYPES,
                                             cudf::string_scalar(""),
                                             cudf::strings::string_character_types::NUMERIC);
  {
    cudf::test::strings_column_wrapper expected({"", "0123456789", "", "", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsCharsTest, FilterCharTypesErrors)
{
  cudf::test::strings_column_wrapper strings({"strings left intentionally blank"});
  EXPECT_THROW(
    cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                             cudf::strings::string_character_types::ALL_TYPES,
                                             cudf::string_scalar(""),
                                             cudf::strings::string_character_types::ALL_TYPES),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::filter_characters_of_type(cudf::strings_column_view(strings),
                                             cudf::strings::string_character_types::ALPHANUM,
                                             cudf::string_scalar(""),
                                             cudf::strings::string_character_types::NUMERIC),
    cudf::logic_error);
}

TEST_F(StringsCharsTest, EmptyStringsColumn)
{
  cudf::test::strings_column_wrapper strings;
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::all_characters_of_type(
    strings_view, cudf::strings::string_character_types::ALPHANUM);
  EXPECT_EQ(cudf::type_id::BOOL8, results->view().type().id());
  EXPECT_EQ(0, results->view().size());

  results = cudf::strings::filter_characters_of_type(
    strings_view, cudf::strings::string_character_types::NUMERIC);
  EXPECT_EQ(cudf::type_id::STRING, results->view().type().id());
  EXPECT_EQ(0, results->view().size());
}
