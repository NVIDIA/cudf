/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "regex_test_utilities.hpp"
#include "special_chars.h"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/strings/findall.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

template <typename RegexBackend>
struct StringsFindallTests : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(StringsFindallTests, cudf::test::regex_backends);

TYPED_TEST(StringsFindallTests, FindallTest)
{
  std::array valids{true, true, true, true, true, false, true, true};
  cudf::test::strings_column_wrapper input(
    {"3-A", "4-May 5-Day 6-Hay", "12-Dec-2021-Jan", "Feb-March", "4 ABC", "", "", "25-9000-Hal"},
    valids.data());
  auto sv = cudf::strings_column_view{input};

  auto pattern = std::string("\\d+-\\w+");

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected(
    {{"3-A"}, {"4-May", "5-Day", "6-Hay"}, {"12-Dec", "2021-Jan"}, {}, {}, {}, {}, {"25-9000"}},
    valids.data());
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::FINDALL, pattern);
  auto results = TypeParam::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TYPED_TEST(StringsFindallTests, Multiline)
{
  cudf::test::strings_column_wrapper input({"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n"});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("^abc$");
  using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({{"abc", "abc"}, {"abc"}, {"abc"}, {}, {"abc"}});
  auto prog = TypeParam::create(
    cudf::experimental::regex_operation::FINDALL, pattern, cudf::strings::regex_flags::MULTILINE);
  auto results = TypeParam::findall(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TYPED_TEST(StringsFindallTests, DotAll)
{
  cudf::test::strings_column_wrapper input({"abc\nfa\nef", "fff\nabbc\nfff", "abcdéf", ""});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("b.*f");
  using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({{"bc\nfa\nef"}, {"bbc\nfff"}, {"bcdéf"}, {}});
  auto prog = TypeParam::create(
    cudf::experimental::regex_operation::FINDALL, pattern, cudf::strings::regex_flags::DOTALL);
  auto results = TypeParam::findall(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TYPED_TEST(StringsFindallTests, SpecialNewLines)
{
  auto input = cudf::test::strings_column_wrapper({"zzé" PARAGRAPH_SEPARATOR "qqq\nzzé",
                                                   "qqq\nzzé" PARAGRAPH_SEPARATOR "lll",
                                                   "zzé",
                                                   "",
                                                   "zzé\r",
                                                   "zzé" LINE_SEPARATOR "zzé" NEXT_LINE});
  auto view  = cudf::strings_column_view(input);

  auto prog = TypeParam::create(
    cudf::experimental::regex_operation::FINDALL, "^zzé$", cudf::strings::regex_flags::EXT_NEWLINE);
  auto results = TypeParam::findall(view, *prog);
  using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({{}, {}, {"zzé"}, {}, {"zzé"}, {}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  auto both_flags = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::EXT_NEWLINE | cudf::strings::regex_flags::MULTILINE);
  auto prog_ml =
    TypeParam::create(cudf::experimental::regex_operation::FINDALL, "^zzé$", both_flags);
  results = TypeParam::findall(view, *prog_ml);
  LCW expected_ml({{"zzé", "zzé"}, {"zzé"}, {"zzé"}, {}, {"zzé"}, {"zzé", "zzé"}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected_ml);
}

TYPED_TEST(StringsFindallTests, MediumRegex)
{
  // This results in 15 regex instructions and falls in the 'medium' range.
  std::string medium_regex = R"(\w+ \w+ \d+)";
  auto prog = TypeParam::create(cudf::experimental::regex_operation::FINDALL, medium_regex);

  cudf::test::strings_column_wrapper input({"first words 1234 and just numbers 9876", "neither"});
  auto strings_view = cudf::strings_column_view(input);
  auto results      = TypeParam::findall(strings_view, *prog);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({{"first words 1234", "just numbers 9876"}, {}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TYPED_TEST(StringsFindallTests, LargeRegex)
{
  // This results in 115 regex instructions and falls in the 'large' range.
  std::string large_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz";
  auto prog = TypeParam::create(cudf::experimental::regex_operation::FINDALL, large_regex);

  cudf::test::strings_column_wrapper input(
    {"hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
     "http://www.world.com I'm here @home zzzz",
     "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012"
     "34"
     "5678901234567890",
     "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmn"
     "op"
     "qrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"});

  auto strings_view = cudf::strings_column_view(input);
  auto results      = TypeParam::findall(strings_view, *prog);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({{large_regex.c_str()}, {}, {}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TYPED_TEST(StringsFindallTests, FindTest)
{
  auto const valids = cudf::test::iterators::null_at(5);
  cudf::test::strings_column_wrapper input(
    {"3A", "May4", "Jan2021", "March", "A9BC", "", "", "abcdef ghijklm 12345"}, valids);
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("\\d+");

  auto prog    = TypeParam::create(cudf::experimental::regex_operation::FIND, pattern);
  auto results = TypeParam::find_re(sv, *prog);
  auto expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 3, 3, -1, 1, 0, -1, 15}, valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TYPED_TEST(StringsFindallTests, NoMatches)
{
  cudf::test::strings_column_wrapper input({"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n"});
  auto sv = cudf::strings_column_view{input};

  auto pattern = std::string("^zzz$");
  using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({{}, {}, {}, {}, {}});
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::FINDALL, pattern);
  auto results = TypeParam::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TYPED_TEST(StringsFindallTests, EmptyTest)
{
  std::string pattern = R"(\w+)";

  auto prog = TypeParam::create(cudf::experimental::regex_operation::FINDALL, pattern);

  cudf::test::strings_column_wrapper input;
  auto sv = cudf::strings_column_view(input);
  {
    auto results = TypeParam::findall(sv, *prog);
    using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
    LCW expected;
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    prog          = TypeParam::create(cudf::experimental::regex_operation::FIND, pattern);
    auto results  = TypeParam::find_re(sv, *prog);
    auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>{};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
}

TYPED_TEST(StringsFindallTests, OneCaptureGroup)
{
  std::array valids{true, true, true, true, true, false, true, true};
  cudf::test::strings_column_wrapper input(
    {"3-A", "4-May 5-Day 6-Hay", "12-Dec-2021-Jan", "Feb-March", "4 ABC", "", "", "25-9000-Hal"},
    valids.data());
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("(\\d+)-\\w+");

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({{"3"}, {"4", "5", "6"}, {"12", "2021"}, {}, {}, {}, {}, {"25"}}, valids.data());
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::FINDALL, pattern);
  auto results = TypeParam::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  expected =
    LCW({{"3-A"}, {"4-May", "5-Day", "6-Hay"}, {"12-Dec", "2021-Jan"}, {}, {}, {}, {}, {"25-9000"}},
        valids.data());
  prog    = TypeParam::create(cudf::experimental::regex_operation::FINDALL,
                           pattern,
                           cudf::strings::regex_flags::DEFAULT,
                           cudf::strings::capture_groups::NON_CAPTURE);
  results = TypeParam::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TYPED_TEST(StringsFindallTests, AlternationPriorityFirstWins)
{
  // Leftmost-first (first-alternative-wins): "foo" is found instead of "foobar" when both
  // alternatives start at the same position.
  auto input =
    cudf::test::strings_column_wrapper({"foo", "foobar", "foobarbaz", "bar", "xfoobar", ""});
  auto sv      = cudf::strings_column_view(input);
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::FINDALL,
                                "foo|foobar",
                                cudf::strings::regex_flags::DEFAULT,
                                cudf::strings::capture_groups::NON_CAPTURE);
  auto results = TypeParam::findall(sv, *prog);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({{"foo"}, {"foo"}, {"foo"}, {}, {"foo"}, {}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TYPED_TEST(StringsFindallTests, EmptyMatch)
{
  auto input = cudf::test::strings_column_wrapper({" ", "hello world", "é\r\ny"});
  auto sv    = cudf::strings_column_view(input);
  using LCW  = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto expected = LCW({{}, {}, {}});
  auto prog     = TypeParam::create(
    cudf::experimental::regex_operation::FINDALL, "^$", cudf::strings::regex_flags::MULTILINE);
  auto results = TypeParam::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  expected = LCW({{}, {"", "", "", ""}, {"", "", "", ""}});
  prog     = TypeParam::create(cudf::experimental::regex_operation::FINDALL, "\\b");
  results  = TypeParam::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  expected = LCW({{}, {"", "", "", ""}, {"", "", "", ""}});
  prog     = TypeParam::create(cudf::experimental::regex_operation::FINDALL, "(\\b)");
  results  = TypeParam::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TYPED_TEST(StringsFindallTests, Errors)
{
  auto input   = cudf::test::strings_column_wrapper({"1 One", "2 Two", "3 Three 4 Four", ""});
  auto sv      = cudf::strings_column_view(input);
  auto pattern = std::string("(\\d+)-(\\w+)");
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::FINDALL, pattern);
  EXPECT_THROW(TypeParam::findall(sv, *prog), cudf::logic_error);
}
