/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "special_chars.h"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/strings/findall.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/transform_iterator.h>

struct StringsFindallTests : public cudf::test::BaseFixture {};

TEST_F(StringsFindallTests, FindallTest)
{
  std::array valids{true, true, true, true, true, false, true, true};
  cudf::test::strings_column_wrapper input(
    {"3-A", "4-May 5-Day 6-Hay", "12-Dec-2021-Jan", "Feb-March", "4 ABC", "", "", "25-9000-Hal"},
    valids.data());
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("\\d+-\\w+");

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"3-A"},
                LCW{"4-May", "5-Day", "6-Hay"},
                LCW{"12-Dec", "2021-Jan"},
                LCW{},
                LCW{},
                LCW{},
                LCW{},
                LCW{"25-9000"}},
               valids.data());
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindallTests, Multiline)
{
  cudf::test::strings_column_wrapper input({"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n"});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("^abc$");
  using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"abc", "abc"}, LCW{"abc"}, LCW{"abc"}, LCW{}, LCW{"abc"}});
  auto prog = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::MULTILINE);
  auto results = cudf::strings::findall(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindallTests, DotAll)
{
  cudf::test::strings_column_wrapper input({"abc\nfa\nef", "fff\nabbc\nfff", "abcdéf", ""});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("b.*f");
  using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"bc\nfa\nef"}, LCW{"bbc\nfff"}, LCW{"bcdéf"}, LCW{}});
  auto prog    = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DOTALL);
  auto results = cudf::strings::findall(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindallTests, SpecialNewLines)
{
  auto input = cudf::test::strings_column_wrapper({"zzé" PARAGRAPH_SEPARATOR "qqq\nzzé",
                                                   "qqq\nzzé" PARAGRAPH_SEPARATOR "lll",
                                                   "zzé",
                                                   "",
                                                   "zzé\r",
                                                   "zzé" LINE_SEPARATOR "zzé" NEXT_LINE});
  auto view  = cudf::strings_column_view(input);

  auto prog =
    cudf::strings::regex_program::create("^zzé$", cudf::strings::regex_flags::EXT_NEWLINE);
  auto results = cudf::strings::findall(view, *prog);
  using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{}, LCW{}, LCW{"zzé"}, LCW{}, LCW{"zzé"}, LCW{}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  auto both_flags = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::EXT_NEWLINE | cudf::strings::regex_flags::MULTILINE);
  auto prog_ml = cudf::strings::regex_program::create("^zzé$", both_flags);
  results      = cudf::strings::findall(view, *prog_ml);
  LCW expected_ml(
    {LCW{"zzé", "zzé"}, LCW{"zzé"}, LCW{"zzé"}, LCW{}, LCW{"zzé"}, LCW{"zzé", "zzé"}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected_ml);
}

TEST_F(StringsFindallTests, MediumRegex)
{
  // This results in 15 regex instructions and falls in the 'medium' range.
  std::string medium_regex = R"(\w+ \w+ \d+)";
  auto prog                = cudf::strings::regex_program::create(medium_regex);

  cudf::test::strings_column_wrapper input({"first words 1234 and just numbers 9876", "neither"});
  auto strings_view = cudf::strings_column_view(input);
  auto results      = cudf::strings::findall(strings_view, *prog);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"first words 1234", "just numbers 9876"}, LCW{}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindallTests, LargeRegex)
{
  // This results in 115 regex instructions and falls in the 'large' range.
  std::string large_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz";
  auto prog = cudf::strings::regex_program::create(large_regex);

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
  auto results      = cudf::strings::findall(strings_view, *prog);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{large_regex.c_str()}, LCW{}, LCW{}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindallTests, FindTest)
{
  auto const valids = cudf::test::iterators::null_at(5);
  cudf::test::strings_column_wrapper input(
    {"3A", "May4", "Jan2021", "March", "A9BC", "", "", "abcdef ghijklm 12345"}, valids);
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("\\d+");

  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::find_re(sv, *prog);
  auto expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 3, 3, -1, 1, 0, -1, 15}, valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindallTests, NoMatches)
{
  cudf::test::strings_column_wrapper input({"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n"});
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("^zzz$");
  using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{}, LCW{}, LCW{}, LCW{}, LCW{}});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindallTests, EmptyTest)
{
  std::string pattern = R"(\w+)";

  auto prog = cudf::strings::regex_program::create(pattern);

  cudf::test::strings_column_wrapper input;
  auto sv = cudf::strings_column_view(input);
  {
    auto results = cudf::strings::findall(sv, *prog);
    using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
    LCW expected;
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto results  = cudf::strings::find_re(sv, *prog);
    auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>{};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
}

TEST_F(StringsFindallTests, OneCaptureGroup)
{
  std::array valids{true, true, true, true, true, false, true, true};
  cudf::test::strings_column_wrapper input(
    {"3-A", "4-May 5-Day 6-Hay", "12-Dec-2021-Jan", "Feb-March", "4 ABC", "", "", "25-9000-Hal"},
    valids.data());
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("(\\d+)-\\w+");

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected(
    {LCW{"3"}, LCW{"4", "5", "6"}, LCW{"12", "2021"}, LCW{}, LCW{}, LCW{}, LCW{}, LCW{"25"}},
    valids.data());
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  expected = LCW({LCW{"3-A"},
                  LCW{"4-May", "5-Day", "6-Hay"},
                  LCW{"12-Dec", "2021-Jan"},
                  LCW{},
                  LCW{},
                  LCW{},
                  LCW{},
                  LCW{"25-9000"}},
                 valids.data());
  prog     = cudf::strings::regex_program::create(
    pattern, cudf::strings::regex_flags::DEFAULT, cudf::strings::capture_groups::NON_CAPTURE);
  results = cudf::strings::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindallTests, AlternationPriorityFirstWins)
{
  // Leftmost-first (first-alternative-wins): "foo" is found instead of "foobar" when both
  // alternatives start at the same position.
  auto input =
    cudf::test::strings_column_wrapper({"foo", "foobar", "foobarbaz", "bar", "xfoobar", ""});
  auto sv   = cudf::strings_column_view(input);
  auto prog = cudf::strings::regex_program::create(
    "foo|foobar", cudf::strings::regex_flags::DEFAULT, cudf::strings::capture_groups::NON_CAPTURE);
  auto results = cudf::strings::findall(sv, *prog);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"foo"}, LCW{"foo"}, LCW{"foo"}, LCW{}, LCW{"foo"}, LCW{}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindallTests, EmptyMatch)
{
  auto input = cudf::test::strings_column_wrapper({" ", "hello world", "é\r\ny"});
  auto sv    = cudf::strings_column_view(input);
  using LCW  = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto expected = LCW({LCW{}, LCW{}, LCW{}});
  auto prog     = cudf::strings::regex_program::create("^$", cudf::strings::regex_flags::MULTILINE);
  auto results  = cudf::strings::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  expected = LCW({LCW{}, LCW{"", "", "", ""}, LCW{"", "", "", ""}});
  prog     = cudf::strings::regex_program::create("\\b");
  results  = cudf::strings::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  expected = LCW({LCW{}, LCW{"", "", "", ""}, LCW{"", "", "", ""}});
  prog     = cudf::strings::regex_program::create("(\\b)");
  results  = cudf::strings::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindallTests, Errors)
{
  auto input   = cudf::test::strings_column_wrapper({"1 One", "2 Two", "3 Three 4 Four", ""});
  auto sv      = cudf::strings_column_view(input);
  auto pattern = std::string("(\\d+)-(\\w+)");
  auto prog    = cudf::strings::regex_program::create(pattern);
  EXPECT_THROW(cudf::strings::findall(sv, *prog), cudf::logic_error);
}
