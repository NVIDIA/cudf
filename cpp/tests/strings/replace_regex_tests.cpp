/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "special_chars.h"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <ranges>
#include <span>
#include <vector>

struct StringsReplaceRegexTest : public cudf::test::BaseFixture {};

TEST_F(StringsReplaceRegexTest, ReplaceRegexTest)
{
  std::vector<char const*> h_strings{"the quick brown fox jumps over the lazy dog",
                                     "the fat cat lays next to the other accénted cat",
                                     "a slow moving turtlé cannot catch the bird",
                                     "which can be composéd together to form a more complete",
                                     "thé result does not include the value in the sum in",
                                     "",
                                     nullptr};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<char const*> h_expected{"= quick brown fox jumps over = lazy dog",
                                      "= fat cat lays next to = other accénted cat",
                                      "a slow moving turtlé cannot catch = bird",
                                      "which can be composéd together to form a more complete",
                                      "thé result does not include = value in = sum in",
                                      "",
                                      nullptr};

  auto pattern = std::string("(\\bthe\\b)");
  auto repl    = cudf::string_scalar("=");
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_re(strings_view, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, InvalidRegex)
{
  // these are quantifiers that do not have a preceding character/class
  EXPECT_THROW(cudf::strings::regex_program::create("*"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("|"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("ab(*)"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("\\"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("\\p"), cudf::logic_error);
}

TEST_F(StringsReplaceRegexTest, WithEmptyPattern)
{
  std::vector<char const*> h_strings{"asd", "xcv"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));
  auto strings_view = cudf::strings_column_view(strings);

  auto empty_pattern = std::string("");
  auto repl          = cudf::string_scalar("bbb");
  auto prog          = cudf::strings::regex_program::create(empty_pattern);
  auto results       = cudf::strings::replace_re(strings_view, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, strings);
}

TEST_F(StringsReplaceRegexTest, MultiReplacement)
{
  cudf::test::strings_column_wrapper input({"aba bcd aba", "abababa abababa"});
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("aba");
  auto repl    = cudf::string_scalar("_");
  cudf::test::strings_column_wrapper expected({"_ bcd _", "_b_ abababa"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_re(sv, *prog, repl, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results = cudf::strings::replace_re(sv, *prog, repl, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, input);
}

TEST_F(StringsReplaceRegexTest, WordBoundary)
{
  cudf::test::strings_column_wrapper input({"aba bcd\naba", "zéz", "A1B2-é3", "e é", "_", "a_b"});
  auto sv = cudf::strings_column_view(input);

  auto pattern  = std::string("\\b");
  auto repl     = cudf::string_scalar("X");
  auto expected = cudf::test::strings_column_wrapper(
    {"XabaX XbcdX\nXabaX", "XzézX", "XA1B2X-Xé3X", "XeX XéX", "X_X", "Xa_bX"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  pattern  = std::string("\\B");
  expected = cudf::test::strings_column_wrapper(
    {"aXbXa bXcXd\naXbXa", "zXéXz", "AX1XBX2-éX3", "e é", "_", "aX_Xb"});
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, Alternation)
{
  cudf::test::strings_column_wrapper input(
    {"16  6  brr  232323  1  hello  90", "123 ABC 00 2022", "abé123  4567  89xyz"});
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string(R"((^|\s)\d+(\s|$))");
  auto repl    = cudf::string_scalar("_");
  auto expected =
    cudf::test::strings_column_wrapper({"__ brr __ hello _", "_ABC_2022", "abé123 _ 89xyz"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  pattern = std::string(R"((\s|^)\d+($|\s))");
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ZeroLengthMatch)
{
  cudf::test::strings_column_wrapper input({"DD", "zéz", "DsDs", ""});
  auto sv = cudf::strings_column_view(input);

  auto pattern  = std::string("D*");
  auto repl     = cudf::string_scalar("_");
  auto expected = cudf::test::strings_column_wrapper({"__", "_z_é_z_", "__s__s_", "_"});
  auto prog     = cudf::strings::regex_program::create(pattern);
  auto results  = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  pattern  = std::string("D?s?");
  expected = cudf::test::strings_column_wrapper({"___", "_z_é_z_", "___", "_"});
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ZeroRangeQuantifier)
{
  auto input = cudf::test::strings_column_wrapper({"a", "", "123", "XYAZ", "abc", "zéyab"});
  auto sv    = cudf::strings_column_view(input);

  auto pattern  = std::string("A{0,5}");
  auto prog     = cudf::strings::regex_program::create(pattern);
  auto repl     = cudf::string_scalar("_");
  auto expected = cudf::test::strings_column_wrapper(
    {"_a_", "_", "_1_2_3_", "_X_Y__Z_", "_a_b_c_", "_z_é_y_a_b_"});
  auto results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern = std::string("[a0-9]{0,2}");
  prog    = cudf::strings::regex_program::create(pattern);
  expected =
    cudf::test::strings_column_wrapper({"__", "_", "___", "_X_Y_A_Z_", "__b_c_", "_z_é_y__b_"});
  results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern = std::string("(?:ab){0,3}");
  prog    = cudf::strings::regex_program::create(pattern);
  expected =
    cudf::test::strings_column_wrapper({"_a_", "_", "_1_2_3_", "_X_Y_A_Z_", "__c_", "_z_é_y__"});
  results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceRegexTest, Multiline)
{
  auto const multiline = cudf::strings::regex_flags::MULTILINE;

  cudf::test::strings_column_wrapper input({"bcd\naba\nefg", "aba\naba abab\naba", "aba"});
  auto sv = cudf::strings_column_view(input);

  // single-replace
  auto pattern = std::string("^aba$");
  auto repl    = cudf::string_scalar("_");
  cudf::test::strings_column_wrapper expected_ml({"bcd\n_\nefg", "_\naba abab\n_", "_"});
  auto prog    = cudf::strings::regex_program::create(pattern, multiline);
  auto results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_ml);

  cudf::test::strings_column_wrapper expected({"bcd\naba\nefg", "aba\naba abab\naba", "_"});
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  // backref-replace
  auto repl_template = std::string("[\\1]");
  pattern            = std::string("(^aba)");
  cudf::test::strings_column_wrapper br_expected_ml(
    {"bcd\n[aba]\nefg", "[aba]\n[aba] abab\n[aba]", "[aba]"});
  prog    = cudf::strings::regex_program::create(pattern, multiline);
  results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, br_expected_ml);

  cudf::test::strings_column_wrapper br_expected(
    {"bcd\naba\nefg", "[aba]\naba abab\naba", "[aba]"});
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, br_expected);
}

TEST_F(StringsReplaceRegexTest, SpecialNewLines)
{
  auto input   = cudf::test::strings_column_wrapper({"zzé" NEXT_LINE "qqq" NEXT_LINE "zzé",
                                                     "qqq" NEXT_LINE "zzé" NEXT_LINE "lll",
                                                     "zzé",
                                                     "",
                                                     "zzé" PARAGRAPH_SEPARATOR,
                                                     "abc\rzzé\r"});
  auto view    = cudf::strings_column_view(input);
  auto repl    = cudf::string_scalar("_");
  auto pattern = std::string("^zzé$");
  auto prog =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::EXT_NEWLINE);
  auto results  = cudf::strings::replace_re(view, *prog, repl);
  auto expected = cudf::test::strings_column_wrapper({"zzé" NEXT_LINE "qqq" NEXT_LINE "zzé",
                                                      "qqq" NEXT_LINE "zzé" NEXT_LINE "lll",
                                                      "_",
                                                      "",
                                                      "_" PARAGRAPH_SEPARATOR,
                                                      "abc\rzzé\r"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  auto both_flags = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::EXT_NEWLINE | cudf::strings::regex_flags::MULTILINE);
  auto prog_ml = cudf::strings::regex_program::create(pattern, both_flags);
  results      = cudf::strings::replace_re(view, *prog_ml, repl);
  expected     = cudf::test::strings_column_wrapper({"_" NEXT_LINE "qqq" NEXT_LINE "_",
                                                     "qqq" NEXT_LINE "_" NEXT_LINE "lll",
                                                     "_",
                                                     "",
                                                     "_" PARAGRAPH_SEPARATOR,
                                                     "abc\r_\r"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

  auto repl_template = std::string("[\\1]");
  pattern            = std::string("(^zzé$)");
  prog               = cudf::strings::regex_program::create(pattern, both_flags);
  results            = cudf::strings::replace_with_backrefs(view, *prog, repl_template);
  expected = cudf::test::strings_column_wrapper({"[zzé]" NEXT_LINE "qqq" NEXT_LINE "[zzé]",
                                                 "qqq" NEXT_LINE "[zzé]" NEXT_LINE "lll",
                                                 "[zzé]",
                                                 "",
                                                 "[zzé]" PARAGRAPH_SEPARATOR,
                                                 "abc\r[zzé]\r"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexTest)
{
  std::vector<char const*> h_strings{"the quick brown fox jumps over the lazy dog",
                                     "the fat cat lays next to the other accénted cat",
                                     "a slow moving turtlé cannot catch the bird",
                                     "which can be composéd together to form a more complete",
                                     "thé result does not include the value in the sum in",
                                     "",
                                     nullptr};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));
  auto sv = cudf::strings_column_view(strings);

  std::vector<char const*> h_expected{"the-quick-brown-fox-jumps-over-the-lazy-dog",
                                      "the-fat-cat-lays-next-to-the-other-accénted-cat",
                                      "a-slow-moving-turtlé-cannot-catch-the-bird",
                                      "which-can-be-composéd-together-to-form-a more-complete",
                                      "thé-result-does-not-include-the-value-in-the-sum-in",
                                      "",
                                      nullptr};

  auto pattern       = std::string("(\\w) (\\w)");
  auto repl_template = std::string("\\1-\\2");
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexAltIndexPatternTest)
{
  cudf::test::strings_column_wrapper input({"12-3 34-5 67-89", "0-99: 777-888:: 5673-0"});
  auto sv = cudf::strings_column_view(input);

  auto pattern       = std::string("(\\d+)-(\\d+)");
  auto repl_template = std::string("${2} X ${1}0");

  cudf::test::strings_column_wrapper expected(
    {"3 X 120 5 X 340 89 X 670", "99 X 00: 888 X 7770:: 0 X 56730"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexReversedTest)
{
  cudf::test::strings_column_wrapper strings(
    {"A543", "Z756", "", "tést-string", "two-thréé four-fivé", "abcd-éfgh", "tést-string-again"});
  auto sv = cudf::strings_column_view(strings);

  auto pattern       = std::string("([a-z])-([a-zé])");
  auto repl_template = std::string("X\\2+\\1Z");

  cudf::test::strings_column_wrapper expected({"A543",
                                               "Z756",
                                               "",
                                               "tésXs+tZtring",
                                               "twXt+oZhréé fouXf+rZivé",
                                               "abcXé+dZfgh",
                                               "tésXs+tZtrinXa+gZgain"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, BackrefWithGreedyQuantifier)
{
  cudf::test::strings_column_wrapper input(
    {"<h1>title</h1><h2>ABC</h2>", "<h1>1234567</h1><h2>XYZ</h2>"});
  auto sv = cudf::strings_column_view(input);

  auto pattern       = std::string("<h1>(.*)</h1><h2>(.*)</h2>");
  auto repl_template = std::string("<h2>\\1</h2><p>\\2</p>");

  cudf::test::strings_column_wrapper expected(
    {"<h2>title</h2><p>ABC</p>", "<h2>1234567</h2><p>XYZ</p>"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  pattern = std::string("<h1>([a-z\\d]+)</h1><h2>([A-Z]+)</h2>");
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexZeroIndexTest)
{
  cudf::test::strings_column_wrapper strings(
    {"TEST123", "TEST1TEST2", "TEST2-TEST1122", "TEST1-TEST-T", "TES3"});
  auto sv = cudf::strings_column_view(strings);

  auto pattern       = std::string("(TEST)(\\d+)");
  auto repl_template = std::string("${0}: ${1}, ${2}; ");

  cudf::test::strings_column_wrapper expected({
    "TEST123: TEST, 123; ",
    "TEST1: TEST, 1; TEST2: TEST, 2; ",
    "TEST2: TEST, 2; -TEST1122: TEST, 1122; ",
    "TEST1: TEST, 1; -TEST-T",
    "TES3",
  });
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsWithEmptyCapture)
{
  cudf::test::strings_column_wrapper input({"one\ntwo", "three\n\n", "four\r\n"});
  auto sv = cudf::strings_column_view(input);

  // https://github.com/rapidsai/cudf/issues/13404
  auto pattern       = std::string("(\r\n|\r)?$");
  auto repl_template = std::string("[\\1]");
  auto expected =
    cudf::test::strings_column_wrapper({"one\ntwo[]", "three\n[]\n[]", "four[\r\n][]"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  // https://github.com/rapidsai/cudf/issues/22707
  pattern  = std::string("^(a?)");
  expected = cudf::test::strings_column_wrapper({"[]one\ntwo", "[]three\n\n", "[]four\r\n"});
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexErrorTest)
{
  cudf::test::strings_column_wrapper strings({"this string left intentionally blank"});
  auto view = cudf::strings_column_view(strings);

  // group index(3) exceeds the group count(2)
  auto prog = cudf::strings::regex_program::create("(\\w).(\\w)");
  EXPECT_THROW(cudf::strings::replace_with_backrefs(view, *prog, "\\3"), cudf::logic_error);
  prog = cudf::strings::regex_program::create("");
  EXPECT_THROW(cudf::strings::replace_with_backrefs(view, *prog, "\\1"), cudf::logic_error);
  prog = cudf::strings::regex_program::create("(\\w)");
  EXPECT_THROW(cudf::strings::replace_with_backrefs(view, *prog, ""), cudf::logic_error);
}

TEST_F(StringsReplaceRegexTest, MediumReplaceRegex)
{
  // This results in 95 regex instructions and falls in the 'medium' range.
  std::string medium_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com";
  auto prog = cudf::strings::regex_program::create(medium_regex);

  std::vector<char const*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com thats all",
    "12345678901234567890",
    "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::replace_re(strings_view, *prog);
  std::vector<char const*> h_expected{
    " thats all", "12345678901234567890", "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceRegexTest, LargeReplaceRegex)
{
  // This results in 117 regex instructions and falls in the 'large' range.
  std::string large_regex =
    "hello @abc @def world The (quick) brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz";
  auto prog = cudf::strings::regex_program::create(large_regex);

  std::vector<char const*> h_strings{
    "zzzz hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz",
    "12345678901234567890",
    "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::replace_re(strings_view, *prog);
  std::vector<char const*> h_expected{
    "zzzz ", "12345678901234567890", "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceRegexTest, CrlfLineAnchorExtNewline)
{
  // \r\n is a single terminator: regexp_replace("abc\r\n","abc$","[X]") -> "[X]\r\n".
  // Expected values verified against OpenJDK 17 java.util.regex replaceAll (default flags).
  auto input = cudf::test::strings_column_wrapper(
    {"abc\r\n", "abc\n", "abc\r", "abc", "a\r\nb", "abc\r\n\r\n", "", "abc" NEXT_LINE});
  auto view = cudf::strings_column_view(input);
  auto prog = cudf::strings::regex_program::create("abc$", cudf::strings::regex_flags::EXT_NEWLINE);
  auto repl = cudf::string_scalar("[X]");

  auto results  = cudf::strings::replace_re(view, *prog, repl);
  auto expected = cudf::test::strings_column_wrapper(
    {"[X]\r\n", "[X]\n", "[X]\r", "[X]", "a\r\nb", "abc\r\n\r\n", "", "[X]" NEXT_LINE});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceRegexTest, CrlfEdgeCasesExtNewline)
{
  // Full edge column; expecteds verified vs OpenJDK 17 replaceAll (default flags).
  // Each row bundles every replacement output for one input, so adding an input or
  // a new replace pattern touches a single row or column.
  struct edge_case {
    char const* s;
    char const* exp_abc_dollar_X;  // replace_re             "abc$"   -> "[X]"   EXT
    char const* exp_abc_backref;   // replace_with_backrefs  "(abc)$" -> "[\\1]" EXT
  };
  // clang-format off
  constexpr static edge_case cases[] = {
    {"abc\r\n",       "[X]\r\n",       "[abc]\r\n"},
    {"abc\n",         "[X]\n",         "[abc]\n"},
    {"abc\r",         "[X]\r",         "[abc]\r"},
    {"abc",           "[X]",           "[abc]"},
    {"a\r\nb",        "a\r\nb",        "a\r\nb"},
    {"abc\r\n\r\n",   "abc\r\n\r\n",   "abc\r\n\r\n"},
    {"",              "",              ""},
    {"abc" NEXT_LINE, "[X]" NEXT_LINE, "[abc]" NEXT_LINE},
    {"a\nb\r\nc",     "a\nb\r\nc",     "a\nb\r\nc"},
    {"\r\n",          "\r\n",          "\r\n"},
    {"\r\nabc",       "\r\n[X]",       "\r\n[abc]"},
    {"x\n\r",         "x\n\r",         "x\n\r"},
    {"a\r\rb",        "a\r\rb",        "a\r\rb"},
    {"a\n\nb",        "a\n\nb",        "a\n\nb"},
  };
  // clang-format on

  auto strings_view = std::span(cases) | std::views::transform(&edge_case::s);
  auto input        = cudf::test::strings_column_wrapper(strings_view.begin(), strings_view.end());
  auto view         = cudf::strings_column_view(input);
  auto const EXT    = cudf::strings::regex_flags::EXT_NEWLINE;

  auto str_col = [](char const* edge_case::* m) {
    auto v = std::span(cases) | std::views::transform([m](auto const& c) { return c.*m; });
    return cudf::test::strings_column_wrapper(v.begin(), v.end());
  };

  {  // replace_re  abc$ -> [X]   (\r\n preserved as a unit)
    auto p = cudf::strings::regex_program::create("abc$", EXT);
    auto r = cudf::string_scalar("[X]");
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::replace_re(view, *p, r),
                                   str_col(&edge_case::exp_abc_dollar_X));
  }
  {  // replace_with_backrefs  (abc)$ -> [\1]   (the spark-rapids scenario, native pattern)
    auto p = cudf::strings::regex_program::create("(abc)$", EXT);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::replace_with_backrefs(view, *p, "[\\1]"),
                                   str_col(&edge_case::exp_abc_backref));
  }
}

TEST_F(StringsReplaceRegexTest, AlternationPriorityFirstWins)
{
  // Leftmost-first (first-alternative-wins): when a shorter first alternative is a prefix of a
  // longer second, the shorter match is consumed and the remainder is left for the next search.
  auto repl = cudf::string_scalar("X");

  {
    // "foo" wins over "foobar": "foobar" becomes "Xbar".
    auto input =
      cudf::test::strings_column_wrapper({"foo", "foobar", "foobarbaz", "bar", "xfoobar", ""});
    auto sv      = cudf::strings_column_view(input);
    auto prog    = cudf::strings::regex_program::create("foo|foobar");
    auto results = cudf::strings::replace_re(sv, *prog, repl);
    cudf::test::strings_column_wrapper expected({"X", "Xbar", "Xbarbaz", "bar", "xXbar", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // "cat" wins over "catch": "catch" becomes "Xch".
    auto input   = cudf::test::strings_column_wrapper({"cat", "catch", "catfish", "dog", ""});
    auto sv      = cudf::strings_column_view(input);
    auto prog    = cudf::strings::regex_program::create("cat|catch");
    auto results = cudf::strings::replace_re(sv, *prog, repl);
    cudf::test::strings_column_wrapper expected({"X", "Xch", "Xfish", "dog", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}
