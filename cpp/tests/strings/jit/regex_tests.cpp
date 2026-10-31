/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../regex_test_utilities.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/experimental/regex.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::unique_ptr<cudf::column> make_int64_offsets_column(cudf::column_view const& input)
{
  auto strings  = cudf::strings_column_view{input};
  auto offsets  = cudf::cast(strings.offsets(), cudf::data_type{cudf::type_id::INT64});
  auto contents = cudf::column{input}.release();
  return cudf::make_strings_column(
    input.size(),
    std::move(offsets),
    std::move(*contents.data),
    input.null_count(),
    cudf::copy_bitmask(input, cudf::get_default_stream(), cudf::get_current_device_resource_ref()));
}

}  // namespace

struct RegexJitProgramTest : public cudf::test::BaseFixture {};

TEST_F(RegexJitProgramTest, ScopedFlags)
{
  struct test_case {
    char const* pattern;
    char const* text;
    bool expected;
  };
  // Includes adaptations of CPython test_scoped_flags and mrab-regex test_ignore_case.
  for (auto const& test :
       std::initializer_list<test_case>{{R"((?i)a(?-i:b)(?i:c))", "AbC", true},
                                        {R"((?i)a(?-i:b)(?i:c))", "ABC", false},
                                        {R"((?m:^a$)(?-m:$))", "x\na\n", true},
                                        {R"((?m:^a$)(?-m:$))", "x\na\nb", false},
                                        {R"((?m)(?-m:^a$))", "x\na\n", false},
                                        {R"((?m)(?-m:^a$))", "a\n", true},
                                        {R"((?s:a.)(?-s:.))", "a\nb", true},
                                        {R"((?s:a.)(?-s:.))", "ab\n", false},
                                        {R"((?a:\b\w+\b)(?u:\w+\b))", "aé", true},
                                        {R"((?a:\b\w+\b)(?u:\w+\b))", "éa", false},
                                        {R"((?u:\b)(?a:\B)é)", "é", true},
                                        {R"((?x:a (?-x: b ) c))", "a b c", true},
                                        {"(?x) a \\# [ ] # ignored\n (?i:b)", "a# B", true},
                                        {R"((?i)(?=a)a(?<=A)(?-i:b))", "Ab", true},
                                        {R"((?i)s)", "ſ", true},
                                        {R"((?ai)Й)", "й", false}}) {
    SCOPED_TRACE(test.pattern);
    SCOPED_TRACE(test.text);
    cudf::test::strings_column_wrapper input({test.text});
    auto program = cudf::experimental::regex_jit_program::create(
      test.pattern, cudf::experimental::regex_operation::CONTAINS);
    auto result = cudf::experimental::contains_re(cudf::strings_column_view{input}, *program);
    cudf::test::fixed_width_column_wrapper<bool> expected({test.expected});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(RegexJitProgramTest, ScopedFlagsPreserveCaptures)
{
  cudf::test::strings_column_wrapper input({"ABc", "AbC", "abC"});
  auto program = cudf::experimental::regex_jit_program::create(
    R"((?i:(a)(?-i:B))(?i:(c)))", cudf::experimental::regex_operation::EXTRACT);
  auto result = cudf::experimental::extract(cudf::strings_column_view{input}, *program);
  cudf::test::strings_column_wrapper first({"A", "", ""}, {true, false, false});
  cudf::test::strings_column_wrapper second({"c", "", ""}, {true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view().column(0), first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view().column(1), second);
}

TEST_F(RegexJitProgramTest, ScopedClassesCount)
{
  cudf::test::strings_column_wrapper input({"ééaa aa", "١٢12", "aa"});
  auto const view = cudf::strings_column_view{input};
  auto check      = [&](char const* pattern,
                   cudf::strings::regex_flags flags,
                   std::initializer_list<cudf::size_type> values) {
    auto program = cudf::experimental::regex_jit_program::create(
      pattern, cudf::experimental::regex_operation::COUNT, {}, flags);
    auto result = cudf::experimental::count_re(view, *program);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(values.begin(), values.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  };
  check(R"((?a:\w{2}))", cudf::strings::regex_flags::DEFAULT, {2, 1, 1});
  check(R"((?u:\w{2}))", cudf::strings::regex_flags::ASCII, {3, 2, 1});
  check(R"((?a:\d{2}))", cudf::strings::regex_flags::DEFAULT, {0, 1, 0});
}

TEST_F(RegexJitProgramTest, InvalidInlineFlags)
{
  for (auto pattern : {"(?i-i:a)",
                       "(?au:a)",
                       "(?-a:a)",
                       "(?i-:a)",
                       "(?q:a)",
                       "(?i)*a",
                       "a(?i)(?s)+b",
                       "(?i",
                       "(?-i)a",
                       "(?i--m:a)"}) {
    SCOPED_TRACE(pattern);
    EXPECT_THROW(cudf::experimental::regex_jit_program::create(
                   pattern, cudf::experimental::regex_operation::CONTAINS),
                 cudf::logic_error);
  }
}

TEST_F(RegexJitProgramTest, MetadataAndOperationValidation)
{
  auto program =
    cudf::experimental::regex_jit_program::create("(a)",
                                                  cudf::experimental::regex_operation::COUNT,
                                                  {},
                                                  cudf::strings::regex_flags::MULTILINE,
                                                  cudf::strings::capture_groups::NON_CAPTURE);

  EXPECT_EQ(program->pattern(), "(a)");
  EXPECT_EQ(program->flags(), cudf::strings::regex_flags::MULTILINE);
  EXPECT_EQ(program->capture(), cudf::strings::capture_groups::NON_CAPTURE);
  EXPECT_EQ(program->operation(), cudf::experimental::regex_operation::COUNT);
  EXPECT_EQ(program->groups_count(), 1);

  auto input = cudf::test::strings_column_wrapper({"a"});
  EXPECT_THROW(cudf::experimental::contains_re(cudf::strings_column_view{input}, *program),
               cudf::logic_error);
}

TEST_F(RegexJitProgramTest, NullableAlternationDoesNotJoinSuccessiveBranches)
{
  auto input = cudf::test::strings_column_wrapper({"abcdefgh"});
  auto prog  = cudf::experimental::regex_jit_program::create(
    "a(bc|de|fg|)h", cudf::experimental::regex_operation::CONTAINS);
  auto results = cudf::experimental::contains_re(cudf::strings_column_view{input}, *prog);
  cudf::test::fixed_width_column_wrapper<bool> expected({false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(RegexJitProgramTest, Lookahead)
{
  auto input = cudf::test::strings_column_wrapper({"ab", "ac", "a", "", "aab", "cab"});
  auto check = [&](std::string_view pattern, std::initializer_list<bool> values) {
    auto program = cudf::experimental::regex_jit_program::create(
      pattern, cudf::experimental::regex_operation::CONTAINS);
    auto actual = cudf::experimental::contains_re(cudf::strings_column_view{input}, *program);
    cudf::test::fixed_width_column_wrapper<bool> expected(values.begin(), values.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*actual, expected);
  };

  EXPECT_NO_THROW(check(R"(a(?=b)b)", {true, false, false, false, true, true}));
  EXPECT_NO_THROW(check(R"(a(?!b))", {false, true, true, false, true, false}));
}

TEST_F(RegexJitProgramTest, Lookbehind)
{
  auto input = cudf::test::strings_column_wrapper({"ab", "cb", "b", "", "aab", "abc"});
  auto check = [&](std::string_view pattern, std::initializer_list<bool> values) {
    auto program = cudf::experimental::regex_jit_program::create(
      pattern, cudf::experimental::regex_operation::CONTAINS);
    auto actual = cudf::experimental::contains_re(cudf::strings_column_view{input}, *program);
    cudf::test::fixed_width_column_wrapper<bool> expected(values.begin(), values.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*actual, expected);
  };

  EXPECT_NO_THROW(check(R"(a(?<=a)b)", {true, false, false, false, true, true}));
  EXPECT_NO_THROW(check(R"((?<!a)b)", {false, true, true, false, false, false}));
}

TEST_F(RegexJitProgramTest, PrioritySensitivePatternsPreserveMatchSpan)
{
  auto verify_matches = [](std::string_view pattern,
                           char const* input_value,
                           cudf::test::lists_column_wrapper<cudf::string_view> const& expected,
                           cudf::strings::regex_flags flags = cudf::strings::regex_flags::DEFAULT) {
    auto input = cudf::test::strings_column_wrapper({input_value});
    auto prog =
      cudf::experimental::regex_jit_program::create(pattern,
                                                    cudf::experimental::regex_operation::FINDALL,
                                                    {},
                                                    flags,
                                                    cudf::strings::capture_groups::NON_CAPTURE);
    auto matches = cudf::experimental::findall(cudf::strings_column_view{input}, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(matches->view(), expected);
  };

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  verify_matches(R"(\ba*(ab)*)", "aab", LCW({{"aa", ""}}));
  verify_matches(R"(\ba{0,1}(ab){0,1})", "ab", LCW({{"a", ""}}));
  verify_matches(R"((?:a|\n)*?$)", "a\n", LCW({{"a", "", ""}}));
  verify_matches(R"((?:a|\n)*$)", "a\n", LCW({{"a\n", ""}}));
  verify_matches(R"((?:a|a\nb)$)", "a\nb", LCW({{"a"}}), cudf::strings::regex_flags::MULTILINE);
}

TEST_F(RegexJitProgramTest, IgnoreCaseCharacterClassRanges)
{
  // The interpreter retains the old range semantics; these exercise the JIT correction.
  auto input = cudf::test::strings_column_wrapper(
    {"à", "ÿ", "Ÿ", "Ā", "ł", "W", "w", "Z", "z", "A", "a", "C", "c", "D", "!", "b", "z"});
  auto view = cudf::strings_column_view(input);

  auto check = [&](std::string_view pattern, std::initializer_list<bool> expected_values) {
    auto program =
      cudf::test::jit_regex_backend::create(cudf::experimental::regex_operation::CONTAINS,
                                            pattern,
                                            cudf::strings::regex_flags::IGNORECASE);
    auto results = cudf::test::jit_regex_backend::contains_re(view, *program);
    cudf::test::fixed_width_column_wrapper<bool> expected(expected_values.begin(),
                                                          expected_values.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  };

  check("[À-ÿ]", {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  check("[W-c]", {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1});
  check("[!-a]", {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  check(R"([\x41-\x5A])", {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1});
}

TEST_F(RegexJitProgramTest, AsciiWhitespaceBoundaries)
{
  // The interpreter retains the old range semantics; these exercise the JIT correction.
  auto input = cudf::test::strings_column_wrapper({"\x1b", " ", "\t", "\n", "\x0e", "\x1f"});
  auto view  = cudf::strings_column_view(input);
  auto flags = cudf::strings::regex_flags::ASCII;

  auto whitespace = cudf::test::jit_regex_backend::create(
    cudf::experimental::regex_operation::COUNT, R"(\s)", flags);
  auto results = cudf::test::jit_regex_backend::count_re(view, *whitespace);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> whitespace_expected({0, 1, 1, 1, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, whitespace_expected);

  auto non_whitespace = cudf::test::jit_regex_backend::create(
    cudf::experimental::regex_operation::COUNT, R"(\S)", flags);
  results = cudf::test::jit_regex_backend::count_re(view, *non_whitespace);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> non_whitespace_expected(
    {1, 0, 0, 0, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, non_whitespace_expected);
}

TEST_F(RegexJitProgramTest, LookbehindCapturesOutsideMatch)
{
  auto input   = cudf::test::strings_column_wrapper({"abc", "aabc", "bc", "a", "", "abc"},
                                                    {true, true, true, true, true, false});
  auto program = cudf::experimental::regex_jit_program::create(
    R"((?<=([ab]+)([bc]+))$)", cudf::experimental::regex_operation::EXTRACT);
  auto actual = cudf::experimental::extract(cudf::strings_column_view{input}, *program);
  cudf::test::strings_column_wrapper first({"a", "aa", "b", "", "", ""},
                                           {true, true, true, false, false, false});
  cudf::test::strings_column_wrapper second({"bc", "bc", "c", "", "", ""},
                                            {true, true, true, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().column(0), first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().column(1), second);
  auto single = cudf::experimental::regex_jit_program::create(
    R"((?<=([ab]+)([bc]+))$)", cudf::experimental::regex_operation::EXTRACT_SINGLE, {.group = 1});
  auto selected = cudf::experimental::extract_single(cudf::strings_column_view{input}, *single, 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*selected, second);
}

TEST_F(RegexJitProgramTest, LookaheadCapturesOutsideMatch)
{
  auto input =
    cudf::test::strings_column_wrapper({"aaa", "ab", "", "aaa"}, {true, true, true, false});
  auto program = cudf::experimental::regex_jit_program::create(
    R"((?=(a+))a)", cudf::experimental::regex_operation::EXTRACT);
  auto actual = cudf::experimental::extract(cudf::strings_column_view{input}, *program);
  cudf::test::strings_column_wrapper expected({"aaa", "a", "", ""}, {true, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().column(0), expected);
}

TEST_F(RegexJitProgramTest, NegativeLookaheadDiscardsCaptures)
{
  auto input   = cudf::test::strings_column_wrapper({"abx", "abc", "a"});
  auto program = cudf::experimental::regex_jit_program::create(
    R"(a(?!(b)c))", cudf::experimental::regex_operation::EXTRACT);
  auto actual = cudf::experimental::extract(cudf::strings_column_view{input}, *program);
  cudf::test::strings_column_wrapper expected({"", "", ""}, {false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().column(0), expected);
}

TEST_F(RegexJitProgramTest, LookaroundMaterializationAndCaching)
{
  using backend   = cudf::test::jit_regex_backend;
  using operation = cudf::experimental::regex_operation;
  using policy    = cudf::experimental::regex_jit_span_cache_policy;
  auto input      = cudf::test::strings_column_wrapper({"ab ab", "ac", ""});
  auto view       = cudf::strings_column_view{input};
  for (auto cache : {policy::OFF, policy::FORCE}) {
    auto program = backend::create(operation::FINDALL, R"(a(?=b))", {.span_cache_policy = cache});
    auto actual  = backend::findall(view, *program);
    using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
    LCW expected({{"a", "a"}, {}, {}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*actual, expected);

    program = backend::create(
      operation::REPLACE, R"(a(?=b))", {.replacement = "x", .span_cache_policy = cache});
    actual = backend::replace_re(view, *program, cudf::string_scalar{"x"});
    cudf::test::strings_column_wrapper replaced({"xb xb", "ac", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*actual, replaced);
    program = backend::create(operation::REPLACE_WITH_BACKREFS,
                              R"((?=(ab))a)",
                              {.replacement = R"(\1)", .span_cache_policy = cache});
    actual  = backend::replace_with_backrefs(view, *program, R"(\1)");
    cudf::test::strings_column_wrapper captured_replacement({"abb abb", "ac", ""});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*actual, captured_replacement);

    program = backend::create(operation::SPLIT_RECORD, R"((?<=a)b)", {.span_cache_policy = cache});
    actual  = backend::split_record_re(view, *program);
    LCW fields({{"a", " a", ""}, {"ac"}, {""}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*actual, fields);
  }
}

TEST_F(RegexJitProgramTest, NullableLookaroundRepeatCaptures)
{
  auto input = cudf::test::strings_column_wrapper({"ab", "a", "ba", "aab", "aaab", ""});
  cudf::test::strings_column_wrapper expected({"", "", "", "", "", ""},
                                              {true, true, true, true, true, false});
  for (auto pattern : {R"(((?=a))*a)", R"(((?=a)a?)*a)"}) {
    auto program = cudf::experimental::regex_jit_program::create(
      pattern, cudf::experimental::regex_operation::EXTRACT);
    auto actual = cudf::experimental::extract(cudf::strings_column_view{input}, *program);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().column(0), expected);
  }
}

TEST_F(RegexJitProgramTest, LookbehindNullableRepeatCaptures)
{
  auto input   = cudf::test::strings_column_wrapper({"ab", "aab", "aaab", "b", "😀ab"});
  auto program = cudf::experimental::regex_jit_program::create(
    R"((?<=((?=a)a?)*)b)", cudf::experimental::regex_operation::EXTRACT);
  auto actual = cudf::experimental::extract(cudf::strings_column_view{input}, *program);
  cudf::test::strings_column_wrapper expected({"", "", "", "", ""},
                                              {true, true, true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().column(0), expected);
}

TEST_F(RegexJitProgramTest, LookbehindExternalWorkspace)
{
  std::vector<std::string> values{std::string(400, 'a') + "b", "b", ""};
  auto input   = cudf::test::strings_column_wrapper(values.begin(), values.end());
  auto program = cudf::experimental::regex_jit_program::create(
    R"((?<=((a|aa){200}))b)", cudf::experimental::regex_operation::EXTRACT);
  auto actual = cudf::experimental::extract(cudf::strings_column_view{input}, *program);
  std::vector<std::string> captured{std::string(200, 'a'), "", ""};
  cudf::test::strings_column_wrapper first(
    captured.begin(), captured.end(), cudf::test::iterators::nulls_at({1, 2}));
  cudf::test::strings_column_wrapper second({"a", "", ""}, {true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().column(0), first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().column(1), second);
}

TEST_F(RegexJitProgramTest, LookaroundEmptyMatches)
{
  using backend   = cudf::test::jit_regex_backend;
  using operation = cudf::experimental::regex_operation;
  using policy    = cudf::experimental::regex_jit_span_cache_policy;
  using LCW       = cudf::test::lists_column_wrapper<cudf::string_view>;
  auto input      = cudf::test::strings_column_wrapper({"aaa", "", "éé"});
  auto view       = cudf::strings_column_view{input};
  auto check      = [&](std::string_view pattern,
                   std::initializer_list<cudf::size_type> counts,
                   LCW const& expected) {
    auto program = backend::create(operation::COUNT, pattern);
    auto actual  = backend::count_re(view, *program);
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_counts(counts.begin(),
                                                                            counts.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*actual, expected_counts);
    for (auto cache : {policy::OFF, policy::FORCE}) {
      program = backend::create(operation::FINDALL, pattern, {.span_cache_policy = cache});
      actual  = backend::findall(view, *program);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*actual, expected);
    }
  };
  check(R"((?=a))", {3, 0, 0}, LCW({{"", "", ""}, {}, {}}));
  check(R"((?<=a))", {3, 0, 0}, LCW({{"", "", ""}, {}, {}}));
  check(R"((?!a))", {1, 1, 3}, LCW({{""}, {""}, {"", "", ""}}));
  check(R"((?=))", {4, 1, 3}, LCW({{"", "", "", ""}, {""}, {"", "", ""}}));
  check(R"((?=é))", {0, 0, 2}, LCW({{}, {}, {"", ""}}));
}

TEST_F(RegexJitProgramTest, LookbehindMalformedUtf8)
{
  std::vector<std::string> values{
    "\x80\x80"
    "a",
    "\xc3\x80\x80"
    "a",
    "\xc3"
    "a",
    "\xc2\x80"
    "a",
    "\xe0\x80"
    "a",
    "\xf0\x80\x80"
    "a"};
  auto input   = cudf::test::strings_column_wrapper(values.begin(), values.end());
  auto program = cudf::experimental::regex_jit_program::create(
    R"((?<=(.))a)", cudf::experimental::regex_operation::EXTRACT);
  auto actual = cudf::experimental::extract(cudf::strings_column_view{input}, *program);
  cudf::test::strings_column_wrapper expected({"\x80", "\x80", "\xc3", "\xc2\x80", "\x80", "\x80"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().column(0), expected);
}

TEST_F(RegexJitProgramTest, UpstreamLookaroundCaptures)
{
  struct test_case {
    char const* pattern;
    char const* text;
    bool matches;
    std::vector<std::pair<std::string, bool>> captures;
  };
  std::vector<test_case> const cases{
    {R"((a(?=\s[^a])))", "a b", true, {{"a", true}}},
    {R"((a)(?=\s(abc|a)))", "a a", true, {{"a", true}, {"a", true}}},
    {R"((a)(?!\s(abc|a)))", "a b", true, {{"a", true}, {"", false}}},
    {R"((a)b(?<=b)(c))", "abc", true, {{"a", true}, {"c", true}}},
    {R"((a)b(?<=c)(c))", "abc", false, {{"", false}, {"", false}}},
    {R"((?<=(\w){3})def)", "abcdef", true, {{"a", true}}},
    {R"((?<=(bc)|(cd)).)", "abcdef", true, {{"bc", true}, {"", false}}},
    {R"((?<=a(?=([bc]{2}(?<!a{2}))d)\w{3})\w\w)", "abcdef", true, {{"bc", true}}},
    {R"(.*(?<=(xx|xxx))(.*))", "xxabcd", true, {{"xx", true}, {"abcd", true}}},
    {R"((?<=((?:b\d{2})+))c)", "ab12b23b34c", true, {{"b12b23b34", true}}},
  };
  for (auto const& test : cases) {
    SCOPED_TRACE(test.pattern);
    auto input   = cudf::test::strings_column_wrapper({test.text});
    auto view    = cudf::strings_column_view{input};
    auto program = cudf::experimental::regex_jit_program::create(
      test.pattern, cudf::experimental::regex_operation::CONTAINS);
    auto contains = cudf::experimental::contains_re(view, *program);
    cudf::test::fixed_width_column_wrapper<bool> matched({test.matches});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*contains, matched);
    program = cudf::experimental::regex_jit_program::create(
      test.pattern, cudf::experimental::regex_operation::EXTRACT);
    auto extracted = cudf::experimental::extract(view, *program);
    ASSERT_EQ(extracted->num_columns(), test.captures.size());
    for (std::size_t group = 0; group < test.captures.size(); ++group) {
      auto const& [value, valid] = test.captures[group];
      cudf::test::strings_column_wrapper expected({value}, {valid});
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(extracted->view().column(group), expected);
    }
  }
}

TEST_F(RegexJitProgramTest, FindallOffsetsAndSpanCache)
{
  std::array valids{true, true, true, true, true, false, true, true};
  cudf::test::strings_column_wrapper input(
    {"3-A", "4-May 5-Day 6-Hay", "12-Dec-2021-Jan", "Feb-March", "4 ABC", "", "", "25-9000-Hal"},
    valids.data());
  auto input64    = std::unique_ptr<cudf::column>{};
  auto input_view = static_cast<cudf::column_view>(input);
  if (cudf::strings::is_large_strings_enabled()) {
    input64    = make_int64_offsets_column(input);
    input_view = input64->view();
    EXPECT_EQ(cudf::strings_column_view{input_view}.offsets().type().id(), cudf::type_id::INT64);
  }
  auto sv = cudf::strings_column_view{input_view};

  auto pattern = std::string("\\d+-\\w+");

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected(
    {{"3-A"}, {"4-May", "5-Day", "6-Hay"}, {"12-Dec", "2021-Jan"}, {}, {}, {}, {}, {"25-9000"}},
    valids.data());
  auto prog = cudf::test::jit_regex_backend::create(
    cudf::experimental::regex_operation::FINDALL,
    pattern,
    {.span_cache_policy = cudf::experimental::regex_jit_span_cache_policy::FORCE});
  auto results = cudf::test::jit_regex_backend::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(RegexJitProgramTest, FindallNoMatchesOffsetsAndSpanCache)
{
  cudf::test::strings_column_wrapper input({"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n"});
  auto input64    = std::unique_ptr<cudf::column>{};
  auto input_view = static_cast<cudf::column_view>(input);
  if (cudf::strings::is_large_strings_enabled()) {
    input64    = make_int64_offsets_column(input);
    input_view = input64->view();
    EXPECT_EQ(cudf::strings_column_view{input_view}.offsets().type().id(), cudf::type_id::INT64);
  }
  auto sv = cudf::strings_column_view{input_view};

  auto pattern = std::string("^zzz$");
  using LCW    = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({{}, {}, {}, {}, {}});
  auto prog = cudf::test::jit_regex_backend::create(
    cudf::experimental::regex_operation::FINDALL,
    pattern,
    {.span_cache_policy = cudf::experimental::regex_jit_span_cache_policy::OFF});
  auto results = cudf::test::jit_regex_backend::findall(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}
