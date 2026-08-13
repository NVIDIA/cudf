/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "special_chars.h"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/strings/extract.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsExtractTests : public cudf::test::BaseFixture {};

TEST_F(StringsExtractTests, ExtractTest)
{
  std::vector<char const*> h_strings{
    "First Last", "Joe Schmoe", "John Smith", "Jane Smith", "Beyonce", "Sting", nullptr, ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<char const*> h_expecteds{"First",
                                       "Joe",
                                       "John",
                                       "Jane",
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       "Last",
                                       "Schmoe",
                                       "Smith",
                                       "Smith",
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr};

  std::string pattern = "(\\w+) (\\w+)";

  cudf::test::strings_column_wrapper expected1(
    h_expecteds.data(),
    h_expecteds.data() + h_strings.size(),
    thrust::make_transform_iterator(h_expecteds.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::strings_column_wrapper expected2(
    h_expecteds.data() + h_strings.size(),
    h_expecteds.data() + h_expecteds.size(),
    thrust::make_transform_iterator(h_expecteds.data() + h_strings.size(),
                                    [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(expected1.release());
  columns.push_back(expected2.release());
  cudf::table expected(std::move(columns));

  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::extract(strings_view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);
}

TEST_F(StringsExtractTests, ExtractDomainTest)
{
  cudf::test::strings_column_wrapper strings({"http://www.google.com",
                                              "gmail.com",
                                              "github.com",
                                              "https://pandas.pydata.org",
                                              "http://www.worldbank.org.kg/",
                                              "waiterrant.blogspot.com",
                                              "http://forums.news.cnn.com.ac/",
                                              "http://forums.news.cnn.ac/",
                                              "ftp://b.cnn.com/",
                                              "a.news.uk",
                                              "a.news.co.uk",
                                              "https://a.news.co.uk",
                                              "107-193-100-2.lightspeed.cicril.sbcglobal.net",
                                              "a23-44-13-2.deploy.static.akamaitechnologies.com"});
  auto strings_view = cudf::strings_column_view(strings);

  std::string pattern = R"(([\w]+[\.].*[^/]|[\-\w]+[\.].*[^/]))";

  cudf::test::strings_column_wrapper expected1({
    "www.google.com",
    "gmail.com",
    "github.com",
    "pandas.pydata.org",
    "www.worldbank.org.kg",
    "waiterrant.blogspot.com",
    "forums.news.cnn.com.ac",
    "forums.news.cnn.ac",
    "b.cnn.com",
    "a.news.uk",
    "a.news.co.uk",
    "a.news.co.uk",
    "107-193-100-2.lightspeed.cicril.sbcglobal.net",
    "a23-44-13-2.deploy.static.akamaitechnologies.com",
  });
  cudf::table_view expected{{expected1}};

  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::extract(strings_view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);
}

TEST_F(StringsExtractTests, ExtractEventTest)
{
  std::vector<std::string> patterns({"(^[0-9]+\\.?[0-9]*),",
                                     R"(search_name="([0-9A-Za-z\s\-\(\)]+))",
                                     R"(message.ip="([\w\.]+))",
                                     R"(message.hostname="([\w\.]+))",
                                     R"(message.user_name="([\w\.\@]+))",
                                     R"(message\.description="([\w\.\s]+))"});

  cudf::test::strings_column_wrapper strings(
    {"15162388.26, search_name=\"Test Search Name\", orig_time=\"1516238826\", "
     "info_max_time=\"1566346500.000000000\", info_min_time=\"1566345300.000000000\", "
     "info_search_time=\"1566305689.361160000\", message.description=\"Test Message Description\", "
     "message.hostname=\"msg.test.hostname\", message.ip=\"100.100.100.123\", "
     "message.user_name=\"user@test.com\", severity=\"info\", urgency=\"medium\"'"});
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<std::string> expecteds({"15162388.26",
                                      "Test Search Name",
                                      "100.100.100.123",
                                      "msg.test.hostname",
                                      "user@test.com",
                                      "Test Message Description"});

  for (std::size_t idx = 0; idx < patterns.size(); ++idx) {
    auto pattern = patterns[idx];
    cudf::test::strings_column_wrapper expected({expecteds[idx]});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::extract(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view().column(0), expected);
  }
}

TEST_F(StringsExtractTests, MultiLine)
{
  auto input = cudf::test::strings_column_wrapper(
    {"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n", "abé\nabc\n"});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("(^[a-c]+$)");
  cudf::test::strings_column_wrapper expected_multiline({"abc", "abc", "abc", "", "abc", "abc"},
                                                        {true, true, true, false, true, true});
  auto expected = cudf::table_view{{expected_multiline}};
  auto prog = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::MULTILINE);
  auto results = cudf::strings::extract(view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);

  pattern = std::string("^([a-c]+)$");
  cudf::test::strings_column_wrapper expected_default({"", "", "abc", "", "abc", ""},
                                                      {false, false, true, false, true, false});
  expected = cudf::table_view{{expected_default}};
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::extract(view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);
}

TEST_F(StringsExtractTests, DotAll)
{
  auto input = cudf::test::strings_column_wrapper({"abc\nfa\nef", "fff\nabbc\nfff", "abcdef", ""});
  auto view  = cudf::strings_column_view(input);

  auto pattern = std::string("(a.*f)");
  cudf::test::strings_column_wrapper expected_dotall({"abc\nfa\nef", "abbc\nfff", "abcdef", ""},
                                                     {true, true, true, false});
  auto expected = cudf::table_view{{expected_dotall}};
  auto prog     = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DOTALL);
  auto results  = cudf::strings::extract(view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);

  cudf::test::strings_column_wrapper expected_default({"", "", "abcdef", ""},
                                                      {false, false, true, false});
  expected = cudf::table_view{{expected_default}};
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::extract(view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);
}

TEST_F(StringsExtractTests, SpecialNewLines)
{
  auto input = cudf::test::strings_column_wrapper({"zzé" NEXT_LINE "qqq" LINE_SEPARATOR "zzé",
                                                   "qqq" LINE_SEPARATOR "zzé\rlll",
                                                   "zzé",
                                                   "",
                                                   "zzé" NEXT_LINE,
                                                   "abc" PARAGRAPH_SEPARATOR "zzé\n"});
  auto view  = cudf::strings_column_view(input);

  auto prog =
    cudf::strings::regex_program::create("(^zzé$)", cudf::strings::regex_flags::EXT_NEWLINE);
  auto results = cudf::strings::extract(view, *prog);
  auto expected =
    cudf::test::strings_column_wrapper({"", "", "zzé", "", "zzé", ""}, {0, 0, 1, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);

  auto both_flags = static_cast<cudf::strings::regex_flags>(
    cudf::strings::regex_flags::EXT_NEWLINE | cudf::strings::regex_flags::MULTILINE);
  auto prog_ml = cudf::strings::regex_program::create("^(zzé)$", both_flags);
  results      = cudf::strings::extract(view, *prog_ml);
  expected =
    cudf::test::strings_column_wrapper({"zzé", "zzé", "zzé", "", "zzé", "zzé"}, {1, 1, 1, 0, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);

  prog     = cudf::strings::regex_program::create("q(q.*l)l");
  expected = cudf::test::strings_column_wrapper({"", "qq" LINE_SEPARATOR "zzé\rll", "", "", "", ""},
                                                {0, 1, 0, 0, 0, 0});
  results  = cudf::strings::extract(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
  // expect no matches here since the newline(s) interrupts the pattern
  prog = cudf::strings::regex_program::create("q(q.*l)l", cudf::strings::regex_flags::EXT_NEWLINE);
  expected = cudf::test::strings_column_wrapper({"", "", "", "", "", ""}, {0, 0, 0, 0, 0, 0});
  results  = cudf::strings::extract(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TEST_F(StringsExtractTests, NestedQuantifier)
{
  auto input   = cudf::test::strings_column_wrapper({"TEST12 1111 2222 3333 4444 5555",
                                                     "0000 AAAA 9999 BBBB 8888",
                                                     "7777 6666 4444 3333",
                                                     "12345 3333 4444 1111 ABCD"});
  auto sv      = cudf::strings_column_view(input);
  auto pattern = std::string(R"((\d{4}\s){4})");
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::extract(sv, *prog);
  // fixed quantifier on capture group only honors the last group
  auto expected = cudf::test::strings_column_wrapper({"4444 ", "", "", "1111 "}, {1, 0, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view().column(0), expected);
}

TEST_F(StringsExtractTests, EmptyExtractTest)
{
  std::vector<char const*> h_strings{nullptr, "AAA", "AAA_A", "AAA_AAA_", "A__", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto pattern = std::string("([^_]*)\\Z");

  std::vector<char const*> h_expected{nullptr, "AAA", "A", "", "", ""};
  cudf::test::strings_column_wrapper expected(
    h_expected.data(),
    h_expected.data() + h_strings.size(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(expected.release());
  cudf::table table_expected(std::move(columns));
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::extract(strings_view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, table_expected);
}

TEST_F(StringsExtractTests, NonParticipatingGroup)
{
  auto input = cudf::test::strings_column_wrapper({"A1", "B2", "C"});
  auto sv    = cudf::strings_column_view(input);

  // the optional group does not participate in the match for "C" and must
  // result in null, not an empty string
  auto prog    = cudf::strings::regex_program::create("(\\D)(\\d)?");
  auto results = cudf::strings::extract(sv, *prog);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::test::strings_column_wrapper({"A", "B", "C"}).release());
  columns.push_back(cudf::test::strings_column_wrapper({"1", "2", ""}, {1, 1, 0}).release());
  auto expected = cudf::table(std::move(columns));
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);

  // a group that participates with an empty match remains an empty string
  auto prog2    = cudf::strings::regex_program::create("(\\D)(\\d*)");
  auto results2 = cudf::strings::extract(sv, *prog2);

  std::vector<std::unique_ptr<cudf::column>> columns2;
  columns2.push_back(cudf::test::strings_column_wrapper({"A", "B", "C"}).release());
  columns2.push_back(cudf::test::strings_column_wrapper({"1", "2", ""}).release());
  auto expected2 = cudf::table(std::move(columns2));
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results2, expected2);
}

TEST_F(StringsExtractTests, ExtractAllTest)
{
  std::vector<char const*> h_input(
    {"123 banana 7 eleven", "41 apple", "6 péar 0 pair", nullptr, "", "bees", "4 paré"});
  auto validity =
    thrust::make_transform_iterator(h_input.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper input(h_input.begin(), h_input.end(), validity);
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("(\\d+) (\\w+)");

  std::array valids{true, true, true, false, false, false, true};
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"123", "banana", "7", "eleven"},
                LCW{"41", "apple"},
                LCW{"6", "péar", "0", "pair"},
                LCW{},
                LCW{},
                LCW{},
                LCW{"4", "paré"}},
               valids.data());
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::extract_all_record(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsExtractTests, ExtractSingle)
{
  auto input = cudf::test::strings_column_wrapper(
    {"123 banana 7 eleven", "41 apple", "6 péar 0 pair", "", "", "bees", "4 paré"},
    {1, 1, 1, 0, 1, 1, 1});
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("(\\d+) (\\w+)");
  auto prog    = cudf::strings::regex_program::create(pattern);

  auto results  = cudf::strings::extract_single(sv, *prog, 1);
  auto expected = cudf::test::strings_column_wrapper(
    {"banana", "apple", "péar", "", "", "", "paré"}, {1, 1, 1, 0, 0, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results = cudf::strings::extract_single(sv, *prog, 0);
  expected =
    cudf::test::strings_column_wrapper({"123", "41", "6", "", "", "", "4"}, {1, 1, 1, 0, 0, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  EXPECT_THROW(cudf::strings::extract_single(sv, *prog, 2), std::invalid_argument);
}

TEST_F(StringsExtractTests, Errors)
{
  cudf::test::strings_column_wrapper input({"this column intentionally left blank"});
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("\\w+");
  auto prog    = cudf::strings::regex_program::create(pattern);

  EXPECT_THROW(cudf::strings::extract(sv, *prog), cudf::logic_error);
  EXPECT_THROW(cudf::strings::extract_all_record(sv, *prog), cudf::logic_error);
}

TEST_F(StringsExtractTests, EmptyInput)
{
  auto const input   = cudf::test::strings_column_wrapper();
  auto const sv      = cudf::strings_column_view(input);
  auto const pattern = std::string("(\\w+)");
  auto const prog    = cudf::strings::regex_program::create(pattern);

  auto rt = cudf::strings::extract(sv, *prog);
  EXPECT_EQ(1, rt->num_columns());
  EXPECT_EQ(0, rt->num_rows());

  auto rl = cudf::strings::extract_all_record(sv, *prog);
  EXPECT_EQ(0, rl->size());

  auto rs = cudf::strings::extract_single(sv, *prog, 1);
  EXPECT_EQ(0, rs->size());
}

TEST_F(StringsExtractTests, MediumRegex)
{
  // This results in 95 regex instructions and falls in the 'medium' range.
  std::string medium_regex =
    "hello @abc @def (world) The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com";
  auto prog = cudf::strings::regex_program::create(medium_regex);

  std::vector<char const*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com thats all",
    "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234"
    "5678901234567890",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnop"
    "qrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::extract(strings_view, *prog);
  std::vector<char const*> h_expected{"world", nullptr, nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->get_column(0), expected);
}

TEST_F(StringsExtractTests, LargeRegex)
{
  // This results in 115 regex instructions and falls in the 'large' range.
  std::string large_regex =
    "hello @abc @def world The (quick) brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz";
  auto prog = cudf::strings::regex_program::create(large_regex);

  std::vector<char const*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz",
    "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234"
    "5678901234567890",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnop"
    "qrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::extract(strings_view, *prog);
  std::vector<char const*> h_expected{"quick", nullptr, nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->get_column(0), expected);
}

TEST_F(StringsExtractTests, CrlfLineAnchorExtNewline)
{
  // extract group 1 of ([a-z]+)$ at \r\n line ends; null where no match.
  // Verified vs OpenJDK 17 group(1) of first match.
  auto input = cudf::test::strings_column_wrapper({"abc\r\n",
                                                   "abc\n",
                                                   "abc\r",
                                                   "abc",
                                                   "a\r\nb",
                                                   "abc\r\n\r\n",
                                                   "",
                                                   "abc" NEXT_LINE,
                                                   "a\nb\r\nc",
                                                   "\r\n",
                                                   "\r\nabc",
                                                   "x\n\r",
                                                   "a\r\rb",
                                                   "a\n\nb"});
  auto view  = cudf::strings_column_view(input);
  auto prog =
    cudf::strings::regex_program::create("([a-z]+)$", cudf::strings::regex_flags::EXT_NEWLINE);
  auto results = cudf::strings::extract(view, *prog);

  std::vector<char const*> h_expected{"abc",
                                      "abc",
                                      "abc",
                                      "abc",
                                      "b",
                                      nullptr,
                                      nullptr,
                                      "abc",
                                      "c",
                                      nullptr,
                                      "abc",
                                      nullptr,
                                      "b",
                                      "b"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->get_column(0), expected);
}
