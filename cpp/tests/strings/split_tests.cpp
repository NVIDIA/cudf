/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/split/partition.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/split/split_re.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsSplitTest : public cudf::test::BaseFixture {};

TEST_F(StringsSplitTest, Split)
{
  std::vector<char const*> h_strings{
    "Héllo thesé", nullptr, "are some", "tést String", "", "no-delimiter"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);

  std::vector<char const*> h_expected1{"Héllo", nullptr, "are", "tést", "", "no-delimiter"};
  cudf::test::strings_column_wrapper expected1(
    h_expected1.begin(),
    h_expected1.end(),
    thrust::make_transform_iterator(h_expected1.begin(), [](auto str) { return str != nullptr; }));
  std::vector<char const*> h_expected2{"thesé", nullptr, "some", "String", nullptr, nullptr};
  cudf::test::strings_column_wrapper expected2(
    h_expected2.begin(),
    h_expected2.end(),
    thrust::make_transform_iterator(h_expected2.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::split(strings_view, cudf::string_scalar(" "));
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, SplitWithMax)
{
  cudf::test::strings_column_wrapper strings(
    {"Héllo::thesé::world", "are::some", "tést::String:", ":last::one", ":::", "x::::y"});
  cudf::strings_column_view strings_view(strings);

  cudf::test::strings_column_wrapper expected1({"Héllo", "are", "tést", ":last", "", "x"});
  cudf::test::strings_column_wrapper expected2(
    {"thesé::world", "some", "String:", "one", ":", "::y"});
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::split(strings_view, cudf::string_scalar("::"), 1);
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, SplitWhitespace)
{
  auto const input = cudf::test::strings_column_wrapper(
    {"Héllo thesé", "", "are\tsome", "tést\nString", "  ", " a  b ", "", " 123 "},
    {1, 0, 1, 1, 1, 1, 1, 1});
  auto const sv = cudf::strings_column_view(input);

  auto expected1 = cudf::test::strings_column_wrapper(
    {"Héllo", "", "are", "tést", "", "a", "", "123"}, {1, 0, 1, 1, 0, 1, 0, 1});
  auto expected2 = cudf::test::strings_column_wrapper(
    {"thesé", "", "some", "String", "", "b", "", ""}, {1, 0, 1, 1, 0, 1, 0, 0});
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto const results = cudf::strings::split(sv);
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, SplitWhitespaceWithMax)
{
  auto const input = cudf::test::strings_column_wrapper(
    {"a bc d", "a  bc  d", " ab cd e", "ab cd e ", " ab cd e ", " abc ", "", " "});
  auto const sv = cudf::strings_column_view(input);

  auto expected1 = cudf::test::strings_column_wrapper({"a", "a", "ab", "ab", "ab", "abc", "", ""},
                                                      {1, 1, 1, 1, 1, 1, 0, 0});
  auto expected2 = cudf::test::strings_column_wrapper(
    {"bc d", "bc  d", "cd e", "cd e ", "cd e ", "", "", ""}, {1, 1, 1, 1, 1, 0, 0, 0});
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto const results = cudf::strings::split(sv, cudf::string_scalar(""), 1);
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RSplit)
{
  std::vector<char const*> h_strings{
    "héllo", nullptr, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", " a b ", " a  bbb   c"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);

  std::vector<char const*> h_expected1{
    "héllo", nullptr, "a", "a", "", "ab", "", " a b ", " a  bbb   c"};
  cudf::test::strings_column_wrapper expected1(
    h_expected1.begin(),
    h_expected1.end(),
    thrust::make_transform_iterator(h_expected1.begin(), [](auto str) { return str != nullptr; }));
  std::vector<char const*> h_expected2{
    nullptr, nullptr, "bc", "", "ab", "cd", nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper expected2(
    h_expected2.begin(),
    h_expected2.end(),
    thrust::make_transform_iterator(h_expected2.begin(), [](auto str) { return str != nullptr; }));
  std::vector<char const*> h_expected3{
    nullptr, nullptr, "déf", "bc", "cd", "", nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper expected3(
    h_expected3.begin(),
    h_expected3.end(),
    thrust::make_transform_iterator(h_expected3.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::rsplit(strings_view, cudf::string_scalar("_"));
  EXPECT_TRUE(results->num_columns() == 3);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RSplitWithMax)
{
  cudf::test::strings_column_wrapper strings(
    {"Héllo::thesé::world", "are::some", "tést::String:", ":last::one", ":::", "x::::y"});
  cudf::strings_column_view strings_view(strings);

  cudf::test::strings_column_wrapper expected1(
    {"Héllo::thesé", "are", "tést", ":last", ":", "x::"});
  cudf::test::strings_column_wrapper expected2({"world", "some", "String:", "one", "", "y"});
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::rsplit(strings_view, cudf::string_scalar("::"), 1);
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RSplitWhitespace)
{
  std::vector<char const*> h_strings{"héllo", nullptr, "a_bc_déf", "", " a\tb ", " a\r bbb   c"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::strings_column_view strings_view(strings);
  std::vector<char const*> h_expected1{"héllo", nullptr, "a_bc_déf", nullptr, "a", "a"};
  cudf::test::strings_column_wrapper expected1(
    h_expected1.begin(),
    h_expected1.end(),
    thrust::make_transform_iterator(h_expected1.begin(), [](auto str) { return str != nullptr; }));
  std::vector<char const*> h_expected2{nullptr, nullptr, nullptr, nullptr, "b", "bbb"};
  cudf::test::strings_column_wrapper expected2(
    h_expected2.begin(),
    h_expected2.end(),
    thrust::make_transform_iterator(h_expected2.begin(), [](auto str) { return str != nullptr; }));
  std::vector<char const*> h_expected3{nullptr, nullptr, nullptr, nullptr, nullptr, "c"};
  cudf::test::strings_column_wrapper expected3(
    h_expected3.begin(),
    h_expected3.end(),
    thrust::make_transform_iterator(h_expected3.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto results = cudf::strings::rsplit(strings_view);
  EXPECT_TRUE(results->num_columns() == 3);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RSplitWhitespaceWithMax)
{
  auto const input = cudf::test::strings_column_wrapper(
    {"a bc d", "a  bc  d", " ab cd e", "ab cd e ", " ab cd e ", " abc ", "", " "});
  auto const sv = cudf::strings_column_view(input);

  auto expected1 = cudf::test::strings_column_wrapper(
    {"a bc", "a  bc", " ab cd", "ab cd", " ab cd", "abc", "", ""}, {1, 1, 1, 1, 1, 1, 0, 0});
  auto expected2 = cudf::test::strings_column_wrapper({"d", "d", "e", "e", "e", "", "", ""},
                                                      {1, 1, 1, 1, 1, 0, 0, 0});
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  auto const results = cudf::strings::rsplit(sv, cudf::string_scalar(""), 1);
  EXPECT_TRUE(results->num_columns() == 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, SplitRecord)
{
  auto const validity = cudf::test::iterators::null_at(1);
  auto const input    = cudf::test::strings_column_wrapper(
    {" Héllo thesé", "", "are some  ", "tést String", "", " 123 "}, validity);
  auto const sv = cudf::strings_column_view(input);

  auto const result = cudf::strings::split_record(sv, cudf::string_scalar(" "));
  using LCW         = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"", "Héllo", "thesé"},
                LCW{},
                LCW{"are", "some", "", ""},
                LCW{"tést", "String"},
                LCW{""},
                LCW{"", "123", ""}},
               validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, SplitRecordWithMaxSplit)
{
  auto const validity = cudf::test::iterators::null_at(1);
  auto const input    = cudf::test::strings_column_wrapper(
    {" Héllo thesé", "", "are some  ", "tést String", "", " 123 "}, validity);
  auto const sv = cudf::strings_column_view(input);

  auto const result = cudf::strings::split_record(sv, cudf::string_scalar(" "), 1);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"", "Héllo thesé"},
                LCW{},
                LCW{"are", "some  "},
                LCW{"tést", "String"},
                LCW{""},
                LCW{"", "123 "}},
               validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, SplitRecordWhitespace)
{
  auto const validity = cudf::test::iterators::null_at(1);
  auto const input    = cudf::test::strings_column_wrapper(
    {"   Héllo thesé", "", "are\tsome  ", "tést\nString", "  ", "", " 123 "}, validity);
  auto const sv = cudf::strings_column_view(input);

  auto result = cudf::strings::split_record(sv);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"Héllo", "thesé"},
                LCW{},
                LCW{"are", "some"},
                LCW{"tést", "String"},
                LCW{},
                LCW{},
                LCW{"123"}},
               validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, SplitRecordWhitespaceWithMaxSplit)
{
  auto const validity = cudf::test::iterators::null_at(1);
  auto const input    = cudf::test::strings_column_wrapper(
    {"   Héllo thesé  ", "", "are\tsome  ", "tést\nString", "  ", "", " 123 "}, validity);
  auto const sv = cudf::strings_column_view(input);

  auto const result = cudf::strings::split_record(sv, cudf::string_scalar(""), 1);
  using LCW         = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"Héllo", "thesé  "},
                LCW{},
                LCW{"are", "some  "},
                LCW{"tést", "String"},
                LCW{},
                LCW{},
                LCW{"123"}},
               validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, SplitAllEmpty)
{
  auto input     = cudf::test::strings_column_wrapper({"", "", "", ""});
  auto sv        = cudf::strings_column_view(input);
  auto empty     = cudf::string_scalar("");
  auto delimiter = cudf::string_scalar("s");

  auto result = cudf::strings::split(sv, delimiter);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view().column(0), input);
  result = cudf::strings::rsplit(sv, delimiter);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view().column(0), input);

  // whitespace hits a special case where nothing matches returns an all-null column
  auto expected = cudf::test::strings_column_wrapper({"", "", "", ""}, {0, 0, 0, 0});
  result        = cudf::strings::split(sv, empty);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view().column(0), expected);
  result = cudf::strings::rsplit(sv, empty);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view().column(0), expected);
}

TEST_F(StringsSplitTest, SplitRecordAllEmpty)
{
  auto input     = cudf::test::strings_column_wrapper({"", "", "", ""});
  auto sv        = cudf::strings_column_view(input);
  auto empty     = cudf::string_scalar("");
  auto delimiter = cudf::string_scalar("s");

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{""}, LCW{""}, LCW{""}, LCW{""}});
  LCW expected_empty({LCW{}, LCW{}, LCW{}, LCW{}});

  auto result = cudf::strings::split_record(sv, delimiter);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  result = cudf::strings::split_record(sv, empty);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected_empty);

  result = cudf::strings::rsplit_record(sv, delimiter);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  result = cudf::strings::rsplit_record(sv, empty);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected_empty);
}

TEST_F(StringsSplitTest, MultiByteDelimiters)
{
  // Overlapping delimiters
  auto input =
    cudf::test::strings_column_wrapper({"u::", "w:::x", "y::::z", "::a", ":::b", ":::c:::"});
  auto view = cudf::strings_column_view(input);
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  {
    auto result        = cudf::strings::split_record(view, cudf::string_scalar("::"));
    auto expected_left = LCW({LCW{"u", ""},
                              LCW{"w", ":x"},
                              LCW{"y", "", "z"},
                              LCW{"", "a"},
                              LCW{"", ":b"},
                              LCW{"", ":c", ":"}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected_left);
    result              = cudf::strings::rsplit_record(view, cudf::string_scalar("::"));
    auto expected_right = LCW({LCW{"u", ""},
                               LCW{"w:", "x"},
                               LCW{"y", "", "z"},
                               LCW{"", "a"},
                               LCW{":", "b"},
                               LCW{":", "c:", ""}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected_right);
  }
  {
    auto result = cudf::strings::split(view, cudf::string_scalar("::"));

    auto c0 = cudf::test::strings_column_wrapper({"u", "w", "y", "", "", ""});
    auto c1 = cudf::test::strings_column_wrapper({"", ":x", "", "a", ":b", ":c"});
    auto c2 = cudf::test::strings_column_wrapper({"", "", "z", "", "", ":"},
                                                 {false, false, true, false, false, true});
    std::vector<std::unique_ptr<cudf::column>> expected_columns;
    expected_columns.push_back(c0.release());
    expected_columns.push_back(c1.release());
    expected_columns.push_back(c2.release());
    auto expected_left = std::make_unique<cudf::table>(std::move(expected_columns));
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result, *expected_left);

    result = cudf::strings::rsplit(view, cudf::string_scalar("::"));

    c0 = cudf::test::strings_column_wrapper({"u", "w:", "y", "", ":", ":"});
    c1 = cudf::test::strings_column_wrapper({"", "x", "", "a", "b", "c:"});
    c2 = cudf::test::strings_column_wrapper({"", "", "z", "", "", ""},
                                            {false, false, true, false, false, true});
    expected_columns.push_back(c0.release());
    expected_columns.push_back(c1.release());
    expected_columns.push_back(c2.release());
    auto expected_right = std::make_unique<cudf::table>(std::move(expected_columns));
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result, *expected_right);
  }

  // Delimiters that span across adjacent strings
  input = cudf::test::strings_column_wrapper({"{a=1}:{b=2}:", "{c=3}", ":{}:{}"});
  view  = cudf::strings_column_view(input);
  {
    auto result   = cudf::strings::split_record(view, cudf::string_scalar("}:{"));
    auto expected = LCW({LCW{"{a=1", "b=2}:"}, LCW{"{c=3}"}, LCW{":{", "}"}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
    result = cudf::strings::rsplit_record(view, cudf::string_scalar("}:{"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
  {
    auto result = cudf::strings::split(view, cudf::string_scalar("}:{"));

    auto c0 = cudf::test::strings_column_wrapper({"{a=1", "{c=3}", ":{"});
    auto c1 = cudf::test::strings_column_wrapper({"b=2}:", "", "}"}, {true, false, true});
    std::vector<std::unique_ptr<cudf::column>> expected_columns;
    expected_columns.push_back(c0.release());
    expected_columns.push_back(c1.release());
    auto expected = std::make_unique<cudf::table>(std::move(expected_columns));
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result, *expected);

    result = cudf::strings::rsplit(view, cudf::string_scalar("}:{"));
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result, *expected);
  }
}

TEST_F(StringsSplitTest, SplitRegex)
{
  std::vector<char const*> h_strings{" Héllo thesé", nullptr, "are some  ", "tést String", ""};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper input(h_strings.begin(), h_strings.end(), validity);
  auto sv = cudf::strings_column_view(input);

  {
    auto pattern = std::string("\\s+");

    cudf::test::strings_column_wrapper col0({"", "", "are", "tést", ""}, validity);
    cudf::test::strings_column_wrapper col1({"Héllo", "", "some", "String", ""},
                                            {true, false, true, true, false});
    cudf::test::strings_column_wrapper col2({"thesé", "", "", "", ""},
                                            {true, false, true, false, false});
    auto expected = cudf::table_view({col0, col1, col2});
    auto prog     = cudf::strings::regex_program::create(pattern);
    auto result   = cudf::strings::split_re(sv, *prog);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected);

    // rsplit == split when using default parameters
    result = cudf::strings::rsplit_re(sv, *prog);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected);
  }

  {
    auto pattern = std::string("[eé]");

    cudf::test::strings_column_wrapper col0({" H", "", "ar", "t", ""}, validity);
    cudf::test::strings_column_wrapper col1({"llo th", "", " som", "st String", ""},
                                            {true, false, true, true, false});
    cudf::test::strings_column_wrapper col2({"s", "", "  ", "", ""},
                                            {true, false, true, false, false});
    cudf::test::strings_column_wrapper col3({"", "", "", "", ""},
                                            {true, false, false, false, false});
    auto expected = cudf::table_view({col0, col1, col2, col3});
    auto prog     = cudf::strings::regex_program::create(pattern);
    auto result   = cudf::strings::split_re(sv, *prog);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected);

    // rsplit == split when using default parameters
    result = cudf::strings::rsplit_re(sv, *prog);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected);
  }
}

TEST_F(StringsSplitTest, SplitRecordRegex)
{
  std::vector<char const*> h_strings{" Héllo thesé", nullptr, "are some  ", "tést String", ""};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper input(h_strings.begin(), h_strings.end(), validity);
  auto sv = cudf::strings_column_view(input);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  {
    auto pattern = std::string("\\s+");

    LCW expected(
      {LCW{"", "Héllo", "thesé"}, LCW{}, LCW{"are", "some", ""}, LCW{"tést", "String"}, LCW{""}},
      validity);
    auto prog   = cudf::strings::regex_program::create(pattern);
    auto result = cudf::strings::split_record_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

    // rsplit == split when using default parameters
    result = cudf::strings::rsplit_record_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
  }

  {
    auto pattern = std::string("[eé]");

    LCW expected({LCW{" H", "llo th", "s", ""},
                  LCW{},
                  LCW{"ar", " som", "  "},
                  LCW{"t", "st String"},
                  LCW{""}},
                 validity);
    auto prog   = cudf::strings::regex_program::create(pattern);
    auto result = cudf::strings::split_record_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

    // rsplit == split when using default parameters
    result = cudf::strings::rsplit_record_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
  }
}

TEST_F(StringsSplitTest, SplitRecordRegexLazyQuantifier)
{
  auto const input = cudf::test::strings_column_wrapper({"\rbaab\r\ra"});
  auto const sv    = cudf::strings_column_view(input);
  using LCW        = cudf::test::lists_column_wrapper<cudf::string_view>;

  {
    LCW expected({LCW{"\rbaa", "\ra"}});
    auto const prog =
      cudf::strings::regex_program::create("[^ \v\n\t\r\f]\\r+?\\n*",
                                           cudf::strings::regex_flags::EXT_NEWLINE,
                                           cudf::strings::capture_groups::NON_CAPTURE);
    auto const result = cudf::strings::split_record_re(sv, *prog);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    LCW expected({LCW{"\rbaa", "a"}});
    auto const prog =
      cudf::strings::regex_program::create("[^ \v\n\t\r\f]\\r+\\n*",
                                           cudf::strings::regex_flags::EXT_NEWLINE,
                                           cudf::strings::capture_groups::NON_CAPTURE);
    auto const result = cudf::strings::split_record_re(sv, *prog);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

TEST_F(StringsSplitTest, SplitRegexWithMaxSplit)
{
  std::vector<char const*> h_strings{" Héllo\tthesé", nullptr, "are\nsome  ", "tést\rString", ""};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper input(h_strings.begin(), h_strings.end(), validity);
  auto sv = cudf::strings_column_view(input);
  {
    auto pattern = std::string("\\s+");

    cudf::test::strings_column_wrapper col0({"", "", "are", "tést", ""},
                                            {true, false, true, true, true});
    cudf::test::strings_column_wrapper col1({"Héllo\tthesé", "", "some  ", "String", ""},
                                            {true, false, true, true, false});
    auto expected = cudf::table_view({col0, col1});
    auto prog     = cudf::strings::regex_program::create(pattern);
    auto result   = cudf::strings::split_re(sv, *prog, 1);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected);

    // split everything is the same output as maxsplit==2 for the test input column here
    result         = cudf::strings::split_re(sv, *prog, 2);
    auto expected2 = cudf::strings::split_re(sv, *prog);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected2->view());
    result = cudf::strings::split_re(sv, *prog, 3);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected2->view());
  }
  {
    auto pattern = std::string("\\s");

    using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
    LCW expected1(
      {LCW{"", "Héllo\tthesé"}, LCW{}, LCW{"are", "some  "}, LCW{"tést", "String"}, LCW{""}},
      validity);
    auto prog   = cudf::strings::regex_program::create(pattern);
    auto result = cudf::strings::split_record_re(sv, *prog, 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected1);

    result = cudf::strings::split_record_re(sv, *prog, 2);
    LCW expected2(
      {LCW{"", "Héllo", "thesé"}, LCW{}, LCW{"are", "some", " "}, LCW{"tést", "String"}, LCW{""}},
      validity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected2);

    // split everything is the same output as maxsplit==3 for the test input column here
    result         = cudf::strings::split_record_re(sv, *prog, 3);
    auto expected0 = cudf::strings::split_record_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected0->view());
    result = cudf::strings::split_record_re(sv, *prog, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected0->view());
  }
}

TEST_F(StringsSplitTest, SplitRegexWordBoundary)
{
  cudf::test::strings_column_wrapper input({"a", "ab", "-+", "e\né"});
  auto sv = cudf::strings_column_view(input);
  {
    auto pattern = std::string("\\b");

    cudf::test::strings_column_wrapper col0({"", "", "-+", ""});
    cudf::test::strings_column_wrapper col1({"a", "ab", "", "e"}, {true, true, false, true});
    cudf::test::strings_column_wrapper col2({"", "", "", "\n"}, {true, true, false, true});
    cudf::test::strings_column_wrapper col3({"", "", "", "é"}, {false, false, false, true});
    cudf::test::strings_column_wrapper col4({"", "", "", ""}, {false, false, false, true});
    auto expected = cudf::table_view({col0, col1, col2, col3, col4});
    auto prog     = cudf::strings::regex_program::create(pattern);
    auto result   = cudf::strings::split_re(sv, *prog);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected);
  }
  {
    auto pattern = std::string("\\B");

    using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
    LCW expected({LCW{"a"}, LCW{"a", "b"}, LCW{"", "-", "+", ""}, LCW{"e\né"}});
    auto prog   = cudf::strings::regex_program::create(pattern);
    auto result = cudf::strings::split_record_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
  }
}

TEST_F(StringsSplitTest, SplitRegexAllEmpty)
{
  auto input = cudf::test::strings_column_wrapper({"", "", "", ""});
  auto sv    = cudf::strings_column_view(input);
  auto prog  = cudf::strings::regex_program::create("[ _]");

  auto result = cudf::strings::split_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view().column(0), input);
  result = cudf::strings::rsplit_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view().column(0), input);

  auto rec_result = cudf::strings::split_record_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view().column(0), input);
  rec_result = cudf::strings::rsplit_record_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view().column(0), input);
}

TEST_F(StringsSplitTest, RSplitRecord)
{
  std::vector<char const*> h_strings{
    "héllo", nullptr, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", " a b ", " a  bbb   c"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"héllo"},
                LCW{},
                LCW{"a", "bc", "déf"},
                LCW{"a", "", "bc"},
                LCW{"", "ab", "cd"},
                LCW{"ab", "cd", ""},
                LCW{""},
                LCW{" a b "},
                LCW{" a  bbb   c"}},
               validity);
  auto result =
    cudf::strings::rsplit_record(cudf::strings_column_view(strings), cudf::string_scalar("_"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, RSplitRecordWithMaxSplit)
{
  std::vector<char const*> h_strings{"héllo",
                                     nullptr,
                                     "a_bc_déf",
                                     "___a__bc",
                                     "_ab_cd_",
                                     "ab_cd_",
                                     "",
                                     " a b ___",
                                     "___ a  bbb   c"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"héllo"},
                LCW{},
                LCW{"a", "bc", "déf"},
                LCW{"___a", "", "bc"},
                LCW{"_ab", "cd", ""},
                LCW{"ab", "cd", ""},
                LCW{""},
                LCW{" a b _", "", ""},
                LCW{"_", "", " a  bbb   c"}},
               validity);

  auto result =
    cudf::strings::rsplit_record(cudf::strings_column_view(strings), cudf::string_scalar("_"), 2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, RSplitRecordWhitespace)
{
  std::vector<char const*> h_strings{"héllo", nullptr, "a_bc_déf", "", " a\tb ", " a\r bbb   c"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"héllo"}, LCW{}, LCW{"a_bc_déf"}, LCW{}, LCW{"a", "b"}, LCW{"a", "bbb", "c"}},
               validity);

  auto result = cudf::strings::rsplit_record(cudf::strings_column_view(strings));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, RSplitRecordWhitespaceWithMaxSplit)
{
  std::vector<char const*> h_strings{
    "  héllo Asher ", nullptr, "   a_bc_déf   ", "", " a\tb ", " a\r bbb   c"};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end(), validity);

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected(
    {LCW{"  héllo", "Asher"}, LCW{}, LCW{"a_bc_déf"}, LCW{}, LCW{" a", "b"}, LCW{" a\r bbb", "c"}},
    validity);

  auto result =
    cudf::strings::rsplit_record(cudf::strings_column_view(strings), cudf::string_scalar(""), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringsSplitTest, RSplitRegexWithMaxSplit)
{
  std::vector<char const*> h_strings{" Héllo\tthesé", nullptr, "are some\n ", "tést\rString", ""};
  auto validity =
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper input(h_strings.begin(), h_strings.end(), validity);
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("\\s+");
  auto prog    = cudf::strings::regex_program::create(pattern);

  {
    cudf::test::strings_column_wrapper col0({" Héllo", "", "are some", "tést", ""}, validity);
    cudf::test::strings_column_wrapper col1({"thesé", "", "", "String", ""},
                                            {true, false, true, true, false});
    auto expected = cudf::table_view({col0, col1});
    auto result   = cudf::strings::rsplit_re(sv, *prog, 1);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expected);
  }
  {
    using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
    LCW expected(
      {LCW{" Héllo", "thesé"}, LCW{}, LCW{"are some", ""}, LCW{"tést", "String"}, LCW{""}},
      validity);
    auto result = cudf::strings::rsplit_record_re(sv, *prog, 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

    // split everything is the same output as any maxsplit > 2 for the test input column here
    result         = cudf::strings::rsplit_record_re(sv, *prog, 3);
    auto expected0 = cudf::strings::rsplit_record_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected0->view());
    result = cudf::strings::rsplit_record_re(sv, *prog, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected0->view());
  }
}

TEST_F(StringsSplitTest, SplitZeroSizeStringsColumns)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto prog    = cudf::strings::regex_program::create("\\s");
  auto results = cudf::strings::split(zero_size_strings_column);
  EXPECT_TRUE(results->num_columns() == 1);
  EXPECT_TRUE(results->num_rows() == 0);
  results = cudf::strings::rsplit(zero_size_strings_column);
  EXPECT_TRUE(results->num_columns() == 1);
  EXPECT_TRUE(results->num_rows() == 0);
  results = cudf::strings::split_re(zero_size_strings_column, *prog);
  EXPECT_TRUE(results->num_columns() == 1);
  EXPECT_TRUE(results->num_rows() == 0);
  results = cudf::strings::rsplit_re(zero_size_strings_column, *prog);
  EXPECT_TRUE(results->num_columns() == 1);
  EXPECT_TRUE(results->num_rows() == 0);

  auto target      = cudf::string_scalar(" ");
  auto list_result = cudf::strings::split_record(zero_size_strings_column);
  EXPECT_TRUE(list_result->size() == 0);
  list_result = cudf::strings::rsplit_record(zero_size_strings_column);
  EXPECT_TRUE(list_result->size() == 0);
  list_result = cudf::strings::split_record(zero_size_strings_column, target);
  EXPECT_TRUE(list_result->size() == 0);
  list_result = cudf::strings::rsplit_record(zero_size_strings_column, target);
  EXPECT_TRUE(list_result->size() == 0);
  list_result = cudf::strings::split_record_re(zero_size_strings_column, *prog);
  EXPECT_TRUE(list_result->size() == 0);
  list_result = cudf::strings::rsplit_record_re(zero_size_strings_column, *prog);
  EXPECT_TRUE(list_result->size() == 0);

  auto part_result = cudf::strings::split_part(zero_size_strings_column);
  EXPECT_TRUE(part_result->size() == 0);
  part_result = cudf::strings::split_part(zero_size_strings_column, target, 1);
  EXPECT_TRUE(part_result->size() == 0);
}

// This test specifically for https://github.com/rapidsai/custrings/issues/119
TEST_F(StringsSplitTest, AllNullsCase)
{
  cudf::test::strings_column_wrapper input({"", "", ""}, {false, false, false});
  auto sv   = cudf::strings_column_view(input);
  auto prog = cudf::strings::regex_program::create("-");

  auto results = cudf::strings::split(sv);
  EXPECT_TRUE(results->num_columns() == 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->get_column(0).view(), input);
  results = cudf::strings::split(sv, cudf::string_scalar("-"));
  EXPECT_TRUE(results->num_columns() == 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->get_column(0).view(), input);
  results = cudf::strings::rsplit(sv);
  EXPECT_TRUE(results->num_columns() == 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->get_column(0).view(), input);
  results = cudf::strings::rsplit(sv, cudf::string_scalar("-"));
  EXPECT_TRUE(results->num_columns() == 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->get_column(0).view(), input);
  results = cudf::strings::split_re(sv, *prog);
  EXPECT_TRUE(results->num_columns() == 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->get_column(0).view(), input);
  results = cudf::strings::rsplit_re(sv, *prog);
  EXPECT_TRUE(results->num_columns() == 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->get_column(0).view(), input);

  auto target      = cudf::string_scalar(" ");
  auto list_result = cudf::strings::split_record(sv);
  using LCW        = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{}, LCW{}, LCW{}}, cudf::test::iterators::all_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(list_result->view(), expected);
  list_result = cudf::strings::rsplit_record(sv);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(list_result->view(), expected);
  list_result = cudf::strings::split_record(sv, target);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(list_result->view(), expected);
  list_result = cudf::strings::rsplit_record(sv, target);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(list_result->view(), expected);
  list_result = cudf::strings::split_record_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(list_result->view(), expected);
  list_result = cudf::strings::rsplit_record_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(list_result->view(), expected);

  auto part_result = cudf::strings::split_part(sv, cudf::string_scalar("-"), 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(part_result->view(), input);
  part_result = cudf::strings::split_part(sv, cudf::string_scalar("-"), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(part_result->view(), input);
}

TEST_F(StringsSplitTest, SplitPart)
{
  cudf::test::strings_column_wrapper input({"a-b-c", "é-bb-c", "a-bé-ccc", "", "", "xx-yy zz"},
                                           {true, true, true, false, true, true});
  auto sv        = cudf::strings_column_view(input);
  auto delimiter = cudf::string_scalar("-");

  auto result   = cudf::strings::split_part(sv, delimiter, 0);
  auto expected = cudf::test::strings_column_wrapper({"a", "é", "a", "", "", "xx"},
                                                     {true, true, true, false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  result   = cudf::strings::split_part(sv, delimiter, 1);
  expected = cudf::test::strings_column_wrapper({"b", "bb", "bé", "", "", "yy zz"},
                                                {true, true, true, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  result   = cudf::strings::split_part(sv, delimiter, 2);
  expected = cudf::test::strings_column_wrapper({"c", "c", "ccc", "", "", ""},
                                                {true, true, true, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  result   = cudf::strings::split_part(sv, delimiter, 3);
  expected = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                {false, false, false, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  result   = cudf::strings::split_part(sv, cudf::string_scalar("-bé-"), 1);
  expected = cudf::test::strings_column_wrapper({"", "", "ccc", "", "", ""},
                                                {false, false, true, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(StringsSplitTest, SplitPartWhitespace)
{
  cudf::test::strings_column_wrapper input({"a b  c", "é  bb c", " a bé ccc ", "", "", "xx yy-zz"},
                                           {true, true, true, false, true, true});
  auto sv        = cudf::strings_column_view(input);
  auto delimiter = cudf::string_scalar("");

  auto result   = cudf::strings::split_part(sv, delimiter, 0);
  auto expected = cudf::test::strings_column_wrapper({"a", "é", "a", "", "", "xx"},
                                                     {true, true, true, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  result   = cudf::strings::split_part(sv, delimiter, 1);
  expected = cudf::test::strings_column_wrapper({"b", "bb", "bé", "", "", "yy-zz"},
                                                {true, true, true, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  result   = cudf::strings::split_part(sv, delimiter, 2);
  expected = cudf::test::strings_column_wrapper({"c", "c", "ccc", "", "", ""},
                                                {true, true, true, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  result   = cudf::strings::split_part(sv, delimiter, 3);
  expected = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                {false, false, false, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(StringsSplitTest, Partition)
{
  std::vector<char const*> h_strings{
    "héllo", nullptr, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", " a b "};
  std::vector<char const*> h_expecteds{
    "héllo", nullptr, "a", "a", "", "ab",    "",       " a b ", "",      nullptr, "_", "_",
    "_",     "_",     "",  "",  "", nullptr, "bc_déf", "_bc",   "ab_cd", "cd_",   "",  ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);
  auto results = cudf::strings::partition(strings_view, cudf::string_scalar("_"));
  EXPECT_TRUE(results->num_columns() == 3);

  auto exp_itr = h_expecteds.begin();
  cudf::test::strings_column_wrapper expected1(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected2(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected3(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, PartitionWhitespace)
{
  std::vector<char const*> h_strings{
    "héllo", nullptr, "a bc déf", "a  bc", " ab cd", "ab cd ", "", "a_b"};
  std::vector<char const*> h_expecteds{"héllo", nullptr, "a",      "a",   "",      "ab",  "", "a_b",
                                       "",      nullptr, " ",      " ",   " ",     " ",   "", "",
                                       "",      nullptr, "bc déf", " bc", "ab cd", "cd ", "", ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);
  auto results = cudf::strings::partition(strings_view);
  EXPECT_TRUE(results->num_columns() == 3);

  auto exp_itr = h_expecteds.begin();
  cudf::test::strings_column_wrapper expected1(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected2(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected3(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RPartition)
{
  std::vector<char const*> h_strings{
    "héllo", nullptr, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", " a b "};
  std::vector<char const*> h_expecteds{"",      nullptr, "a_bc", "a_", "_ab", "ab_cd", "", "",
                                       "",      nullptr, "_",    "_",  "_",   "_",     "", "",
                                       "héllo", nullptr, "déf",  "bc", "cd",  "",      "", " a b "};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);
  auto results = cudf::strings::rpartition(strings_view, cudf::string_scalar("_"));
  EXPECT_TRUE(results->num_columns() == 3);

  auto exp_itr = h_expecteds.begin();
  cudf::test::strings_column_wrapper expected1(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected2(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected3(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, RPartitionWhitespace)
{
  std::vector<char const*> h_strings{
    "héllo", nullptr, "a bc déf", "a  bc", " ab cd", "ab cd ", "", "a_b"};
  std::vector<char const*> h_expecteds{"",      nullptr, "a bc", "a ", " ab", "ab cd", "", "",
                                       "",      nullptr, " ",    " ",  " ",   " ",     "", "",
                                       "héllo", nullptr, "déf",  "bc", "cd",  "",      "", "a_b"};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::strings_column_view strings_view(strings);
  auto results = cudf::strings::rpartition(strings_view);
  EXPECT_TRUE(results->num_columns() == 3);

  auto exp_itr = h_expecteds.begin();
  cudf::test::strings_column_wrapper expected1(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected2(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  exp_itr += h_strings.size();
  cudf::test::strings_column_wrapper expected3(
    exp_itr,
    exp_itr + h_strings.size(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(expected1.release());
  expected_columns.push_back(expected2.release());
  expected_columns.push_back(expected3.release());
  auto expected = std::make_unique<cudf::table>(std::move(expected_columns));

  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, *expected);
}

TEST_F(StringsSplitTest, PartitionZeroSizeStringsColumns)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto results = cudf::strings::partition(zero_size_strings_column);
  EXPECT_TRUE(results->num_columns() == 0);
  results = cudf::strings::rpartition(zero_size_strings_column);
  EXPECT_TRUE(results->num_columns() == 0);
}

TEST_F(StringsSplitTest, InvalidParameter)
{
  cudf::test::strings_column_wrapper input({"string left intentionally blank"});
  auto strings_view = cudf::strings_column_view(input);
  auto prog         = cudf::strings::regex_program::create("");
  EXPECT_THROW(cudf::strings::split(strings_view, cudf::string_scalar("", false)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::rsplit(strings_view, cudf::string_scalar("", false)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::split_record(strings_view, cudf::string_scalar("", false)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::rsplit_record(strings_view, cudf::string_scalar("", false)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::split_re(strings_view, *prog), cudf::logic_error);
  EXPECT_THROW(cudf::strings::split_record_re(strings_view, *prog), cudf::logic_error);
  EXPECT_THROW(cudf::strings::rsplit_re(strings_view, *prog), cudf::logic_error);
  EXPECT_THROW(cudf::strings::rsplit_record_re(strings_view, *prog), cudf::logic_error);
  EXPECT_THROW(cudf::strings::partition(strings_view, cudf::string_scalar("", false)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::rpartition(strings_view, cudf::string_scalar("", false)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::split_part(strings_view, cudf::string_scalar("", false), 0),
               std::invalid_argument);
  EXPECT_THROW(cudf::strings::split_part(strings_view, cudf::string_scalar(" "), -1),
               std::invalid_argument);
}
