/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>

using namespace cudf::test::iterators;

struct StringsListsConcatenateTest : public cudf::test::BaseFixture {};

namespace {
using STR_LISTS = cudf::test::lists_column_wrapper<cudf::string_view>;
using STR_COL   = cudf::test::strings_column_wrapper;
using INT_LISTS = cudf::test::lists_column_wrapper<int32_t>;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
}  // namespace

TEST_F(StringsListsConcatenateTest, InvalidInput)
{
  // Invalid list type
  {
    auto const string_lists = INT_LISTS{{1, 2, 3}, {4, 5, 6}}.release();
    auto const string_lv    = cudf::lists_column_view(string_lists->view());
    EXPECT_THROW(cudf::strings::join_list_elements(string_lv), cudf::logic_error);
  }

  // Invalid scalar separator
  {
    auto const string_lists =
      STR_LISTS{STR_LISTS{""}, STR_LISTS{"", "", ""}, STR_LISTS{"", ""}}.release();
    auto const string_lv = cudf::lists_column_view(string_lists->view());
    EXPECT_THROW(cudf::strings::join_list_elements(string_lv, cudf::string_scalar("", false)),
                 cudf::logic_error);
  }

  // Invalid column separators
  {
    auto const string_lists =
      STR_LISTS{STR_LISTS{""}, STR_LISTS{"", "", ""}, STR_LISTS{"", ""}}.release();
    auto const string_lv  = cudf::lists_column_view(string_lists->view());
    auto const separators = STR_COL{"+++"}.release();  // size doesn't match with lists column size
    EXPECT_THROW(cudf::strings::join_list_elements(string_lv, separators->view()),
                 cudf::logic_error);
  }
}

TEST_F(StringsListsConcatenateTest, EmptyInput)
{
  auto const string_lists = STR_LISTS{}.release();
  auto const string_lv    = cudf::lists_column_view(string_lists->view());
  auto const expected     = STR_COL{};
  auto results            = cudf::strings::join_list_elements(string_lv);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);

  auto const separators = STR_COL{}.release();
  results               = cudf::strings::join_list_elements(string_lv, separators->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
}

TEST_F(StringsListsConcatenateTest, ZeroSizeStringsInput)
{
  auto const string_lists =
    STR_LISTS{STR_LISTS{""}, STR_LISTS{"", "", ""}, STR_LISTS{"", ""}, STR_LISTS{}}.release();
  auto const string_lv = cudf::lists_column_view(string_lists->view());

  // Empty list results in empty string
  {
    auto const expected = STR_COL{"", "", "", ""};

    auto results = cudf::strings::join_list_elements(string_lv);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);

    auto const separators = STR_COL{"", "", "", ""}.release();
    results               = cudf::strings::join_list_elements(string_lv, separators->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
  }

  // Empty list results in null
  {
    auto const expected = STR_COL{{"", "", "", "" /*NULL*/}, null_at(3)};
    auto results =
      cudf::strings::join_list_elements(string_lv,
                                        cudf::string_scalar(""),
                                        cudf::string_scalar(""),
                                        cudf::strings::separator_on_nulls::NO,
                                        cudf::strings::output_if_empty_list::NULL_ELEMENT);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);

    auto const separators = STR_COL{"", "", "", ""}.release();
    results = cudf::strings::join_list_elements(string_lv,
                                                separators->view(),
                                                cudf::string_scalar(""),
                                                cudf::string_scalar(""),
                                                cudf::strings::separator_on_nulls::NO,
                                                cudf::strings::output_if_empty_list::NULL_ELEMENT);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }
}

TEST_F(StringsListsConcatenateTest, ColumnHasEmptyListAndNullListInput)
{
  auto const string_lists =
    STR_LISTS{{STR_LISTS{"abc", "def", ""}, STR_LISTS{} /*NULL*/, STR_LISTS{}, STR_LISTS{"gh"}},
              null_at(1)}
      .release();
  auto const string_lv = cudf::lists_column_view(string_lists->view());

  // Empty list results in empty string
  {
    auto const expected = STR_COL{{"abc-def-", "" /*NULL*/, "", "gh"}, null_at(1)};

    auto results = cudf::strings::join_list_elements(string_lv, cudf::string_scalar("-"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);

    auto const separators = STR_COL{"-", "", "", ""}.release();
    results               = cudf::strings::join_list_elements(string_lv, separators->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Empty list results in null
  {
    auto const expected = STR_COL{{"abc-def-", "" /*NULL*/, "" /*NULL*/, "gh"}, nulls_at({1, 2})};
    auto results =
      cudf::strings::join_list_elements(string_lv,
                                        cudf::string_scalar("-"),
                                        cudf::string_scalar(""),
                                        cudf::strings::separator_on_nulls::NO,
                                        cudf::strings::output_if_empty_list::NULL_ELEMENT);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);

    auto const separators = STR_COL{"-", "", "", ""}.release();
    results = cudf::strings::join_list_elements(string_lv,
                                                separators->view(),
                                                cudf::string_scalar(""),
                                                cudf::string_scalar(""),
                                                cudf::strings::separator_on_nulls::NO,
                                                cudf::strings::output_if_empty_list::NULL_ELEMENT);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }
}

TEST_F(StringsListsConcatenateTest, AllNullsStringsInput)
{
  auto const string_lists = STR_LISTS{
    STR_LISTS{{""}, all_nulls()},
    STR_LISTS{{"", "", ""}, all_nulls()},
    STR_LISTS{{"", ""},
              all_nulls()}}.release();
  auto const string_lv = cudf::lists_column_view(string_lists->view());
  auto const expected  = STR_COL{{"", "", ""}, all_nulls()};

  auto results = cudf::strings::join_list_elements(string_lv);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);

  auto const separators = STR_COL{{"", "", ""}, all_nulls()}.release();
  results               = cudf::strings::join_list_elements(string_lv, separators->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
}

TEST_F(StringsListsConcatenateTest, ScalarSeparator)
{
  auto const string_lists = STR_LISTS{{STR_LISTS{{"a", "bb" /*NULL*/, "ccc"}, null_at(1)},
                                       STR_LISTS{}, /*NULL*/
                                       STR_LISTS{{"ddd" /*NULL*/, "efgh", "ijk"}, null_at(0)},
                                       STR_LISTS{"zzz", "xxxxx"},
                                       STR_LISTS{{"v", "", "", "w"}, nulls_at({1, 2})}},
                                      null_at(1)}
                              .release();
  auto const string_lv    = cudf::lists_column_view(string_lists->view());

  // No null replacement
  {
    auto const results = cudf::strings::join_list_elements(string_lv, cudf::string_scalar("+++"));
    std::vector<char const*> h_expected{nullptr, nullptr, nullptr, "zzz+++xxxxx", nullptr};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // With null replacement
  {
    auto const results = cudf::strings::join_list_elements(
      string_lv, cudf::string_scalar("+++"), cudf::string_scalar("___"));
    std::vector<char const*> h_expected{
      "a+++___+++ccc", nullptr, "___+++efgh+++ijk", "zzz+++xxxxx", "v+++___+++___+++w"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Turn off separator-on-nulls
  {
    auto const results = cudf::strings::join_list_elements(string_lv,
                                                           cudf::string_scalar("+++"),
                                                           cudf::string_scalar(""),
                                                           cudf::strings::separator_on_nulls::NO);
    std::vector<char const*> h_expected{"a+++ccc", nullptr, "efgh+++ijk", "zzz+++xxxxx", "v+++w"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }
}

TEST_F(StringsListsConcatenateTest, SlicedListsWithScalarSeparator)
{
  auto const string_lists = STR_LISTS{
    {STR_LISTS{{"a", "bb" /*NULL*/, "ccc"}, null_at(1)},
     STR_LISTS{}, /*NULL*/
     STR_LISTS{{"ddd" /*NULL*/, "efgh", "ijk"}, null_at(0)},
     STR_LISTS{"zzz", "xxxxx"},
     STR_LISTS{"11111", "11111", "11111", "11111", "11111"}, /*NULL*/
     STR_LISTS{{"abcdef", "012345", "" /*NULL*/, "xxx000"}, null_at(2)},
     STR_LISTS{{"xyz" /*NULL*/, "11111", "00000"}, null_at(0)},
     STR_LISTS{"0a0b0c", "5x5y5z"},
     STR_LISTS{"xxx"}, /*NULL*/
     STR_LISTS{"ééé", "12345abcdef"},
     STR_LISTS{"aaaééébbbéééccc", "12345"}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 1 && i != 4 && i != 8;
    })}.release();

  // Sliced the entire lists column, no null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {0, 11})[0]);
    auto const results   = cudf::strings::join_list_elements(string_lv, cudf::string_scalar("+++"));
    std::vector<char const*> h_expected{nullptr,
                                        nullptr,
                                        nullptr,
                                        "zzz+++xxxxx",
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        "0a0b0c+++5x5y5z",
                                        nullptr,
                                        "ééé+++12345abcdef",
                                        "aaaééébbbéééccc+++12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the entire lists column, with null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {0, 11})[0]);
    auto const results   = cudf::strings::join_list_elements(
      string_lv, cudf::string_scalar("+++"), cudf::string_scalar("___"));
    std::vector<char const*> h_expected{"a+++___+++ccc",
                                        nullptr,
                                        "___+++efgh+++ijk",
                                        "zzz+++xxxxx",
                                        nullptr,
                                        "abcdef+++012345+++___+++xxx000",
                                        "___+++11111+++00000",
                                        "0a0b0c+++5x5y5z",
                                        nullptr,
                                        "ééé+++12345abcdef",
                                        "aaaééébbbéééccc+++12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the first half of the lists column, no null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {0, 4})[0]);
    auto const results   = cudf::strings::join_list_elements(string_lv, cudf::string_scalar("+++"));
    std::vector<char const*> h_expected{nullptr, nullptr, nullptr, "zzz+++xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the first half of the lists column, with null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {0, 4})[0]);
    auto const results   = cudf::strings::join_list_elements(
      string_lv, cudf::string_scalar("+++"), cudf::string_scalar("___"));
    std::vector<char const*> h_expected{
      "a+++___+++ccc", nullptr, "___+++efgh+++ijk", "zzz+++xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the second half of the lists column, no null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {5, 11})[0]);
    auto const results   = cudf::strings::join_list_elements(string_lv, cudf::string_scalar("+++"));
    std::vector<char const*> h_expected{
      nullptr, nullptr, "0a0b0c+++5x5y5z", nullptr, "ééé+++12345abcdef", "aaaééébbbéééccc+++12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the second half of the lists column, with null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {5, 11})[0]);
    auto const results   = cudf::strings::join_list_elements(
      string_lv, cudf::string_scalar("+++"), cudf::string_scalar("___"));
    std::vector<char const*> h_expected{"abcdef+++012345+++___+++xxx000",
                                        "___+++11111+++00000",
                                        "0a0b0c+++5x5y5z",
                                        nullptr,
                                        "ééé+++12345abcdef",
                                        "aaaééébbbéééccc+++12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the middle part of the lists column, no null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {3, 8})[0]);
    auto const results   = cudf::strings::join_list_elements(string_lv, cudf::string_scalar("+++"));
    std::vector<char const*> h_expected{
      "zzz+++xxxxx", nullptr, nullptr, nullptr, "0a0b0c+++5x5y5z"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the middle part of the lists column, with null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {3, 8})[0]);
    auto const results   = cudf::strings::join_list_elements(
      string_lv, cudf::string_scalar("+++"), cudf::string_scalar("___"));
    std::vector<char const*> h_expected{"zzz+++xxxxx",
                                        nullptr,
                                        "abcdef+++012345+++___+++xxx000",
                                        "___+++11111+++00000",
                                        "0a0b0c+++5x5y5z"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }
}

TEST_F(StringsListsConcatenateTest, ColumnSeparators)
{
  auto const string_lists = STR_LISTS{{STR_LISTS{{"a", "bb" /*NULL*/, "ccc"}, null_at(1)},
                                       STR_LISTS{}, /*NULL*/
                                       STR_LISTS{"0a0b0c", "xyzééé"},
                                       STR_LISTS{{"ddd" /*NULL*/, "efgh", "ijk"}, null_at(0)},
                                       STR_LISTS{{"ééé" /*NULL*/, "ááá", "ííí"}, null_at(0)},
                                       STR_LISTS{"zzz", "xxxxx"}},
                                      null_at(1)}
                              .release();
  auto const string_lv    = cudf::lists_column_view(string_lists->view());
  auto const separators   = STR_COL{
    {"+++", "***", "!!!" /*NULL*/, "$$$" /*NULL*/, "%%%", "^^^"},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 2 && i != 3;
    })}.release();

  // No null replacement
  {
    auto const results = cudf::strings::join_list_elements(string_lv, separators->view());
    std::vector<char const*> h_expected{nullptr, nullptr, nullptr, nullptr, nullptr, "zzz^^^xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // With null replacement for separators
  {
    auto const results =
      cudf::strings::join_list_elements(string_lv, separators->view(), cudf::string_scalar("|||"));
    std::vector<char const*> h_expected{
      nullptr, nullptr, "0a0b0c|||xyzééé", nullptr, nullptr, "zzz^^^xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // With null replacement for strings
  {
    auto const results = cudf::strings::join_list_elements(
      string_lv, separators->view(), cudf::string_scalar("", false), cudf::string_scalar("XXXXX"));
    std::vector<char const*> h_expected{
      "a+++XXXXX+++ccc", nullptr, nullptr, nullptr, "XXXXX%%%ááá%%%ííí", "zzz^^^xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // With null replacement for both separators and strings
  {
    auto const results = cudf::strings::join_list_elements(
      string_lv, separators->view(), cudf::string_scalar("|||"), cudf::string_scalar("XXXXX"));
    std::vector<char const*> h_expected{"a+++XXXXX+++ccc",
                                        nullptr,
                                        "0a0b0c|||xyzééé",
                                        "XXXXX|||efgh|||ijk",
                                        "XXXXX%%%ááá%%%ííí",
                                        "zzz^^^xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Turn off separator-on-nulls
  {
    auto const results = cudf::strings::join_list_elements(string_lv,
                                                           separators->view(),
                                                           cudf::string_scalar("+++"),
                                                           cudf::string_scalar(""),
                                                           cudf::strings::separator_on_nulls::NO);
    std::vector<char const*> h_expected{
      "a+++ccc", nullptr, "0a0b0c+++xyzééé", "efgh+++ijk", "ááá%%%ííí", "zzz^^^xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }
}

TEST_F(StringsListsConcatenateTest, SlicedListsWithColumnSeparators)
{
  auto const string_lists = STR_LISTS{
    {STR_LISTS{{"a", "bb" /*NULL*/, "ccc"}, null_at(1)},
     STR_LISTS{}, /*NULL*/
     STR_LISTS{{"ddd" /*NULL*/, "efgh", "ijk"}, null_at(0)},
     STR_LISTS{"zzz", "xxxxx"},
     STR_LISTS{"11111", "11111", "11111", "11111", "11111"}, /*NULL*/
     STR_LISTS{{"abcdef", "012345", "" /*NULL*/, "xxx000"}, null_at(2)},
     STR_LISTS{{"xyz" /*NULL*/, "11111", "00000"}, null_at(0)},
     STR_LISTS{"0a0b0c", "5x5y5z"},
     STR_LISTS{"xxx"}, /*NULL*/
     STR_LISTS{"ééé", "12345abcdef"},
     STR_LISTS{"aaaééébbbéééccc", "12345"}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 1 && i != 4 && i != 8;
    })}.release();
  auto const separators = STR_COL{
    {"+++", "***", "!!!" /*NULL*/, "$$$" /*NULL*/, "%%%", "^^^", "~!~", "###", "&&&", "-+-", "=+="},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i != 2 && i != 3;
    })}.release();

  // Sliced the entire lists column, no null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {0, 11})[0]);
    auto const sep_col   = cudf::strings_column_view(cudf::slice(separators->view(), {0, 11})[0]);
    auto const results   = cudf::strings::join_list_elements(string_lv, sep_col);
    std::vector<char const*> h_expected{nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        "0a0b0c###5x5y5z",
                                        nullptr,
                                        "ééé-+-12345abcdef",
                                        "aaaééébbbéééccc=+=12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the entire lists column, with null replacements
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {0, 11})[0]);
    auto const sep_col   = cudf::strings_column_view(cudf::slice(separators->view(), {0, 11})[0]);
    auto const results   = cudf::strings::join_list_elements(
      string_lv, sep_col, cudf::string_scalar("|||"), cudf::string_scalar("___"));
    std::vector<char const*> h_expected{"a+++___+++ccc",
                                        nullptr,
                                        "___|||efgh|||ijk",
                                        "zzz|||xxxxx",
                                        nullptr,
                                        "abcdef^^^012345^^^___^^^xxx000",
                                        "___~!~11111~!~00000",
                                        "0a0b0c###5x5y5z",
                                        nullptr,
                                        "ééé-+-12345abcdef",
                                        "aaaééébbbéééccc=+=12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the first half of the lists column, no null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {0, 4})[0]);
    auto const sep_col   = cudf::strings_column_view(cudf::slice(separators->view(), {0, 4})[0]);
    auto const results   = cudf::strings::join_list_elements(string_lv, sep_col);
    std::vector<char const*> h_expected{nullptr, nullptr, nullptr, nullptr};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the first half of the lists column, with null replacements
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {0, 4})[0]);
    auto const sep_col   = cudf::strings_column_view(cudf::slice(separators->view(), {0, 4})[0]);
    auto const results   = cudf::strings::join_list_elements(
      string_lv, sep_col, cudf::string_scalar("|||"), cudf::string_scalar("___"));
    std::vector<char const*> h_expected{
      "a+++___+++ccc", nullptr, "___|||efgh|||ijk", "zzz|||xxxxx"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the second half of the lists column, no null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {5, 11})[0]);
    auto const sep_col   = cudf::strings_column_view(cudf::slice(separators->view(), {5, 11})[0]);
    auto const results   = cudf::strings::join_list_elements(string_lv, sep_col);
    std::vector<char const*> h_expected{
      nullptr, nullptr, "0a0b0c###5x5y5z", nullptr, "ééé-+-12345abcdef", "aaaééébbbéééccc=+=12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the second half of the lists column, with null replacements
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {5, 11})[0]);
    auto const sep_col   = cudf::strings_column_view(cudf::slice(separators->view(), {5, 11})[0]);
    auto const results   = cudf::strings::join_list_elements(
      string_lv, sep_col, cudf::string_scalar("|||"), cudf::string_scalar("___"));
    std::vector<char const*> h_expected{"abcdef^^^012345^^^___^^^xxx000",
                                        "___~!~11111~!~00000",
                                        "0a0b0c###5x5y5z",
                                        nullptr,
                                        "ééé-+-12345abcdef",
                                        "aaaééébbbéééccc=+=12345"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the middle part of the lists column, no null replacement
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {3, 8})[0]);
    auto const sep_col   = cudf::strings_column_view(cudf::slice(separators->view(), {3, 8})[0]);
    auto const results   = cudf::strings::join_list_elements(string_lv, sep_col);
    std::vector<char const*> h_expected{nullptr, nullptr, nullptr, nullptr, "0a0b0c###5x5y5z"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }

  // Sliced the middle part of the lists column, with null replacements
  {
    auto const string_lv = cudf::lists_column_view(cudf::slice(string_lists->view(), {3, 8})[0]);
    auto const sep_col   = cudf::strings_column_view(cudf::slice(separators->view(), {3, 8})[0]);
    auto const results   = cudf::strings::join_list_elements(
      string_lv, sep_col, cudf::string_scalar("|||"), cudf::string_scalar("___"));
    std::vector<char const*> h_expected{"zzz|||xxxxx",
                                        nullptr,
                                        "abcdef^^^012345^^^___^^^xxx000",
                                        "___~!~11111~!~00000",
                                        "0a0b0c###5x5y5z"};
    auto const expected =
      STR_COL{h_expected.begin(), h_expected.end(), nulls_from_nullptrs(h_expected)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected, verbosity);
  }
}
