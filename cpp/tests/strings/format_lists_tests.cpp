/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_lists.hpp>

struct StringsFormatListsTest : public cudf::test::BaseFixture {};

TEST_F(StringsFormatListsTest, EmptyList)
{
  using STR_LISTS = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto const input = STR_LISTS{};
  auto const view  = cudf::lists_column_view(input);

  auto results = cudf::strings::format_list_column(view);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsFormatListsTest, EmptyNestedList)
{
  using STR_LISTS = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto const input = STR_LISTS{{{}, {}}, {{}}};
  auto const view  = cudf::lists_column_view(input);

  auto results  = cudf::strings::format_list_column(view);
  auto expected = cudf::test::strings_column_wrapper({"[[],[]]", "[[]]"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFormatListsTest, WithNulls)
{
  using STR_LISTS = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto const input = STR_LISTS{{{{"a", "", "ccc"}, cudf::test::iterators::null_at(1)},
                                {},
                                {{"", "bb", "ddd"}, cudf::test::iterators::null_at(0)},
                                {"zzz", "xxxxx"},
                                {{"v", "", "", "w"}, cudf::test::iterators::null_at(2)}},
                               cudf::test::iterators::null_at(1)};
  auto const view  = cudf::lists_column_view(input);

  auto null_scalar = cudf::string_scalar("NULL");
  auto results     = cudf::strings::format_list_column(view, null_scalar);
  auto expected    = cudf::test::strings_column_wrapper(
    {"[a,NULL,ccc]", "NULL", "[NULL,bb,ddd]", "[zzz,xxxxx]", "[v,,NULL,w]"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFormatListsTest, CustomParameters)
{
  using STR_LISTS = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto const input =
    STR_LISTS{{{{{"a", "", "ccc"}, cudf::test::iterators::null_at(1)}, {}, {"ddd", "ee", "f"}},
               cudf::test::iterators::null_at(1)},
              {{"gg", "hhh"}, {"i", "", "", "jj"}}};
  auto const view = cudf::lists_column_view(input);
  auto separators = cudf::test::strings_column_wrapper({": ", "{", "}"});

  auto results = cudf::strings::format_list_column(
    view, cudf::string_scalar("null"), cudf::strings_column_view(separators));
  auto expected = cudf::test::strings_column_wrapper(
    {"{{a: null: ccc}: null: {ddd: ee: f}}", "{{gg: hhh}: {i: : : jj}}"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFormatListsTest, NestedList)
{
  using STR_LISTS = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto const input =
    STR_LISTS{{{"a", "bb", "ccc"}, {}, {"ddd", "ee", "f"}}, {{"gg", "hhh"}, {"i", "", "", "jj"}}};
  auto const view = cudf::lists_column_view(input);

  auto results = cudf::strings::format_list_column(view);
  auto expected =
    cudf::test::strings_column_wrapper({"[[a,bb,ccc],[],[ddd,ee,f]]", "[[gg,hhh],[i,,,jj]]"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFormatListsTest, SlicedLists)
{
  using STR_LISTS = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto const input = STR_LISTS{{{{"a", "", "bb"}, cudf::test::iterators::null_at(1)},
                                {},
                                {{"", "ccc", "dddd"}, cudf::test::iterators::null_at(0)},
                                {"zzz", ""},
                                {},
                                {{"abcdef", "012345", "", ""}, cudf::test::iterators::null_at(2)},
                                {{"", "11111", "00000"}, cudf::test::iterators::null_at(0)},
                                {"hey hey", "way way"},
                                {},
                                {"ééé", "12345abcdef"},
                                {"www", "12345"}},
                               cudf::test::iterators::nulls_at({1, 4, 8})};

  // matching expected strings
  auto const h_expected = std::vector<std::string>({"[a,NULL,bb]",
                                                    "NULL",
                                                    "[NULL,ccc,dddd]",
                                                    "[zzz,]",
                                                    "NULL",
                                                    "[abcdef,012345,NULL,]",
                                                    "[NULL,11111,00000]",
                                                    "[hey hey,way way]",
                                                    "NULL",
                                                    "[ééé,12345abcdef]",
                                                    "[www,12345]"});

  auto null_scalar = cudf::string_scalar("NULL");

  // set of slice intervals: covers slicing the front, back, and middle
  std::vector<std::pair<int32_t, int32_t>> index_pairs({{0, 11}, {0, 4}, {3, 8}, {5, 11}});
  for (auto indexes : index_pairs) {
    auto sliced   = cudf::lists_column_view(cudf::slice(input, {indexes.first, indexes.second})[0]);
    auto results  = cudf::strings::format_list_column(sliced, null_scalar);
    auto expected = cudf::test::strings_column_wrapper(h_expected.begin() + indexes.first,
                                                       h_expected.begin() + indexes.second);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsFormatListsTest, Errors)
{
  using STR_LISTS = cudf::test::lists_column_wrapper<cudf::string_view>;

  cudf::test::lists_column_wrapper<int32_t> invalid({1, 2, 3});
  EXPECT_THROW(cudf::strings::format_list_column(cudf::lists_column_view(invalid)),
               cudf::logic_error);

  auto const input = STR_LISTS{{}, {}};
  auto const view  = cudf::lists_column_view(input);
  auto separators  = cudf::test::strings_column_wrapper({"{", "}"});

  EXPECT_THROW(cudf::strings::format_list_column(
                 view, cudf::string_scalar(""), cudf::strings_column_view(separators)),
               cudf::logic_error);

  EXPECT_THROW(cudf::strings::format_list_column(view, cudf::string_scalar("", false)),
               cudf::logic_error);
}
