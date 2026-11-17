/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

struct StringsCombineTest : public cudf::test::BaseFixture {};

TEST_F(StringsCombineTest, Concatenate)
{
  std::vector<char const*> h_strings1{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings1(
    h_strings1.begin(),
    h_strings1.end(),
    cuda::transform_iterator(h_strings1.begin(), [](auto str) { return str != nullptr; }));
  std::vector<char const*> h_strings2{"xyz", "abc", "d", "éa", "", nullptr, "f"};
  cudf::test::strings_column_wrapper strings2(
    h_strings2.begin(),
    h_strings2.end(),
    cuda::transform_iterator(h_strings2.begin(), [](auto str) { return str != nullptr; }));

  std::vector<cudf::column_view> strings_columns;
  strings_columns.push_back(strings1);
  strings_columns.push_back(strings2);

  cudf::table_view table(strings_columns);

  {
    std::vector<char const*> h_expected{"eeexyz", "bbabc", nullptr, "éa", "aa", nullptr, "éééf"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

    auto results = cudf::strings::concatenate(table);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_expected{
      "eee:xyz", "bb:abc", nullptr, ":éa", "aa:", nullptr, "ééé:f"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

    auto results = cudf::strings::concatenate(table, cudf::string_scalar(":"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_expected{"eee:xyz", "bb:abc", "_:d", ":éa", "aa:", "bbb:_", "ééé:f"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

    auto results =
      cudf::strings::concatenate(table, cudf::string_scalar(":"), cudf::string_scalar("_"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<char const*> h_expected{"eeexyz", "bbabc", "d", "éa", "aa", "bbb", "éééf"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

    auto results =
      cudf::strings::concatenate(table, cudf::string_scalar(""), cudf::string_scalar(""));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsCombineTest, ConcatenateSkipNulls)
{
  cudf::test::strings_column_wrapper strings1({"eee", "", "", "", "aa", "bbb", "ééé"},
                                              {true, false, false, true, true, true, true});
  cudf::test::strings_column_wrapper strings2({"xyz", "", "d", "éa", "", "", "f"},
                                              {true, false, true, true, true, false, true});
  cudf::test::strings_column_wrapper strings3({"q", "", "s", "t", "u", "", "w"},
                                              {true, true, true, true, true, false, true});

  cudf::table_view table({strings1, strings2, strings3});

  {
    cudf::test::strings_column_wrapper expected(
      {"eee+xyz+q", "++", "+d+s", "+éa+t", "aa++u", "bbb++", "ééé+f+w"});
    auto results = cudf::strings::concatenate(table,
                                              cudf::string_scalar("+"),
                                              cudf::string_scalar(""),
                                              cudf::strings::separator_on_nulls::YES);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper expected(
      {"eee+xyz+q", "", "d+s", "+éa+t", "aa++u", "bbb", "ééé+f+w"});
    auto results = cudf::strings::concatenate(table,
                                              cudf::string_scalar("+"),
                                              cudf::string_scalar(""),
                                              cudf::strings::separator_on_nulls::NO);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper expected(
      {"eee+xyz+q", "", "", "+éa+t", "aa++u", "", "ééé+f+w"},
      {true, false, false, true, true, false, true});
    auto results = cudf::strings::concatenate(table,
                                              cudf::string_scalar("+"),
                                              cudf::string_scalar("", false),
                                              cudf::strings::separator_on_nulls::NO);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper sep_col({"+", "-", ".", "@", "*", "^^", "#"});
    auto results = cudf::strings::concatenate(table,
                                              cudf::strings_column_view(sep_col),
                                              cudf::string_scalar(""),
                                              cudf::string_scalar(""),
                                              cudf::strings::separator_on_nulls::NO);

    cudf::test::strings_column_wrapper expected(
      {"eee+xyz+q", "", "d.s", "@éa@t", "aa**u", "bbb", "ééé#f#w"});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
}

TEST_F(StringsCombineTest, ConcatZeroSizeStringsColumns)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  std::vector<cudf::column_view> strings_columns;
  strings_columns.push_back(zero_size_strings_column);
  strings_columns.push_back(zero_size_strings_column);
  cudf::table_view table(strings_columns);
  auto results = cudf::strings::concatenate(table);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsCombineTest, SingleColumnErrorCheck)
{
  auto const col0 = cudf::make_empty_column(cudf::type_id::STRING);
  EXPECT_THROW(cudf::strings::concatenate(cudf::table_view{{col0->view()}}), cudf::logic_error);
}

struct StringsConcatenateWithColSeparatorTest : public cudf::test::BaseFixture {};

TEST_F(StringsConcatenateWithColSeparatorTest, ExceptionTests)
{
  // Exception tests
  // 0. 0 columns passed
  // 1. > 0 columns passed; some using non string data types
  // 2. separator column of different size to column size
  {
    EXPECT_THROW(cudf::strings::concatenate(cudf::table_view{},
                                            cudf::strings_column_view{cudf::column_view{}}),
                 cudf::logic_error);
  }

  {
    auto const col0 = cudf::make_empty_column(cudf::type_id::STRING)->view();
    cudf::test::fixed_width_column_wrapper<int64_t> col1{{1}};

    EXPECT_THROW(
      cudf::strings::concatenate(cudf::table_view{{col0, col1}}, cudf::strings_column_view(col0)),
      cudf::logic_error);
  }

  {
    auto col0    = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
    auto sep_col = cudf::test::strings_column_wrapper({"", ""}, {true, false});

    EXPECT_THROW(
      cudf::strings::concatenate(cudf::table_view{{col0}}, cudf::strings_column_view(sep_col)),
      cudf::logic_error);
  }
}

TEST_F(StringsConcatenateWithColSeparatorTest, ZeroSizedColumns)
{
  auto const col0 = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0}}, cudf::strings_column_view(col0));
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnEmptyAndNullStringsNoReplacements)
{
  auto col0    = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
  auto sep_col = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, true});

  auto exp_results =
    cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, false});
  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0}}, cudf::strings_column_view(sep_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnEmptyAndNullStringsSeparatorReplacement)
{
  auto col0    = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
  auto sep_col = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, true});
  auto sep_rep = cudf::string_scalar("");

  auto exp_results =
    cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0}}, cudf::strings_column_view(sep_col), sep_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnEmptyAndNullStringsColumnReplacement)
{
  auto col0    = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
  auto sep_col = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, true});
  auto col_rep = cudf::string_scalar("");

  auto exp_results =
    cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, true});

  auto results = cudf::strings::concatenate(cudf::table_view{{col0}},
                                            cudf::strings_column_view(sep_col),
                                            cudf::string_scalar("", false),
                                            col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest,
       SingleColumnEmptyAndNullStringsSeparatorAndColumnReplacement)
{
  auto col0    = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
  auto sep_col = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, false, true});
  auto sep_rep = cudf::string_scalar("");
  auto col_rep = cudf::string_scalar("");

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", ""});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0}}, cudf::strings_column_view(sep_col), sep_rep, col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, exp_results, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnStringMixNoReplacements)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "bbabc", "invalid", "d", "éa", "invalid", "bbb", "éééf"},
    {true, false, true, true, false, true, true, false, true, true});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~", "!", "@", "#", "$", "%", "^", "&", "*"},
    {false, false, true, true, true, true, true, false, true, true});

  auto exp_results = cudf::test::strings_column_wrapper(
    {"", "", "", "bbabc", "", "d", "éa", "", "bbb", "éééf"},
    {false, false, true, true, false, true, true, false, true, true});

  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0}}, cudf::strings_column_view(sep_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnStringMixSeparatorReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "bbabc", "invalid", "d", "éa", "invalid", "bbb", "éééf"},
    {true, false, true, true, false, true, true, false, true, true});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~", "!", "@", "#", "$", "%", "^", "&", "*"},
    {false, false, false, true, true, true, true, false, true, true});
  auto sep_rep = cudf::string_scalar("-");

  auto exp_results = cudf::test::strings_column_wrapper(
    {"eeexyz", "", "", "bbabc", "", "d", "éa", "", "bbb", "éééf"},
    {true, false, true, true, false, true, true, false, true, true});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0}}, cudf::strings_column_view(sep_col), sep_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnStringMixColumnReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "bbabc", "invalid", "d", "éa", "invalid", "bbb", "éééf"},
    {true, false, true, true, false, true, true, false, true, true});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~", "!", "@", "#", "$", "%", "^", "&", "*"},
    {false, false, false, true, true, true, true, false, true, true});
  auto col_rep = cudf::string_scalar("goobly");

  auto exp_results = cudf::test::strings_column_wrapper(
    {"", "", "", "bbabc", "goobly", "d", "éa", "", "bbb", "éééf"},
    {false, false, false, true, true, true, true, false, true, true});

  auto results = cudf::strings::concatenate(cudf::table_view{{col0}},
                                            cudf::strings_column_view(sep_col),
                                            cudf::string_scalar("", false),
                                            col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, SingleColumnStringMixSeparatorAndColumnReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "bbabc", "invalid", "d", "éa", "invalid", "bbb", "éééf"},
    {true, false, true, true, false, true, true, false, true, true});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~", "!", "@", "#", "$", "%", "^", "&", "*"},
    {false, false, false, true, true, true, true, false, true, true});
  auto sep_rep = cudf::string_scalar("-");
  auto col_rep = cudf::string_scalar("goobly");

  // All valid, as every invalid element is replaced - a non nullable column
  auto exp_results = cudf::test::strings_column_wrapper(
    {"eeexyz", "goobly", "", "bbabc", "goobly", "d", "éa", "goobly", "bbb", "éééf"});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0}}, cudf::strings_column_view(sep_col), sep_rep, col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, exp_results, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnEmptyAndNullStringsNoReplacements)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {false, false, true, true, true, true, false, false});
  auto col1 = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {false, false, true, true, false, false, true, true});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {true, false, true, false, true, false, true, false});

  auto exp_results1 = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {false, false, true, false, false, false, false, false});
  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0, col1}}, cudf::strings_column_view(sep_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results1, verbosity);

  auto exp_results2 = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {true, false, true, false, true, false, true, false});
  results = cudf::strings::concatenate(cudf::table_view{{col0, col1}},
                                       cudf::strings_column_view(sep_col),
                                       cudf::string_scalar("", false),
                                       cudf::string_scalar(""),
                                       cudf::strings::separator_on_nulls::NO);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results2, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnStringMixNoReplacements)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "éééf", "éa", "", "", "invalid", "null", "NULL", "-1", ""},
    {true, true, true, true, true, true, false, false, false, false, false, false});
  auto col1 = cudf::test::strings_column_wrapper(
    {"foo", "", "éaff", "", "invalid", "NULL", "éaff", "valid", "doo", "", "<null>", "-1"},
    {true, true, true, false, false, false, true, true, true, false, false, false});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~~~", "", "@", "", "", "", "^^^^", "", "--", "*****", "######"},
    {true, true, false, true, false, true, false, true, true, true, true, true});

  auto exp_results1 = cudf::test::strings_column_wrapper(
    {"eeexyzfoo", "<null>~~~", "", "", "", "", "", "", "", "", "", ""},
    {true, true, false, false, false, false, false, false, false, false, false, false});

  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0, col1}}, cudf::strings_column_view(sep_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results1, verbosity);

  auto exp_results2 = cudf::test::strings_column_wrapper(
    {"eeexyzfoo", "<null>~~~", "", "éééf", "", "", "", "valid", "doo", "", "", ""},
    {true, true, false, true, false, true, false, true, true, true, true, true});
  results = cudf::strings::concatenate(cudf::table_view{{col0, col1}},
                                       cudf::strings_column_view(sep_col),
                                       cudf::string_scalar("", false),
                                       cudf::string_scalar(""),
                                       cudf::strings::separator_on_nulls::NO);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results2, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnStringMixSeparatorReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "éééf", "éa", "", "", "invalid", "null", "NULL", "-1", ""},
    {true, true, true, true, true, true, false, false, false, false, false, false});
  auto col1 = cudf::test::strings_column_wrapper(
    {"foo", "", "éaff", "", "invalid", "NULL", "éaff", "valid", "doo", "", "<null>", "-1"},
    {true, true, true, false, false, false, true, true, true, false, false, false});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~~~", "", "@", "", "", "", "^^^^", "", "--", "*****", "######"},
    {true, true, false, true, false, true, false, true, true, true, true, true});
  auto sep_rep = cudf::string_scalar("!!!!!!!");

  auto exp_results1 = cudf::test::strings_column_wrapper(
    {"eeexyzfoo", "<null>~~~", "!!!!!!!éaff", "éééf", "éa", "", "éaff", "valid", "doo", "", "", ""},
    {true, true, true, false, false, false, false, false, false, false, false, false});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0, col1}}, cudf::strings_column_view(sep_col), sep_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results1, verbosity);

  auto exp_results2 = cudf::test::strings_column_wrapper(
    {"eeexyzfoo", "<null>~~~", "!!!!!!!éaff", "éééf", "éa", "", "éaff", "valid", "doo", "", "", ""},
    {true, true, true, true, true, true, true, true, true, true, true, true});

  results = cudf::strings::concatenate(cudf::table_view{{col0, col1}},
                                       cudf::strings_column_view(sep_col),
                                       sep_rep,
                                       cudf::string_scalar(""),
                                       cudf::strings::separator_on_nulls::NO);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results2, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnStringMixColumnReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "éééf", "éa", "", "", "invalid", "null", "NULL", "-1", ""},
    {true, true, true, true, true, true, false, false, false, false, false, false});
  auto col1 = cudf::test::strings_column_wrapper(
    {"foo", "", "éaff", "", "invalid", "NULL", "éaff", "valid", "doo", "", "<null>", "-1"},
    {true, true, true, false, false, false, true, true, true, false, false, false});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~~~", "", "@", "", "", "", "^^^^", "", "--", "*****", "######"},
    {true, true, false, true, false, true, false, true, true, true, true, true});
  auto col_rep = cudf::string_scalar("_col_replacement_");

  auto exp_results = cudf::test::strings_column_wrapper(
    {"eeexyzfoo",
     "<null>~~~",
     "",
     "éééf@_col_replacement_",
     "",
     "_col_replacement_",
     "",
     "_col_replacement_^^^^valid",
     "_col_replacement_doo",
     "_col_replacement_--_col_replacement_",
     "_col_replacement_*****_col_replacement_",
     "_col_replacement_######_col_replacement_"},
    {true, true, false, true, false, true, false, true, true, true, true, true});

  auto results = cudf::strings::concatenate(cudf::table_view{{col0, col1}},
                                            cudf::strings_column_view(sep_col),
                                            cudf::string_scalar("", false),
                                            col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnStringMixSeparatorAndColumnReplacement)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"eeexyz", "<null>", "", "éééf", "éa", "", "", "invalid", "null", "NULL", "-1", ""},
    {true, true, true, true, true, true, false, false, false, false, false, false});
  auto col1 = cudf::test::strings_column_wrapper(
    {"foo", "", "éaff", "", "invalid", "NULL", "éaff", "valid", "doo", "", "<null>", "-1"},
    {true, true, true, false, false, false, true, true, true, false, false, false});
  auto sep_col = cudf::test::strings_column_wrapper(
    {"", "~~~", "", "@", "", "", "", "^^^^", "", "--", "*****", "######"},
    {true, true, false, true, false, true, false, true, true, true, true, true});
  auto sep_rep = cudf::string_scalar("!!!!!!!!!!");
  auto col_rep = cudf::string_scalar("_col_replacement_");

  // Every null item (separator/column) is replaced - a non nullable column
  auto exp_results =
    cudf::test::strings_column_wrapper({"eeexyzfoo",
                                        "<null>~~~",
                                        "!!!!!!!!!!éaff",
                                        "éééf@_col_replacement_",
                                        "éa!!!!!!!!!!_col_replacement_",
                                        "_col_replacement_",
                                        "_col_replacement_!!!!!!!!!!éaff",
                                        "_col_replacement_^^^^valid",
                                        "_col_replacement_doo",
                                        "_col_replacement_--_col_replacement_",
                                        "_col_replacement_*****_col_replacement_",
                                        "_col_replacement_######_col_replacement_"});

  auto results = cudf::strings::concatenate(
    cudf::table_view{{col0, col1}}, cudf::strings_column_view(sep_col), sep_rep, col_rep);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, exp_results, verbosity);
}

TEST_F(StringsConcatenateWithColSeparatorTest, MultiColumnNonNullableStrings)
{
  auto col0 =
    cudf::test::strings_column_wrapper({"eeexyz", "<null>", "éaff", "éééf", "", "", "", ""});
  auto col1    = cudf::test::strings_column_wrapper({"foo", "nan", "", "", "NULL", "éaff", "", ""});
  auto sep_col = cudf::test::strings_column_wrapper({"", "~~~", "", "@", "", "+++", "", "^^^^"});

  // Every item (separator/column) is used, as everything is valid producing a non nullable column
  auto exp_results = cudf::test::strings_column_wrapper(
    {"eeexyzfoo", "<null>~~~nan", "éaff", "éééf@", "NULL", "+++éaff", "", "^^^^"});

  auto results =
    cudf::strings::concatenate(cudf::table_view{{col0, col1}}, cudf::strings_column_view(sep_col));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, exp_results, verbosity);
}
