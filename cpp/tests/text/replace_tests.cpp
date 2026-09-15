/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/replace.hpp>

#include <cuda/iterator>

#include <vector>

struct TextReplaceTest : public cudf::test::BaseFixture {};

TEST_F(TextReplaceTest, ReplaceTokens)
{
  std::vector<char const*> h_strings{"the fox jumped over the dog",
                                     "is theme of the thesis",
                                     nullptr,
                                     "",
                                     "no change",
                                     "thé is the cheese is"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  cudf::test::strings_column_wrapper targets({"is", "the"});
  cudf::test::strings_column_wrapper repls({"___", ""});
  std::vector<char const*> h_expected{" fox jumped over  dog",
                                      "___ theme of  thesis",
                                      nullptr,
                                      "",
                                      "no change",
                                      "thé ___  cheese ___"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = nvtext::replace_tokens(cudf::strings_column_view(strings),
                                        cudf::strings_column_view(targets),
                                        cudf::strings_column_view(repls));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::replace_tokens(cudf::strings_column_view(strings),
                                   cudf::strings_column_view(targets),
                                   cudf::strings_column_view(repls),
                                   cudf::string_scalar("o "));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextReplaceTest, ReplaceTokensSingleRepl)
{
  cudf::test::strings_column_wrapper strings({"this\t is that", "is then \tis", "us them is us"});
  cudf::test::strings_column_wrapper targets({"is", "us"});
  cudf::test::strings_column_wrapper repls({"_"});
  cudf::test::strings_column_wrapper expected({"this\t _ that", "_ then \t_", "_ them _ _"});

  auto results = nvtext::replace_tokens(cudf::strings_column_view(strings),
                                        cudf::strings_column_view(targets),
                                        cudf::strings_column_view(repls));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextReplaceTest, ReplaceTokensEmptyTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::strings_column_view strings_view(strings->view());
  auto const results = nvtext::replace_tokens(strings_view, strings_view, strings_view);
  EXPECT_EQ(results->size(), 0);
  EXPECT_EQ(results->has_nulls(), false);
}

TEST_F(TextReplaceTest, ReplaceTokensLongStrings)
{
  cudf::test::strings_column_wrapper input{
    "pellentesque ut euismod semo phaselus tristiut libero ut dui congusem non pellentesque nunc ",
    "pellentesque ut euismod se phaselus tristiut libero ut dui congusem non pellentesque ",
    "pellentesque ut euismod phaselus tristiut libero ut dui congusem non pellentesque nun ",
    "pellentesque ut euismod seem phaselus tristiut libero ut dui congusem non pellentesque un "};
  cudf::test::strings_column_wrapper targets({"ut", "pellentesque"});
  cudf::test::strings_column_wrapper repls({"___", "é"});

  auto expected = cudf::test::strings_column_wrapper{
    "é ___ euismod semo phaselus tristiut libero ___ dui congusem non é nunc ",
    "é ___ euismod se phaselus tristiut libero ___ dui congusem non é ",
    "é ___ euismod phaselus tristiut libero ___ dui congusem non é nun ",
    "é ___ euismod seem phaselus tristiut libero ___ dui congusem non é un "};

  auto results = nvtext::replace_tokens(cudf::strings_column_view(input),
                                        cudf::strings_column_view(targets),
                                        cudf::strings_column_view(repls));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextReplaceTest, ReplaceTokensErrorTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::strings_column_view strings_view(strings->view());
  cudf::test::strings_column_wrapper notnulls({"", "", ""});
  cudf::strings_column_view notnulls_view(notnulls);
  cudf::test::strings_column_wrapper nulls({"", ""}, {false, false});
  cudf::strings_column_view nulls_view(nulls);

  EXPECT_THROW(nvtext::replace_tokens(strings_view, nulls_view, notnulls_view), cudf::logic_error);
  EXPECT_THROW(nvtext::replace_tokens(strings_view, notnulls_view, nulls_view), cudf::logic_error);
  EXPECT_THROW(nvtext::replace_tokens(notnulls_view, notnulls_view, strings_view),
               cudf::logic_error);
  EXPECT_THROW(
    nvtext::replace_tokens(notnulls_view, nulls_view, strings_view, cudf::string_scalar("", false)),
    cudf::logic_error);
}

TEST_F(TextReplaceTest, FilterTokens)
{
  cudf::test::strings_column_wrapper strings({" one two three ", "four  fivé  six", "sevén eight"});

  auto results = nvtext::filter_tokens(cudf::strings_column_view(strings), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);  // no change

  {
    auto results = nvtext::filter_tokens(cudf::strings_column_view(strings), 4);
    cudf::test::strings_column_wrapper expected({"   three ", "four  fivé  ", "sevén eight"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = nvtext::filter_tokens(cudf::strings_column_view(strings), 5);
    cudf::test::strings_column_wrapper expected({"   three ", "    ", "sevén eight"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results =
      nvtext::filter_tokens(cudf::strings_column_view(strings), 4, cudf::string_scalar("--"));
    cudf::test::strings_column_wrapper expected({" -- -- three ", "four  fivé  --", "sevén eight"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextReplaceTest, FilterTokensEmptyTest)
{
  auto strings       = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto const results = nvtext::filter_tokens(cudf::strings_column_view(strings->view()), 7);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(TextReplaceTest, FilterTokensErrorTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  EXPECT_THROW(nvtext::filter_tokens(
                 cudf::strings_column_view(strings->view()), 1, cudf::string_scalar("", false)),
               cudf::logic_error);
  EXPECT_THROW(nvtext::filter_tokens(cudf::strings_column_view(strings->view()),
                                     1,
                                     cudf::string_scalar("-"),
                                     cudf::string_scalar("", false)),
               cudf::logic_error);
}
