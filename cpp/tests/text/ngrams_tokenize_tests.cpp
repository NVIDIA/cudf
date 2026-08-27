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

#include <nvtext/ngrams_tokenize.hpp>

#include <cuda/iterator>

#include <vector>

struct TextNgramsTokenizeTest : public cudf::test::BaseFixture {};

TEST_F(TextNgramsTokenizeTest, Tokenize)
{
  std::vector<char const*> h_strings{"the fox jumped over the dog",
                                     "the dog chased  the cat",
                                     " the cat chased the mouse ",
                                     nullptr,
                                     "",
                                     "the mousé ate the cheese"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));

  cudf::strings_column_view strings_view(strings);

  {
    cudf::test::strings_column_wrapper expected{"the_fox",
                                                "fox_jumped",
                                                "jumped_over",
                                                "over_the",
                                                "the_dog",
                                                "the_dog",
                                                "dog_chased",
                                                "chased_the",
                                                "the_cat",
                                                "the_cat",
                                                "cat_chased",
                                                "chased_the",
                                                "the_mouse",
                                                "the_mousé",
                                                "mousé_ate",
                                                "ate_the",
                                                "the_cheese"};
    auto results =
      nvtext::ngrams_tokenize(strings_view, 2, std::string_view(), std::string_view("_"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper expected{"the:fox:jumped",
                                                "fox:jumped:over",
                                                "jumped:over:the",
                                                "over:the:dog",
                                                "the:dog:chased",
                                                "dog:chased:the",
                                                "chased:the:cat",
                                                "the:cat:chased",
                                                "cat:chased:the",
                                                "chased:the:mouse",
                                                "the:mousé:ate",
                                                "mousé:ate:the",
                                                "ate:the:cheese"};
    auto results =
      nvtext::ngrams_tokenize(strings_view, 3, std::string_view{" "}, std::string_view{":"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::strings_column_wrapper expected{"the--fox--jumped--over",
                                                "fox--jumped--over--the",
                                                "jumped--over--the--dog",
                                                "the--dog--chased--the",
                                                "dog--chased--the--cat",
                                                "the--cat--chased--the",
                                                "cat--chased--the--mouse",
                                                "the--mousé--ate--the",
                                                "mousé--ate--the--cheese"};
    auto results =
      nvtext::ngrams_tokenize(strings_view, 4, std::string_view{" "}, std::string_view{"--"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextNgramsTokenizeTest, TokenizeOneGram)
{
  cudf::test::strings_column_wrapper strings{"aaa bbb", "  ccc  ddd  ", "eee"};
  cudf::strings_column_view strings_view(strings);
  auto const empty = cudf::string_scalar("");

  cudf::test::strings_column_wrapper expected{"aaa", "bbb", "ccc", "ddd", "eee"};
  auto results = nvtext::ngrams_tokenize(strings_view, 1, empty, empty);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextNgramsTokenizeTest, TokenizeEmptyTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::strings_column_view strings_view(strings->view());
  auto const empty = cudf::string_scalar("");
  auto results     = nvtext::ngrams_tokenize(strings_view, 2, empty, empty);
  EXPECT_EQ(results->size(), 0);
  EXPECT_EQ(results->has_nulls(), false);
}

TEST_F(TextNgramsTokenizeTest, TokenizeErrorTest)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view strings_view(strings);
  auto const empty = cudf::string_scalar("");
  EXPECT_THROW(nvtext::ngrams_tokenize(strings_view, 0, empty, empty), cudf::logic_error);
}
