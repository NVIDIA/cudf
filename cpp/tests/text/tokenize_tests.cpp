/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/tokenize.hpp>

#include <cuda/iterator>

#include <vector>

struct TextTokenizeTest : public cudf::test::BaseFixture {};

TEST_F(TextTokenizeTest, Tokenize)
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

  cudf::test::strings_column_wrapper expected{
    "the", "fox", "jumped", "over", "the",   "dog", "the",   "dog", "chased", "the",   "cat",
    "the", "cat", "chased", "the",  "mouse", "the", "mousé", "ate", "the",    "cheese"};

  auto results = nvtext::tokenize(strings_view, cudf::string_scalar(" "));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::tokenize(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  cudf::test::fixed_width_column_wrapper<int32_t> expected_counts{6, 5, 5, 0, 0, 5};
  results = nvtext::count_tokens(strings_view, cudf::string_scalar(": #"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_counts);
  results = nvtext::count_tokens(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_counts);
}

TEST_F(TextTokenizeTest, TokenizeMulti)
{
  std::vector<char const*> h_strings{"the fox jumped over the dog",
                                     "the dog chased  the cat",
                                     "the cat chased the mouse ",
                                     nullptr,
                                     "",
                                     "the over ",
                                     "the mousé ate the cheese"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  cudf::strings_column_view strings_view(strings);

  cudf::test::strings_column_wrapper delimiters{"the ", "over "};
  cudf::strings_column_view delimiters_view(delimiters);

  auto results = nvtext::tokenize(strings_view, delimiters_view);

  cudf::test::strings_column_wrapper expected{
    "fox jumped ", "dog", "dog chased  ", "cat", "cat chased ", "mouse ", "mousé ate ", "cheese"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_counts{2, 2, 2, 0, 0, 0, 2};
  results = nvtext::count_tokens(strings_view, delimiters_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_counts);
}

TEST_F(TextTokenizeTest, TokenizeErrorTest)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view strings_view(strings);

  {
    cudf::test::strings_column_wrapper delimiters;  // empty delimiters
    cudf::strings_column_view delimiters_view(delimiters);
    EXPECT_THROW(nvtext::tokenize(strings_view, delimiters_view), cudf::logic_error);
    EXPECT_THROW(nvtext::count_tokens(strings_view, delimiters_view), cudf::logic_error);
  }
  {
    cudf::test::strings_column_wrapper delimiters({"", ""}, {false, false});  // null delimiters
    cudf::strings_column_view delimiters_view(delimiters);
    EXPECT_THROW(nvtext::tokenize(strings_view, delimiters_view), cudf::logic_error);
    EXPECT_THROW(nvtext::count_tokens(strings_view, delimiters_view), cudf::logic_error);
  }
}

TEST_F(TextTokenizeTest, CharacterTokenize)
{
  cudf::test::strings_column_wrapper input({"the mousé ate", "the cheese", ""});

  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected{LCW{"t", "h", "e", " ", "m", "o", "u", "s", "é", " ", "a", "t", "e"},
               LCW{"t", "h", "e", " ", "c", "h", "e", "e", "s", "e"},
               LCW{}};

  auto results = nvtext::character_tokenize(cudf::strings_column_view(input));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextTokenizeTest, TokenizeEmptyTest)
{
  auto input = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto view  = cudf::strings_column_view(input->view());
  cudf::test::strings_column_wrapper all_empty_wrapper({"", "", ""});
  auto all_empty = cudf::strings_column_view(all_empty_wrapper);
  cudf::test::strings_column_wrapper all_null_wrapper({"", "", ""}, {false, false, false});
  auto all_null = cudf::strings_column_view(all_null_wrapper);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({0, 0, 0});

  auto results = nvtext::tokenize(view);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::tokenize(all_empty);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::tokenize(all_null);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::count_tokens(view);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::count_tokens(all_empty);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::count_tokens(cudf::strings_column_view(all_null));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::character_tokenize(view);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::character_tokenize(all_empty);
  EXPECT_EQ(results->size(), 0);
  auto const delimiter = cudf::string_scalar{""};
  results              = nvtext::tokenize_with_vocabulary(view, all_empty, delimiter);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::tokenize_with_vocabulary(all_null, all_empty, delimiter);
  EXPECT_EQ(results->size(), results->null_count());
}

TEST_F(TextTokenizeTest, Detokenize)
{
  cudf::test::strings_column_wrapper strings{
    "the", "fox", "jumped", "over",   "the", "dog",   "the", "dog",   "chased", "the",
    "cat", "the", "cat",    "chased", "the", "mouse", "the", "mousé", "ate",    "cheese"};

  {
    cudf::test::fixed_width_column_wrapper<int32_t> rows{0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                                         1, 2, 2, 2, 2, 2, 3, 3, 3, 3};
    auto results = nvtext::detokenize(cudf::strings_column_view(strings), rows);
    cudf::test::strings_column_wrapper expected{"the fox jumped over the dog",
                                                "the dog chased the cat",
                                                "the cat chased the mouse",
                                                "the mousé ate cheese"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int16_t> rows{0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                                         1, 2, 2, 2, 2, 2, 3, 3, 3, 0};
    auto results =
      nvtext::detokenize(cudf::strings_column_view(strings), rows, cudf::string_scalar("_"));
    cudf::test::strings_column_wrapper expected{"the_fox_jumped_over_the_dog_cheese",
                                                "the_dog_chased_the_cat",
                                                "the_cat_chased_the_mouse",
                                                "the_mousé_ate"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(TextTokenizeTest, DetokenizeErrors)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view strings_view(strings);

  cudf::test::fixed_width_column_wrapper<int32_t> one({0});
  cudf::test::fixed_width_column_wrapper<int32_t> none;

  EXPECT_THROW(nvtext::detokenize(strings_view, none), cudf::logic_error);
  EXPECT_THROW(nvtext::detokenize(strings_view, one, cudf::string_scalar("", false)),
               cudf::logic_error);
}

TEST_F(TextTokenizeTest, Vocabulary)
{
  cudf::test::strings_column_wrapper vocabulary(  // leaving out 'cat' on purpose
    {"ate", "chased", "cheese", "dog", "fox", "jumped", "mouse", "mousé", "over", "the"});
  auto vocab = nvtext::load_vocabulary(cudf::strings_column_view(vocabulary));

  auto validity = cudf::test::iterators::null_at(5);
  auto input    = cudf::test::strings_column_wrapper({" the fox jumped over the dog ",
                                                      " the dog chased the cat",
                                                      "",
                                                      "the cat chased the mouse ",
                                                      "the mousé  ate  cheese",
                                                      "",
                                                      "dog"},
                                                  validity);

  auto input_view = cudf::strings_column_view(input);
  auto delimiter  = cudf::string_scalar(" ");
  auto default_id = -7;  // should be the token for the missing 'cat'
  auto results    = nvtext::tokenize_with_vocabulary(input_view, *vocab, delimiter, default_id);

  using LCW = cudf::test::lists_column_wrapper<cudf::size_type>;
  // clang-format off
  LCW expected({LCW{ 9, 4, 5, 8, 9, 3},
                LCW{ 9, 3, 1, 9,-7},
                LCW{},
                LCW{ 9,-7, 1, 9, 6},
                LCW{ 9, 7, 0, 2},
                LCW{}, LCW{3}},
                validity);
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto sliced          = cudf::slice(input, {1, 4}).front();
  auto sliced_expected = cudf::slice(expected, {1, 4}).front();

  input_view = cudf::strings_column_view(sliced);

  results = nvtext::tokenize_with_vocabulary(input_view, *vocab, delimiter, default_id);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), sliced_expected);
}

TEST_F(TextTokenizeTest, VocabularyLongStrings)
{
  cudf::test::strings_column_wrapper vocabulary(
    {"ate", "chased", "cheese", "dog", "fox", "jumped", "mouse", "mousé", "over", "the"});
  auto vocab = nvtext::load_vocabulary(cudf::strings_column_view(vocabulary));

  std::vector<std::string> h_strings(
    4,
    "the fox jumped chased the dog cheese mouse at the over there dog mouse cat plus the horse "
    "jumped  over  the mousé  house with the dog  ");
  cudf::test::strings_column_wrapper input(h_strings.begin(), h_strings.end());
  auto input_view = cudf::strings_column_view(input);
  auto delimiter  = cudf::string_scalar(" ");
  auto default_id = -1;
  auto results    = nvtext::tokenize_with_vocabulary(input_view, *vocab, delimiter, default_id);

  using LCW = cudf::test::lists_column_wrapper<cudf::size_type>;
  // clang-format off
  LCW expected({LCW{ 9, 4, 5, 1, 9, 3, 2, 6, -1, 9, 8, -1, 3, 6, -1, -1, 9, -1, 5, 8, 9, 7, -1, -1, 9, 3},
                LCW{ 9, 4, 5, 1, 9, 3, 2, 6, -1, 9, 8, -1, 3, 6, -1, -1, 9, -1, 5, 8, 9, 7, -1, -1, 9, 3},
                LCW{ 9, 4, 5, 1, 9, 3, 2, 6, -1, 9, 8, -1, 3, 6, -1, -1, 9, -1, 5, 8, 9, 7, -1, -1, 9, 3},
                LCW{ 9, 4, 5, 1, 9, 3, 2, 6, -1, 9, 8, -1, 3, 6, -1, -1, 9, -1, 5, 8, 9, 7, -1, -1, 9, 3}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto sliced          = cudf::slice(input, {1, 3}).front();
  auto sliced_expected = cudf::slice(expected, {1, 3}).front();

  input_view = cudf::strings_column_view(sliced);

  results = nvtext::tokenize_with_vocabulary(input_view, *vocab, delimiter, default_id);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), sliced_expected);
}

TEST_F(TextTokenizeTest, TokenizeErrors)
{
  cudf::test::strings_column_wrapper empty{};
  cudf::strings_column_view view(empty);
  EXPECT_THROW(nvtext::load_vocabulary(view), cudf::logic_error);

  cudf::test::strings_column_wrapper vocab_nulls({""}, {false});
  cudf::strings_column_view nulls(vocab_nulls);
  EXPECT_THROW(nvtext::load_vocabulary(nulls), cudf::logic_error);

  cudf::test::strings_column_wrapper some{"hello"};
  auto vocab = nvtext::load_vocabulary(cudf::strings_column_view(some));
  EXPECT_THROW(nvtext::tokenize_with_vocabulary(view, *vocab, cudf::string_scalar("", false)),
               cudf::logic_error);
}
