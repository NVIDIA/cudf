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
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/wordpiece_tokenize.hpp>

struct TextSubwordTest : public cudf::test::BaseFixture {};

TEST(TextSubwordTest, WordPiece)
{
  auto vocabulary = cudf::test::strings_column_wrapper(
    {"ate", "brown", "cheese", "dog", "fox", "jumped", "lazy", "quick", "over", "the", "[UNK]"});
  auto vocab = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));

  auto input = cudf::test::strings_column_wrapper(
    {"the quick brown fox jumped over",
     "the  lazy  brown  dog",
     " ate brown cheese dog fox jumped lazy quick over the [UNK] "});
  auto sv      = cudf::strings_column_view(input);
  auto results = nvtext::wordpiece_tokenize(sv, *vocab);

  using LCW = cudf::test::lists_column_wrapper<cudf::size_type>;
  // clang-format off
  auto expected = LCW({LCW{ 9, 7, 1, 4, 5, 8},
                       LCW{ 9, 6, 1, 3},
                       LCW{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::wordpiece_tokenize(sv, *vocab, 5);
  // clang-format off
  expected = LCW({LCW{ 9, 7, 1, 4, 5},
                  LCW{ 9, 6, 1, 3},
                  LCW{ 0, 1, 2, 3, 4}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST(TextSubwordTest, WordPieceWithSubwords)
{
  auto vocabulary =
    cudf::test::strings_column_wrapper({"", "[UNK]", "!", "a", "I", "G", "have", "##P", "##U"});
  auto vocab = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));

  auto input =
    cudf::test::strings_column_wrapper({"I have a GPU ! ", "do not have a gpu", "no gpu"});
  auto sv      = cudf::strings_column_view(input);
  auto results = nvtext::wordpiece_tokenize(sv, *vocab);

  using LCW = cudf::test::lists_column_wrapper<cudf::size_type>;
  // clang-format off
  auto expected = LCW({LCW{4, 6, 3, 5, 7, 8, 2},
                       LCW{1, 1, 6, 3, 1},
                       LCW{1, 1}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  // max is applied to input words and not output tokens
  results = nvtext::wordpiece_tokenize(sv, *vocab, 6);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::wordpiece_tokenize(sv, *vocab, 4);
  // clang-format off
  expected = LCW({LCW{4, 6, 3, 5, 7, 8},
                  LCW{1, 1, 6, 3},
                  LCW{1, 1}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST(TextSubwordTest, WordPieceSliced)
{
  auto vocabulary = cudf::test::strings_column_wrapper(
    {"ate", "brown", "cheese", "dog", "fox", "jumped", "lazy", "quick", "over", "the", "[UNK]"});
  auto vocab = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));

  auto input = cudf::test::strings_column_wrapper(
    {" ate the cheese dog quick over  lazy day ",
     "the quick brown fox jumped over",
     "the  lazy  brown  dog",
     " ate brown cheese dog fox jumped lazy quick over the [UNK] ",
     " ate the cheese dog quick over  lazy day "});

  auto sliced  = cudf::slice(input, {1, 4});
  auto sv      = cudf::strings_column_view(sliced.front());
  auto results = nvtext::wordpiece_tokenize(sv, *vocab);

  using LCW = cudf::test::lists_column_wrapper<cudf::size_type>;
  // clang-format off
  auto expected = LCW({LCW{ 9, 7, 1, 4, 5, 8},
                       LCW{ 9, 6, 1, 3},
                       LCW{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::wordpiece_tokenize(sv, *vocab, 5);
  // clang-format off
  expected = LCW({LCW{ 9, 7, 1, 4, 5},
                  LCW{ 9, 6, 1, 3},
                  LCW{ 0, 1, 2, 3, 4}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST(TextSubwordTest, WordPieceEmpty)
{
  auto vocabulary = cudf::test::strings_column_wrapper({""});
  auto vocab      = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));
  auto input      = cudf::test::strings_column_wrapper();
  auto sv         = cudf::strings_column_view(input);
  auto results    = nvtext::wordpiece_tokenize(sv, *vocab);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::wordpiece_tokenize(sv, *vocab, 10);
  EXPECT_EQ(results->size(), 0);
}

TEST(TextSubwordTest, WordPieceAllNulls)
{
  auto vocabulary = cudf::test::strings_column_wrapper({""});
  auto vocab      = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));
  auto input      = cudf::test::strings_column_wrapper({"", "", ""}, {false, false, false});
  auto sv         = cudf::strings_column_view(input);
  auto results    = nvtext::wordpiece_tokenize(sv, *vocab);
  using LCW       = cudf::test::lists_column_wrapper<cudf::size_type>;
  auto expected   = LCW({LCW{}, LCW{}, LCW{}}, cudf::test::iterators::all_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::wordpiece_tokenize(sv, *vocab, 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST(TextSubwordTest, WordPieceNoTokens)
{
  auto vocabulary = cudf::test::strings_column_wrapper({"x"});
  auto vocab      = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));
  auto input      = cudf::test::strings_column_wrapper({"  ", " www ", "xxxx"});
  auto sv         = cudf::strings_column_view(input);
  auto results    = nvtext::wordpiece_tokenize(sv, *vocab);
  using LCW       = cudf::test::lists_column_wrapper<cudf::size_type>;
  LCW expected({LCW{}, LCW{-1}, LCW{-1}});  // -1 indicates [unk] not found in vocabulary
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::wordpiece_tokenize(sv, *vocab, 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST(TextSubwordTest, WordPieceErrors)
{
  auto empty = cudf::test::strings_column_wrapper();
  EXPECT_THROW(nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(empty)),
               std::invalid_argument);
  auto nulls = cudf::test::strings_column_wrapper({"", "", ""}, {false, false, false});
  EXPECT_THROW(nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(nulls)),
               std::invalid_argument);

  auto vocabulary = cudf::test::strings_column_wrapper({"x"});
  auto vocab      = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));
  auto input      = cudf::test::strings_column_wrapper({"  "});
  EXPECT_THROW(nvtext::wordpiece_tokenize(cudf::strings_column_view(input), *vocab, -1),
               std::invalid_argument);
}
