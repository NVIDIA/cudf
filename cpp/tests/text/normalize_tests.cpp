/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/normalize.hpp>

#include <cuda/iterator>

#include <vector>

struct TextNormalizeTest : public cudf::test::BaseFixture {};

TEST_F(TextNormalizeTest, NormalizeSpaces)
{
  std::vector<char const*> h_strings{"the\t fox  jumped over the      dog",
                                     "the dog\f chased  the cat\r",
                                     " the cat  chaséd  the mouse\n",
                                     nullptr,
                                     "",
                                     " \r\t\n",
                                     "no change",
                                     "the mousé ate the cheese"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));

  cudf::strings_column_view strings_view(strings);

  std::vector<char const*> h_expected{"the fox jumped over the dog",
                                      "the dog chased the cat",
                                      "the cat chaséd the mouse",
                                      nullptr,
                                      "",
                                      "",
                                      "no change",
                                      "the mousé ate the cheese"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto const results = nvtext::normalize_spaces(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextNormalizeTest, NormalizeEmptyTest)
{
  auto strings = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  cudf::strings_column_view strings_view(strings->view());
  auto results = nvtext::normalize_spaces(strings_view);
  EXPECT_EQ(results->size(), 0);

  auto normalizer = nvtext::create_character_normalizer(true);
  results         = nvtext::normalize_characters(strings_view, *normalizer);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(TextNormalizeTest, AllNullStrings)
{
  cudf::test::strings_column_wrapper strings({"", "", ""}, {false, false, false});
  cudf::strings_column_view strings_view(strings);
  auto results = nvtext::normalize_spaces(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);

  auto normalizer = nvtext::create_character_normalizer(true);
  results         = nvtext::normalize_characters(strings_view, *normalizer);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);
}

TEST_F(TextNormalizeTest, SomeNullStrings)
{
  cudf::test::strings_column_wrapper strings({"", ".", "a"}, {false, true, true});
  cudf::strings_column_view strings_view(strings);
  cudf::test::strings_column_wrapper expected({"", " . ", "a"}, {false, true, true});

  auto normalizer = nvtext::create_character_normalizer(true);
  auto results    = nvtext::normalize_characters(strings_view, *normalizer);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextNormalizeTest, WithNormalizer)
{
  auto long_row =
    "this entry is intended to pad out past 256 bytes which is currently the block size";
  // the following include punctuation, accents, whitespace, and CJK characters
  auto input = cudf::test::strings_column_wrapper({"abc£def",
                                                   "",
                                                   "éè â îô\taeio",
                                                   "\tĂĆĖÑ  Ü",
                                                   "ACEN U",
                                                   "P^NP",
                                                   "$41.07",
                                                   "[a,b]",
                                                   "丏丟",
                                                   "",
                                                   long_row,
                                                   long_row,
                                                   long_row},
                                                  {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto const sv = cudf::strings_column_view(input);

  auto normalizer = nvtext::create_character_normalizer(true);
  auto results    = nvtext::normalize_characters(sv, *normalizer);
  auto expected   = cudf::test::strings_column_wrapper({"abc£def",
                                                        "",
                                                        "ee a io aeio",
                                                        " acen  u",
                                                        "acen u",
                                                        "p ^ np",
                                                        " $ 41 . 07",
                                                        " [ a , b ] ",
                                                        " 丏  丟 ",
                                                        "",
                                                        long_row,
                                                        long_row,
                                                        long_row},
                                                       {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::normalize_characters(sv, *normalizer);  // test normalizer re-use
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  normalizer = nvtext::create_character_normalizer(false);
  results    = nvtext::normalize_characters(sv, *normalizer);
  expected   = cudf::test::strings_column_wrapper({"abc£def",
                                                   "",
                                                   "éè â îô aeio",
                                                   " ĂĆĖÑ  Ü",
                                                   "ACEN U",
                                                   "P ^ NP",
                                                   " $ 41 . 07",
                                                   " [ a , b ] ",
                                                   " 丏  丟 ",
                                                   "",
                                                   long_row,
                                                   long_row,
                                                   long_row},
                                                  {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::normalize_characters(sv, *normalizer);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextNormalizeTest, SpecialTokens)
{
  auto long_row =
    "this entry is intended to pad out past 256 bytes which is currently the block size";
  auto input =
    cudf::test::strings_column_wrapper({"[BOS]Some strings with [PAD] special[SEP]tokens[EOS]",
                                        "[bos]these should[sep]work for lowercase[eos]",
                                        "some[non]tokens[eol]too",
                                        long_row,
                                        long_row,
                                        long_row});

  auto sv             = cudf::strings_column_view(input);
  auto special_tokens = cudf::test::strings_column_wrapper({"[BOS]", "[EOS]", "[SEP]", "[PAD]"});
  auto stv            = cudf::strings_column_view(special_tokens);

  auto normalizer = nvtext::create_character_normalizer(true, stv);
  auto results    = nvtext::normalize_characters(sv, *normalizer);
  auto expected   = cudf::test::strings_column_wrapper(
    {" [BOS] some strings with  [PAD]  special [SEP] tokens [EOS] ",
       " [BOS] these should [SEP] work for lowercase [EOS] ",
       "some [ non ] tokens [ eol ] too",
       long_row,
       long_row,
       long_row});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::normalize_characters(sv, *normalizer);  // and again
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  normalizer = nvtext::create_character_normalizer(false, stv);
  results    = nvtext::normalize_characters(sv, *normalizer);
  expected   = cudf::test::strings_column_wrapper(
    {" [BOS] Some strings with  [PAD]  special [SEP] tokens [EOS] ",
       " [ bos ] these should [ sep ] work for lowercase [ eos ] ",
       "some [ non ] tokens [ eol ] too",
       long_row,
       long_row,
       long_row});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::normalize_characters(sv, *normalizer);  // and again
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextNormalizeTest, NormalizeCharactersStripAccents)
{
  // Tests normalize_flags::STRIP_ACCENTS via the flags overload of normalize_characters.
  // Row 2: NFD form — plain 'a' followed by U+0301 (combining acute accent) followed by 'b'.
  // Constructed via char casts to avoid source-encoding ambiguity with precomposed á (U+00E1).
  char const u0301[]        = {(char)0xcc, (char)0x81, 0};  // UTF-8 for U+0301
  std::string const nfd_str = std::string("a") + u0301 + "b";
  auto input                = cudf::test::strings_column_wrapper({
    "éàïü",   // lowercase precomposed NFC accented
    "ÉÀÏÜ",   // uppercase precomposed NFC accented
    nfd_str,  // NFD: a + U+0301 (combining acute accent) + b
    "ACENU",  // no accents
  });
  auto sv                   = cudf::strings_column_view(input);

  // strip_accents=true, do_lower_case=false: de-accent while preserving case
  auto normalizer = nvtext::create_character_normalizer(false);
  auto flags    = nvtext::normalize_flags::STRIP_ACCENTS | nvtext::normalize_flags::PAD_PUNCTUATION;
  auto results  = nvtext::normalize_characters(sv, *normalizer, flags);
  auto expected = cudf::test::strings_column_wrapper({"eaiu", "EAIU", "ab", "ACENU"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  // strip_accents=true, do_lower_case=true: same result as do_lower_case=true alone
  normalizer = nvtext::create_character_normalizer(true);
  results    = nvtext::normalize_characters(sv, *normalizer, flags);
  expected   = cudf::test::strings_column_wrapper({"eaiu", "eaiu", "ab", "acenu"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextNormalizeTest, NormalizeCharactersNoTokenizePunctuation)
{
  // Tests normalize_flags::NONE / pad_punctuation=false.
  // Punctuation and CJK characters should not receive padding spaces.
  // Whitespace normalization (tab→space via ALWAYS_REPLACE) is unaffected.
  auto input = cudf::test::strings_column_wrapper({"P^NP", "$41.07", "[a,b]", "丏丟", "éè\tâ"});
  auto sv    = cudf::strings_column_view(input);

  // no pad_punctuation, do_lower_case=false: chars pass through unchanged
  auto normalizer = nvtext::create_character_normalizer(false);
  auto results    = nvtext::normalize_characters(sv, *normalizer, nvtext::normalize_flags::NONE);
  auto expected   = cudf::test::strings_column_wrapper({"P^NP", "$41.07", "[a,b]", "丏丟", "éè â"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  // no pad_punctuation, do_lower_case=true: lowercase + de-accent, no punct padding
  normalizer = nvtext::create_character_normalizer(true);
  results    = nvtext::normalize_characters(sv, *normalizer, nvtext::normalize_flags::NONE);
  expected   = cudf::test::strings_column_wrapper({"p^np", "$41.07", "[a,b]", "丏丟", "ee a"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(TextNormalizeTest, NormalizeSlicedColumn)
{
  cudf::test::strings_column_wrapper strings(
    {"abc£def", "éè â îô\taeio", "ACEN U", "P^NP", "$41.07", "[a,b]", "丏丟"});

  std::vector<cudf::column_view> sliced = cudf::split(strings, {4});

  auto normalizer = nvtext::create_character_normalizer(true);
  auto results =
    nvtext::normalize_characters(cudf::strings_column_view(sliced.front()), *normalizer);
  auto expected =
    cudf::test::strings_column_wrapper({"abc£def", "ee a io aeio", "acen u", "p ^ np"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  normalizer = nvtext::create_character_normalizer(false);
  results    = nvtext::normalize_characters(cudf::strings_column_view(sliced[1]), *normalizer);
  expected   = cudf::test::strings_column_wrapper({" $ 41 . 07", " [ a , b ] ", " 丏  丟 "});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}
