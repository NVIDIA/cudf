/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/wordpiece_tokenize.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <vector>

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
  auto expected = LCW({{ 9, 7, 1, 4, 5, 8},
                       { 9, 6, 1, 3},
                       { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::wordpiece_tokenize(sv, *vocab, 5);
  // clang-format off
  expected = LCW({{ 9, 7, 1, 4, 5},
                  { 9, 6, 1, 3},
                  { 0, 1, 2, 3, 4}});
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
  auto expected = LCW({{4, 6, 3, 5, 7, 8, 2},
                       {1, 1, 6, 3, 1},
                       {1, 1}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  // max is applied to input words and not output tokens
  results = nvtext::wordpiece_tokenize(sv, *vocab, 6);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::wordpiece_tokenize(sv, *vocab, 4);
  // clang-format off
  expected = LCW({{4, 6, 3, 5, 7, 8},
                  {1, 1, 6, 3},
                  {1, 1}});
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
  auto expected = LCW({{ 9, 7, 1, 4, 5, 8},
                       { 9, 6, 1, 3},
                       { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::wordpiece_tokenize(sv, *vocab, 5);
  // clang-format off
  expected = LCW({{ 9, 7, 1, 4, 5},
                  { 9, 6, 1, 3},
                  { 0, 1, 2, 3, 4}});
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
  auto expected   = LCW({{}, {}, {}}, cudf::test::iterators::all_nulls());
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
  LCW expected({{}, {-1}, {-1}});  // -1 indicates [unk] not found in vocabulary
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

TEST(TextSubwordTest, WordPieceMaxWordsSmall)
{
  auto vocabulary = cudf::test::strings_column_wrapper({"[UNK]", "a", "b", "c", "d", "##d"});
  auto vocab      = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));
  auto input      = cudf::test::strings_column_wrapper({"a", "b c", " d", "", "  ", "dd a", "c  "});
  auto sv         = cudf::strings_column_view(input);

  using LCW     = cudf::test::lists_column_wrapper<cudf::size_type>;
  auto results  = nvtext::wordpiece_tokenize(sv, *vocab, 1);
  auto expected = LCW({LCW{1}, LCW{2}, LCW{4}, LCW{}, LCW{}, LCW{4, 5}, LCW{3}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = nvtext::wordpiece_tokenize(sv, *vocab, 2);
  expected = LCW({LCW{1}, LCW{2, 3}, LCW{4}, LCW{}, LCW{}, LCW{4, 5, 1}, LCW{3}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

namespace {

/**
 * @brief Host implementation of the wordpiece tokenizer used to verify the results
 */
struct host_wordpiece {
  std::map<std::string, cudf::size_type> vocab;
  std::map<std::string, cudf::size_type> sub_vocab;
  cudf::size_type unk_id;

  explicit host_wordpiece(std::vector<std::string> const& entries)
  {
    for (std::size_t i = 0; i < entries.size(); ++i) {
      auto const& entry = entries[i];
      vocab[entry]      = static_cast<cudf::size_type>(i);
      if (entry.size() >= 2 && entry[0] == '#' && entry[1] == '#') {
        sub_vocab[entry.substr(2)] = static_cast<cudf::size_type>(i);
      }
    }
    unk_id = vocab.count("[UNK]") ? vocab["[UNK]"] : -1;
  }

  void tokenize_word(std::string const& word, std::vector<cudf::size_type>& output) const
  {
    if (word.size() >= 200) {
      output.push_back(unk_id);
      return;
    }
    // longest prefix found in the vocabulary
    auto size = word.size();
    while (size > 0 && !vocab.count(word.substr(0, size))) {
      --size;
    }
    if (size == 0) {
      output.push_back(unk_id);
      return;
    }
    std::vector<cudf::size_type> tokens{vocab.at(word.substr(0, size))};
    // longest prefixes of the remaining characters found in the sub-word vocabulary
    auto rest = word.substr(size);
    while (!rest.empty()) {
      size = rest.size();
      while (size > 0 && !sub_vocab.count(rest.substr(0, size))) {
        --size;
      }
      if (size == 0) {
        tokens = {unk_id};
        break;
      }
      tokens.push_back(sub_vocab.at(rest.substr(0, size)));
      rest = rest.substr(size);
    }
    output.insert(output.end(), tokens.begin(), tokens.end());
  }

  std::vector<cudf::size_type> tokenize(std::string const& row, int max_words) const
  {
    std::vector<cudf::size_type> output;
    int count = 0;
    auto pos  = row.find_first_not_of(' ');
    while (pos != std::string::npos && (max_words == 0 || count < max_words)) {
      auto end = row.find(' ', pos);
      if (end == std::string::npos) { end = row.size(); }
      tokenize_word(row.substr(pos, end - pos), output);
      ++count;
      pos = row.find_first_not_of(' ', end);
    }
    return output;
  }
};

}  // namespace

TEST(TextSubwordTest, WordPieceMaxWords)
{
  std::mt19937 gen(7);
  auto random_int = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(gen); };
  std::string const letters = "abcdefghij";
  auto random_word          = [&](int lo, int hi) {
    std::string word;
    for (auto n = random_int(lo, hi); n > 0; --n) {
      word += letters[random_int(0, letters.size() - 1)];
    }
    return word;
  };

  // vocabulary of single characters, words, and sub-words
  std::vector<std::string> entries = {"[UNK]", "[PAD]"};
  for (auto ch : letters) {
    entries.emplace_back(1, ch);
  }
  for (auto ch : std::string("abcdefgh")) {
    entries.push_back(std::string("##") + ch);
  }
  for (int i = 0; i < 300; ++i) {
    entries.push_back(random_word(2, 6));
  }
  for (int i = 0; i < 100; ++i) {
    entries.push_back("##" + random_word(2, 4));
  }
  std::sort(entries.begin() + 2, entries.end());
  entries.erase(std::unique(entries.begin() + 2, entries.end()), entries.end());
  auto vocabulary = cudf::test::strings_column_wrapper(entries.begin(), entries.end());
  auto vocab      = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));
  auto const host = host_wordpiece(entries);

  // rows with a mix of empty rows, extra spaces, unknown characters, and long words
  std::vector<std::string> rows(5000);
  std::vector<bool> valids(rows.size());
  for (std::size_t r = 0; r < rows.size(); ++r) {
    valids[r]       = random_int(0, 49) != 0;
    auto const kind = random_int(0, 19);
    if (kind == 0) { continue; }
    if (kind == 1) {
      rows[r] = std::string(random_int(1, 4), ' ');
      continue;
    }
    if (random_int(0, 3) == 0) { rows[r] += std::string(random_int(1, 3), ' '); }
    auto const num_words = random_int(1, kind < 10 ? 8 : 120);
    for (int i = 0; i < num_words; ++i) {
      auto const category = random_int(0, 99);
      if (category < 50) {
        rows[r] += entries[random_int(2, entries.size() - 1)];
      } else if (category < 70) {
        rows[r] += random_word(1, 12);
      } else if (category < 75) {
        rows[r] += random_word(1, 3) + "zz";  // not in the vocabulary
      } else if (category < 77) {
        rows[r] += random_word(195, 260);  // longer than max_word_size
      } else {
        rows[r] += random_word(1, 1);
      }
      rows[r] += std::string(random_int(0, 9) == 0 ? random_int(2, 4) : 1, ' ');
    }
    if (random_int(0, 1)) { rows[r].pop_back(); }
  }
  auto input = cudf::test::strings_column_wrapper(rows.begin(), rows.end(), valids.begin());

  auto expected_results = [&](int begin, int end, int max_words) {
    std::vector<cudf::size_type> offsets{0};
    std::vector<cudf::size_type> tokens;
    for (auto r = begin; r < end; ++r) {
      if (valids[r]) {
        auto const row_tokens = host.tokenize(rows[r], max_words);
        tokens.insert(tokens.end(), row_tokens.begin(), row_tokens.end());
      }
      offsets.push_back(static_cast<cudf::size_type>(tokens.size()));
    }
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(valids.begin() + begin, valids.begin() + end);
    return cudf::make_lists_column(
      end - begin,
      cudf::test::fixed_width_column_wrapper<cudf::size_type>(offsets.begin(), offsets.end())
        .release(),
      cudf::test::fixed_width_column_wrapper<cudf::size_type>(tokens.begin(), tokens.end())
        .release(),
      null_count,
      std::move(null_mask));
  };

  auto const slice_begin = 37;
  auto const slice_end   = 4001;
  auto sliced            = cudf::slice(input, {slice_begin, slice_end}).front();
  for (auto max_words : {0, 1, 3, 20, 200}) {
    auto results  = nvtext::wordpiece_tokenize(cudf::strings_column_view(input), *vocab, max_words);
    auto expected = expected_results(0, static_cast<int>(rows.size()), max_words);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, *expected);

    results  = nvtext::wordpiece_tokenize(cudf::strings_column_view(sliced), *vocab, max_words);
    expected = expected_results(slice_begin, slice_end, max_words);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, *expected);
  }
}
