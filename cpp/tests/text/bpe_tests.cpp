/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/text/bpe_data_generator.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/byte_pair_encoding.hpp>

#include <rmm/device_buffer.hpp>

#include <random>
#include <string>
#include <unordered_set>
#include <vector>

struct TextBytePairEncoding : public cudf::test::BaseFixture {};

TEST_F(TextBytePairEncoding, BytePairEncoding)
{
  // partial table based on values from https://huggingface.co/gpt2/raw/main/merges.txt
  auto mpt = cudf::test::strings_column_wrapper({
    "e n",    // 14
    "i t",    // 16
    "i s",    // 17
    "e s",    // 20
    "en t",   // 44
    "c e",    // 90
    "es t",   // 141
    "en ce",  // 340
    "t h",    // 146
    "h i",    // 5049
    "th is",  // 5407
    "t est",  // 9034
    "s i",    // 13142
    "s ent"   // 33832
  });

  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  auto validity = cudf::test::iterators::null_at(4);
  cudf::test::strings_column_wrapper input(
    {"thisisit", "thisis test-sentence-1", "thisistestsentence-2", "this-istestsentence 3", "", ""},
    validity);
  auto sv = cudf::strings_column_view(input);

  auto results  = nvtext::byte_pair_encoding(sv, *merge_pairs);
  auto expected = cudf::test::strings_column_wrapper({"this is it",
                                                      "this is   test - sent ence - 1",
                                                      "this is test sent ence - 2",
                                                      "this - is test sent ence   3",
                                                      "",
                                                      ""},
                                                     validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto sliced          = cudf::slice(input, {1, 4}).front();
  auto sliced_expected = cudf::slice(expected, {1, 4}).front();

  sv      = cudf::strings_column_view(sliced);
  results = nvtext::byte_pair_encoding(sv, *merge_pairs);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), sliced_expected);
}

TEST_F(TextBytePairEncoding, BytePairEncodingSeparator)
{
  auto mpt = cudf::test::strings_column_wrapper(
    {"Ġ t", "Ġt he", "h e", "e n", "i t", "e s", "en t", "c e", "es t", "en ce", "t est", "s ent"});

  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  cudf::test::strings_column_wrapper input(
    {"Ġthe test sentence", "test Ġthe sentence", "Ġthetest sentence", "testĠthesentence"});
  auto sv = cudf::strings_column_view(input);

  auto results = nvtext::byte_pair_encoding(sv, *merge_pairs, std::string_view("$"));

  auto expected = cudf::test::strings_column_wrapper({"Ġthe$ $test$ $sent$ence",
                                                      "test$ $Ġthe$ $sent$ence",
                                                      "Ġthe$test$ $sent$ence",
                                                      "test$Ġthe$sent$ence"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(TextBytePairEncoding, BPEAdjacentPairs)
{
  auto mpt         = cudf::test::strings_column_wrapper({
    "▁ H",    //    157
    "m m",    //  10742
    "? !",    //  50675
    "▁H mm",  // 174381
    "mm m",   // 262776
    "?! !",   // 352313
    "? !?",   // 352314
    "mm mm",  // 387733
    "▁H m",   // 471269
    "?! ?!",  // 506981
    "?!? !",  // 506982
  });
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  cudf::test::strings_column_wrapper input({"▁Hmmmmm", "?!?!?!"});

  auto results  = nvtext::byte_pair_encoding(cudf::strings_column_view(input), *merge_pairs);
  auto expected = cudf::test::strings_column_wrapper({"▁Hmm mmm", "?!?! ?!"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(TextBytePairEncoding, BPE_Empty)
{
  auto mpt         = cudf::test::strings_column_wrapper({"i s", "i t"});
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));
  auto empty       = cudf::make_empty_column(cudf::type_id::STRING);
  auto results = nvtext::byte_pair_encoding(cudf::strings_column_view(empty->view()), *merge_pairs);
  EXPECT_EQ(0, results->size());
}

TEST_F(TextBytePairEncoding, BPE_AllEmptyRows)
{
  auto mpt         = cudf::test::strings_column_wrapper({"i s", "i t"});
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));
  auto input       = cudf::test::strings_column_wrapper({"", "", ""}, {true, false, true});
  auto results     = nvtext::byte_pair_encoding(cudf::strings_column_view(input), *merge_pairs);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), input);
}

TEST_F(TextBytePairEncoding, BPE_Error)
{
  auto empty = cudf::make_empty_column(cudf::type_id::STRING);
  EXPECT_THROW(nvtext::load_merge_pairs(cudf::strings_column_view(*empty)), cudf::logic_error);
  auto null_pairs = cudf::test::strings_column_wrapper({"", ""}, {true, false});
  EXPECT_THROW(nvtext::load_merge_pairs(cudf::strings_column_view(null_pairs)), cudf::logic_error);
  auto duplicates = cudf::test::strings_column_wrapper({"a b", "c d", "ab c", "c d", "e f"});
  EXPECT_THROW(nvtext::load_merge_pairs(cudf::strings_column_view(duplicates)),
               std::invalid_argument);
  // same strings split differently are not duplicates
  auto not_duplicates = cudf::test::strings_column_wrapper({"a bc", "ab c", "abc d"});
  EXPECT_NO_THROW(nvtext::load_merge_pairs(cudf::strings_column_view(not_duplicates)));
}

// ---------------------------------------------------------------------------------------------
// Reference-based tests
// ---------------------------------------------------------------------------------------------
namespace {

namespace bpe = cudf::test::bpe;

// builds a random but well-formed table: every half exists before it is used
std::vector<std::string> random_merges(std::vector<std::string> const& alphabet,
                                       int count,
                                       std::mt19937& rng,
                                       std::size_t max_token = 8)
{
  std::vector<std::string> vocab(alphabet);
  std::vector<std::string> pairs;
  std::unordered_set<std::string> seen;
  while (static_cast<int>(pairs.size()) < count) {
    // bias toward recently created tokens like real tables
    std::uniform_int_distribution<std::size_t> pick(0, vocab.size() - 1);
    auto const& l = vocab[std::max(pick(rng), pick(rng))];
    auto const& r = vocab[std::max(pick(rng), pick(rng))];
    if (l.size() + r.size() > max_token || !seen.insert(l + " " + r).second) { continue; }
    pairs.push_back(l + " " + r);
    vocab.push_back(l + r);
  }
  return pairs;
}

std::unique_ptr<cudf::column> reference_column(std::vector<std::string> const& merges,
                                               std::vector<std::string> const& rows,
                                               std::vector<bool> const& validity = {})
{
  auto const encoder = bpe::reference_encoder(merges);
  std::vector<std::string> expected;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    auto const valid = validity.empty() || validity[i];
    expected.push_back(valid ? encoder.encode(rows[i]) : std::string{});
  }
  if (validity.empty()) {
    return cudf::test::strings_column_wrapper(expected.begin(), expected.end()).release();
  }
  return cudf::test::strings_column_wrapper(expected.begin(), expected.end(), validity.begin())
    .release();
}

void check_against_reference(std::vector<std::string> const& merges,
                             std::vector<std::string> const& rows)
{
  auto mpt      = cudf::test::strings_column_wrapper(merges.begin(), merges.end());
  auto mp       = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));
  auto input    = cudf::test::strings_column_wrapper(rows.begin(), rows.end());
  auto results  = nvtext::byte_pair_encoding(cudf::strings_column_view(input), *mp);
  auto expected = reference_column(merges, rows);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected->view());
}

// splits text into rows of approximately `width` bytes on UTF-8 character boundaries
std::vector<std::string> split_rows(std::string const& text, std::size_t width)
{
  std::vector<std::string> rows;
  std::size_t pos = 0;
  while (pos < text.size()) {
    auto end = std::min(pos + width, text.size());
    while (end < text.size() && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) {
      ++end;
    }
    rows.push_back(text.substr(pos, end - pos));
    pos = end;
  }
  return rows;
}

// shared synthetic corpus and trained merges table
struct synthetic_data {
  std::vector<std::string> merges;
  std::string prose;  // raw text
  std::string code;   // raw code-like text
};

synthetic_data const& get_synthetic_data()
{
  static synthetic_data const data = [] {
    bpe::text_generator gen(7);
    synthetic_data d;
    d.prose  = gen.generate(400'000);
    d.code   = gen.generate(100'000, true);
    d.merges = bpe::train_merges(d.prose + d.code, 8000);
    return d;
  }();
  return data;
}

}  // namespace

TEST_F(TextBytePairEncoding, ReferenceEncoder)
{
  // sanity check the reference implementation against the hand-verified results above
  auto const encoder = bpe::reference_encoder({"e n",
                                               "i t",
                                               "i s",
                                               "e s",
                                               "en t",
                                               "c e",
                                               "es t",
                                               "en ce",
                                               "t h",
                                               "h i",
                                               "th is",
                                               "t est",
                                               "s i",
                                               "s ent"});
  EXPECT_EQ(encoder.encode("thisis test-sentence-1"), "this is   test - sent ence - 1");
  auto const adjacent = bpe::reference_encoder(
    {"▁ H", "m m", "? !", "▁H mm", "mm m", "?! !", "? !?", "mm mm", "▁H m", "?! ?!", "?!? !"});
  EXPECT_EQ(adjacent.encode("▁Hmmmmm"), "▁Hmm mmm");
  EXPECT_EQ(adjacent.encode("?!?!?!"), "?!?! ?!");
}

TEST_F(TextBytePairEncoding, RandomAgainstReference)
{
  std::mt19937 rng(12345);
  // includes multi-byte characters; ' ' and '-' never appear in the table
  std::vector<std::string> alphabet{"a", "b", "c", "d", "e", "é", "ü", "中"};
  for (int trial = 0; trial < 5; ++trial) {
    auto pairs = random_merges(alphabet, 60, rng);
    std::vector<std::string> rows;
    std::uniform_int_distribution<int> len_dist(0, 600);
    std::uniform_int_distribution<int> ch_dist(0, static_cast<int>(alphabet.size()) + 1);
    for (int i = 0; i < 300; ++i) {
      std::string s;
      auto const len = len_dist(rng);
      for (int j = 0; j < len; ++j) {
        auto c = ch_dist(rng);
        s += c < static_cast<int>(alphabet.size()) ? alphabet[c] : (c & 1 ? " " : "-");
      }
      rows.push_back(s);
    }
    check_against_reference(pairs, rows);
  }
}

TEST_F(TextBytePairEncoding, SyntheticText)
{
  auto const& data = get_synthetic_data();
  // raw text: spaces/newlines are not in the table (GPT-2 table with un-mapped input)
  check_against_reference(data.merges, split_rows(data.prose, 1000));
  // GPT-2 byte-level mapped text (Ġ/Ċ) with varying row sizes
  auto const mapped = bpe::gpt2_byte_map(data.prose);
  check_against_reference(data.merges, split_rows(mapped.substr(0, 200'000), 100));
  check_against_reference(data.merges, split_rows(mapped, 5000));
  check_against_reference(data.merges, split_rows(mapped, 100'000));
}

TEST_F(TextBytePairEncoding, SyntheticCode)
{
  // code-like text has long runs of (mapped) spaces which are mergeable with each other
  auto const& data = get_synthetic_data();
  check_against_reference(data.merges, split_rows(bpe::gpt2_byte_map(data.code), 2000));
}

TEST_F(TextBytePairEncoding, SyntheticNoSpaces)
{
  // no whitespace: fewer natural word boundaries
  auto const& data = get_synthetic_data();
  auto text        = data.prose.substr(0, 100'000);
  std::erase_if(text, [](char c) { return c == ' ' || c == '\n'; });
  check_against_reference(data.merges, split_rows(bpe::gpt2_byte_map(text), 4000));
}

TEST_F(TextBytePairEncoding, SyntheticDuplicateLongSegments)
{
  // rows repeating the same text so long segments (and those beyond the
  // de-duplication size limit) occur many times
  auto const& data = get_synthetic_data();
  auto text        = data.prose.substr(0, 20'000);
  std::erase_if(text, [](char c) { return c == ' ' || c == '\n'; });
  auto const mapped = bpe::gpt2_byte_map(text);
  std::vector<std::string> rows;
  for (int i = 0; i < 20; ++i) {
    rows.push_back(mapped.substr(0, 3000));
    rows.push_back(std::string(5000 + (i % 2), 'a') + mapped.substr(100, 500));
  }
  check_against_reference(data.merges, rows);
}

TEST_F(TextBytePairEncoding, SyntheticSlicedWithNulls)
{
  auto const& data = get_synthetic_data();
  auto const rows  = split_rows(bpe::gpt2_byte_map(data.prose.substr(0, 50'000)), 300);
  std::vector<bool> validity(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    validity[i] = (i % 7) != 3;
  }
  // nulls created this way have no chars data
  auto input    = cudf::test::strings_column_wrapper(rows.begin(), rows.end(), validity.begin());
  auto mpt      = cudf::test::strings_column_wrapper(data.merges.begin(), data.merges.end());
  auto mp       = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));
  auto expected = reference_column(data.merges, rows, validity);

  auto results = nvtext::byte_pair_encoding(cudf::strings_column_view(input), *mp);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected->view());

  auto const bounds = std::vector<cudf::size_type>{5, 77};
  auto sliced       = cudf::slice(input, bounds).front();
  results           = nvtext::byte_pair_encoding(cudf::strings_column_view(sliced), *mp);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(),
                                      cudf::slice(expected->view(), bounds).front());
}

TEST_F(TextBytePairEncoding, LongRunsOfSameCharacter)
{
  // "a a" overlapping same-rank pairs: exercises the adjacent equal-rank deferral logic
  std::vector<std::string> pairs{"a a", "aa a", "aa aa", "aaaa aaaa", "b a"};
  std::vector<std::string> rows;
  for (int n : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 100, 1000, 5000}) {
    rows.push_back(std::string(n, 'a'));
    rows.push_back("b" + std::string(n, 'a'));
  }
  check_against_reference(pairs, rows);
}

TEST_F(TextBytePairEncoding, SingleLongRow)
{
  std::mt19937 rng(777);
  std::vector<std::string> alphabet{"a", "b", "c", "d", "e", "f"};
  auto pairs = random_merges(alphabet, 100, rng, 12);
  std::string s;
  std::uniform_int_distribution<int> ch(0, 5);
  for (int i = 0; i < 20000; ++i) {
    s += alphabet[ch(rng)];
  }
  check_against_reference(pairs, {s});
}

TEST_F(TextBytePairEncoding, NonEmptyNulls)
{
  // null rows may contain chars data; output must be the same as with sanitized nulls
  auto mpt = cudf::test::strings_column_wrapper({"e n", "i t", "i s", "t h", "th is"});
  auto mp  = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  auto input    = cudf::test::strings_column_wrapper({"thisit", std::string(20000, 'x'), "isit"});
  auto contents = input.release()->release();
  // mark row 1 null without clearing its chars
  auto null_mask  = cudf::test::detail::make_null_mask(cudf::test::iterators::null_at(1),
                                                      cudf::test::iterators::null_at(1) + 3);
  auto with_nulls = cudf::make_strings_column(3,
                                              std::move(contents.children.front()),
                                              std::move(*contents.data),
                                              1,
                                              std::move(null_mask.first));

  auto results  = nvtext::byte_pair_encoding(cudf::strings_column_view(with_nulls->view()), *mp);
  auto expected = cudf::test::strings_column_wrapper({"this it", "", "is it"}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}
