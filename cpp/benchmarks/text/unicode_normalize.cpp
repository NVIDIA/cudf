/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/unicode_normalize.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

/**
 * @brief Synthetic table in the 3-column UnicodeData.txt layout used to build the normalizer
 *
 * The real UnicodeData.txt has about 35K rows, but most of them have CCC 0 and no decomposition
 * and so do not affect normalization. This table has about 2.4K rows with a realistic number of
 * canonical composition pairs (about 1.1K vs about 940 in Unicode 15), multi-step canonical
 * decompositions, singleton decompositions, and several kinds of compatibility decompositions.
 */
struct synthetic_unicode_data {
  std::vector<std::string> codepoints;
  std::vector<int32_t> ccc;
  std::vector<std::string> decomps;

  std::map<char, std::vector<uint32_t>> latin_composites;  // ASCII letter -> its composites
  std::vector<uint32_t> greek_bases;
  std::vector<uint32_t> greek_composites;

  void add(uint32_t cp, int32_t cc, std::string decomp)
  {
    codepoints.push_back(hex(cp));
    ccc.push_back(cc);
    decomps.push_back(std::move(decomp));
  }

  static std::string hex(uint32_t cp)
  {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%04X", cp);
    return buffer;
  }
};

/// Approximates the Canonical_Combining_Class values of the U+0300-U+036F block
int32_t combining_mark_ccc(uint32_t cp)
{
  if (cp >= 0x0334 && cp <= 0x0338) { return 1; }
  if (cp == 0x0321 || cp == 0x0322 || cp == 0x0327 || cp == 0x0328) { return 202; }
  if (cp == 0x031B) { return 216; }
  if (cp == 0x0315 || cp == 0x031A) { return 232; }
  if (cp == 0x0345) { return 240; }
  if ((cp >= 0x0316 && cp <= 0x0319) || (cp >= 0x031C && cp <= 0x0320) ||
      (cp >= 0x0323 && cp <= 0x0326) || (cp >= 0x0329 && cp <= 0x0333) ||
      (cp >= 0x0339 && cp <= 0x033C) || (cp >= 0x0347 && cp <= 0x0349) || cp == 0x034D ||
      cp == 0x034E) {
    return 220;
  }
  return 230;
}

synthetic_unicode_data create_unicode_data(std::mt19937& gen)
{
  synthetic_unicode_data data;
  auto const hex = synthetic_unicode_data::hex;

  // combining marks (CGJ U+034F has CCC 0 and is left out)
  for (uint32_t cp = 0x0300; cp <= 0x036F; ++cp) {
    if (cp != 0x034F) { data.add(cp, combining_mark_ccc(cp), ""); }
  }
  // Hebrew points with distinct CCC values exercise canonical reordering
  for (uint32_t cp = 0x05B0; cp <= 0x05BC; ++cp) {
    data.add(cp, static_cast<int32_t>(cp - 0x05B0 + 10), "");
  }

  // codepoints assigned to synthetic precomposed characters
  std::vector<uint32_t> latin_slots;
  for (auto [first, last] : std::array<std::pair<uint32_t, uint32_t>, 3>{
         {{0x00C0, 0x024F}, {0x1E00, 0x1EFF}, {0x0400, 0x04FF}}}) {
    for (auto cp = first; cp <= last; ++cp) {
      latin_slots.push_back(cp);
    }
  }
  auto next_slot = latin_slots.begin();

  // single-mark Latin composites: 12 marks for each ASCII letter
  std::vector<uint32_t> marks = {0x0300,
                                 0x0301,
                                 0x0302,
                                 0x0303,
                                 0x0304,
                                 0x0306,
                                 0x0307,
                                 0x0308,
                                 0x0309,
                                 0x030A,
                                 0x030B,
                                 0x030C,
                                 0x0323,
                                 0x0327,
                                 0x0328,
                                 0x0331};
  std::vector<std::pair<uint32_t, uint32_t>> second_level;  // (composite, first mark)
  std::string const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  for (auto letter : letters) {
    std::shuffle(marks.begin(), marks.end(), gen);
    for (std::size_t m = 0; m < 12; ++m) {
      auto const cp = *next_slot++;
      data.add(cp, 0, hex(static_cast<uint32_t>(letter)) + " " + hex(marks[m]));
      data.latin_composites[letter].push_back(cp);
      if (marks[m] == 0x0302 || marks[m] == 0x0304 || marks[m] == 0x0306 || marks[m] == 0x0308) {
        second_level.emplace_back(cp, static_cast<uint32_t>(letter));
      }
    }
  }
  // two-mark Latin composites decompose in two steps (e.g. U+1EA4 -> U+00C2 U+0301)
  // and use second marks with the same CCC as the first so no reordering is needed
  std::vector<uint32_t> second_marks = {0x0300, 0x0301, 0x0303, 0x0309};
  for (auto [composite, letter] : second_level) {
    std::shuffle(second_marks.begin(), second_marks.end(), gen);
    for (std::size_t m = 0; m < 2 && next_slot != latin_slots.end(); ++m) {
      auto const cp = *next_slot++;
      data.add(cp, 0, hex(composite) + " " + hex(second_marks[m]));
      data.latin_composites[static_cast<char>(letter)].push_back(cp);
    }
  }

  // Greek composites
  for (uint32_t cp = 0x0391; cp <= 0x03C9; ++cp) {
    if (cp != 0x03A2 && (cp <= 0x03A9 || cp >= 0x03B1) && cp != 0x03C2) {
      data.greek_bases.push_back(cp);
    }
  }
  uint32_t greek_slot = 0x1F00;
  for (auto base : data.greek_bases) {
    for (auto mark : {0x0301u, 0x0313u, 0x0314u, 0x0342u}) {
      data.add(greek_slot, 0, hex(base) + " " + hex(mark));
      data.greek_composites.push_back(greek_slot++);
    }
  }

  // singleton canonical decompositions
  data.add(0x2126, 0, "03A9");  // OHM SIGN
  data.add(0x212A, 0, "004B");  // KELVIN SIGN
  data.add(0x212B, 0, hex(data.latin_composites['A'].front()));
  for (uint32_t cp = 0xF900; cp <= 0xFA2D; ++cp) {  // CJK compatibility ideographs
    data.add(cp, 0, hex(0x4E00 + ((cp - 0xF900) * 37) % 20000));
  }

  // compatibility decompositions
  for (uint32_t cp = 0xFF01; cp <= 0xFF5E; ++cp) {
    data.add(cp, 0, "<wide> " + hex(cp - 0xFF01 + 0x21));
  }
  for (uint32_t cp = 0xFF66; cp <= 0xFF9D; ++cp) {
    data.add(cp, 0, "<narrow> " + hex(cp - 0xFF66 + 0x30A1));
  }
  data.add(0xFB00, 0, "<compat> 0066 0066");
  data.add(0xFB01, 0, "<compat> 0066 0069");
  data.add(0xFB02, 0, "<compat> 0066 006C");
  data.add(0xFB03, 0, "<compat> 0066 0066 0069");
  data.add(0xFB04, 0, "<compat> 0066 0066 006C");
  for (uint32_t i = 1; i <= 20; ++i) {
    auto digits = std::to_string(i);
    std::string circled, parenthesized = "<compat> 0028";
    for (auto d : digits) {
      circled += (circled.empty() ? "" : " ") + hex(static_cast<uint32_t>(d));
      parenthesized += " " + hex(static_cast<uint32_t>(d));
    }
    data.add(0x2460 + i - 1, 0, "<circle> " + circled);
    data.add(0x2474 + i - 1, 0, parenthesized + " 0029");
  }
  data.add(0x2103, 0, "<compat> 00B0 0043");  // DEGREE CELSIUS
  data.add(0x00BD, 0, "<fraction> 0031 2044 0032");
  std::uniform_int_distribution<std::size_t> letter_dist(0, letters.size() - 1);
  for (uint32_t cp = 0x3380; cp <= 0x33DF; ++cp) {  // squared abbreviations
    std::string decomp = "<square>";
    for (int i = 0; i < 2 + static_cast<int>(cp % 3); ++i) {
      decomp += " " + hex(static_cast<uint32_t>(letters[letter_dist(gen)]));
    }
    data.add(cp, 0, decomp);
  }
  for (uint32_t cp = 0x1D400; cp < 0x1D400 + 13 * 52; ++cp) {  // mathematical alphanumerics
    data.add(cp, 0, "<font> " + hex(static_cast<uint32_t>(letters[(cp - 0x1D400) % 52])));
  }
  // Hangul compatibility jamo decompose to conjoining jamo that NFKC recomposes
  for (uint32_t i = 0; i < 19; ++i) {
    data.add(0x3131 + i, 0, "<compat> " + hex(0x1100 + i));
  }
  for (uint32_t i = 0; i < 21; ++i) {
    data.add(0x314F + i, 0, "<compat> " + hex(0x1161 + i));
  }
  return data;
}

void append_utf8(std::string& str, uint32_t cp)
{
  if (cp < 0x80) {
    str += static_cast<char>(cp);
  } else if (cp < 0x800) {
    str += static_cast<char>(0xC0 | (cp >> 6));
    str += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    str += static_cast<char>(0xE0 | (cp >> 12));
    str += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    str += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    str += static_cast<char>(0xF0 | (cp >> 18));
    str += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    str += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    str += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

/**
 * @brief Generates random words of different kinds of text
 */
struct word_generator {
  synthetic_unicode_data const& data;
  std::mt19937& gen;

  uint32_t uniform(uint32_t first, uint32_t last)
  {
    return std::uniform_int_distribution<uint32_t>(first, last)(gen);
  }
  bool chance(double probability) { return std::bernoulli_distribution(probability)(gen); }
  template <typename T>
  T pick(std::vector<T> const& values)
  {
    return values[uniform(0, static_cast<uint32_t>(values.size() - 1))];
  }

  std::string english()
  {
    // letter frequencies roughly following English text
    static std::string const letters =
      "eeeeeeeeeeeettttttttaaaaaaaaoooooooiiiiiiinnnnnnnsssssshhhhhhrrrrrrddddllllcccuuummwwffgg"
      "yyppbbvkjxqz";
    std::string word;
    auto const size = uniform(2, 9);
    for (uint32_t i = 0; i < size; ++i) {
      word += letters[uniform(0, static_cast<uint32_t>(letters.size() - 1))];
    }
    if (chance(0.1)) { word[0] = static_cast<char>(word[0] - 'a' + 'A'); }
    return word;
  }

  // precomposed (NFC) Latin letters mixed into English-like words
  std::string accented()
  {
    std::string word;
    for (auto ch : english()) {
      if (chance(0.3)) {
        append_utf8(word, pick(data.latin_composites.at(ch)));
      } else {
        word += ch;
      }
    }
    return word;
  }

  // letters followed by unordered runs of combining marks
  std::string decomposed()
  {
    std::string word;
    for (auto ch : english()) {
      word += ch;
      if (chance(0.3)) {
        auto const num_marks = uniform(1, 3);
        for (uint32_t i = 0; i < num_marks; ++i) {
          append_utf8(word, uniform(0x0300, 0x0333));
        }
      }
    }
    return word;
  }

  std::string greek()
  {
    std::string word;
    auto const size = uniform(3, 8);
    for (uint32_t i = 0; i < size; ++i) {
      append_utf8(word, chance(0.3) ? pick(data.greek_composites) : pick(data.greek_bases));
    }
    return word;
  }

  std::string from_range(uint32_t first, uint32_t last, uint32_t min_size, uint32_t max_size)
  {
    std::string word;
    auto const size = uniform(min_size, max_size);
    for (uint32_t i = 0; i < size; ++i) {
      append_utf8(word, uniform(first, last));
    }
    return word;
  }

  std::string cjk()
  {
    std::string word;
    auto const size = uniform(2, 4);
    for (uint32_t i = 0; i < size; ++i) {
      append_utf8(word, chance(0.1) ? uniform(0xF900, 0xFA2D) : uniform(0x4E00, 0x9FFF));
    }
    return word;
  }

  std::string width_variant()
  {
    return chance(0.5) ? from_range(0xFF21, 0xFF5A, 3, 6) : from_range(0xFF66, 0xFF9D, 3, 6);
  }

  std::string symbol()
  {
    switch (uniform(0, 5)) {
      case 0: return from_range(0xFB00, 0xFB04, 1, 1) + english();
      case 1: return from_range(0x2460, 0x2473, 1, 1);
      case 2: return from_range(0x2474, 0x2487, 1, 1);
      case 3: return from_range(0x3380, 0x33DF, 1, 1);
      case 4: return from_range(0x1D400, 0x1D400 + 13 * 52 - 1, 3, 6);
      default: return std::to_string(uniform(0, 40)) + "\xE2\x84\x83";  // degrees Celsius
    }
  }

  // Hebrew letters with up to 2 points in random order
  std::string hebrew()
  {
    std::string word;
    auto const size = uniform(3, 6);
    for (uint32_t i = 0; i < size; ++i) {
      append_utf8(word, uniform(0x05D0, 0x05EA));
      auto const num_points = uniform(0, 2);
      for (uint32_t p = 0; p < num_points; ++p) {
        append_utf8(word, uniform(0x05B0, 0x05BC));
      }
    }
    return word;
  }

  std::string compat_jamo()
  {
    std::string word;
    auto const size = uniform(1, 3);
    for (uint32_t i = 0; i < size; ++i) {
      append_utf8(word, uniform(0x3131, 0x3143));
      append_utf8(word, uniform(0x314F, 0x3163));
    }
    return word;
  }

  /**
   * @brief Returns a random word for the given text type
   *
   * "latin" is mostly ASCII with some precomposed accented letters and a few decomposed
   * sequences, which is enough to make NFC/NFKC run the full pipeline.
   * "mixed" includes all the word types including Hangul, CJK, and compatibility characters.
   */
  std::string word(std::string const& text_type)
  {
    if (text_type == "latin") {
      std::discrete_distribution<int> dist({85, 13, 2});
      switch (dist(gen)) {
        case 0: return english();
        case 1: return accented();
        default: return decomposed();
      }
    }
    std::discrete_distribution<int> dist({35, 12, 5, 6, 10, 10, 5, 7, 6, 4});
    switch (dist(gen)) {
      case 0: return english();
      case 1: return accented();
      case 2: return decomposed();
      case 3: return greek();
      case 4: return from_range(0xAC00, 0xD7A3, 2, 4);  // Hangul syllables
      case 5: return cjk();
      case 6: return width_variant();
      case 7: return symbol();
      case 8: return hebrew();
      default: return compat_jamo();
    }
  }
};

std::vector<std::string> create_rows(synthetic_unicode_data const& data,
                                     std::mt19937& gen,
                                     cudf::size_type num_rows,
                                     cudf::size_type row_width,
                                     std::string const& text_type)
{
  word_generator words{data, gen};
  std::vector<std::string> rows(num_rows);
  for (auto& row : rows) {
    while (true) {
      auto const word = words.word(text_type);
      if (row.size() + word.size() + 1 > static_cast<std::size_t>(row_width)) { break; }
      if (!row.empty()) { row += words.chance(0.08) ? ", " : " "; }
      row += word;
    }
  }
  return rows;
}

}  // namespace

static void bench_unicode_normalize(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const form_str  = state.get_string("form");
  auto const text_type = state.get_string("text_type");

  if (static_cast<int64_t>(num_rows) * row_width > (int64_t{1} << 29)) {
    state.skip("Skip benchmarks greater than 512MB");
    return;
  }

  auto const form = [&] {
    if (form_str == "NFD") { return nvtext::unicode_normalization_form::NFD; }
    if (form_str == "NFKD") { return nvtext::unicode_normalization_form::NFKD; }
    if (form_str == "NFKC") { return nvtext::unicode_normalization_form::NFKC; }
    return nvtext::unicode_normalization_form::NFC;
  }();

  std::mt19937 gen(0);
  auto const data = create_unicode_data(gen);
  auto const codepoints =
    cudf::test::strings_column_wrapper(data.codepoints.begin(), data.codepoints.end());
  auto const ccc =
    cudf::test::fixed_width_column_wrapper<int32_t>(data.ccc.begin(), data.ccc.end());
  auto const decomps = cudf::test::strings_column_wrapper(data.decomps.begin(), data.decomps.end());
  // the normalizer is created once and reused so construction time is not measured
  auto const normalizer =
    nvtext::create_unicode_normalizer(cudf::table_view({codepoints, ccc, decomps}), form);

  // create a pool of unique rows and then randomly sample from it to build the input
  auto const pool_size       = std::min(num_rows, cudf::size_type{4096});
  auto const h_rows          = create_rows(data, gen, pool_size, row_width, text_type);
  auto const pool            = cudf::test::strings_column_wrapper(h_rows.begin(), h_rows.end());
  data_profile const profile = data_profile_builder().no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 0, pool_size - 1);
  auto const indices = create_random_column(cudf::type_id::INT32, row_count{num_rows}, profile);
  auto const table   = cudf::gather(cudf::table_view({pool}), indices->view());
  auto const input   = cudf::strings_column_view(table->view().column(0));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  auto const chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  {
    // the output size depends on the data so compute it before timing
    auto const result = nvtext::normalize_unicode(input, *normalizer);
    state.add_global_memory_writes<nvbench::int8_t>(
      cudf::strings_column_view(result->view()).chars_size(cudf::get_default_stream()));
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = nvtext::normalize_unicode(input, *normalizer);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_unicode_normalize)
  .set_name("unicode_normalize")
  .add_string_axis("form", {"NFD", "NFC", "NFKD", "NFKC"})
  .add_string_axis("text_type", {"latin", "mixed"})
  .add_int64_axis("num_rows", {32768, 262144})
  .add_int64_axis("row_width", {128, 512, 2048, 8192});
