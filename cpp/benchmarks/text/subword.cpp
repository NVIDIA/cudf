/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <nvtext/wordpiece_tokenize.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

/**
 * @brief Synthetic vocabulary resembling a BERT-style wordpiece vocabulary
 *
 * Contains special tokens, all single characters (with and without the `##` prefix),
 * about 22K whole words and about 6K `##` sub-word pieces for a total of about 28K entries.
 */
struct synthetic_vocabulary {
  std::vector<std::string> entries;  // all vocabulary entries; the token id is the index
  std::vector<std::string> words;    // whole words
  std::vector<std::string> pieces;   // sub-word pieces without the `##` prefix
};

std::string random_word(std::mt19937& gen, int min_size, int max_size)
{
  // letter frequencies roughly following English text
  static std::string const letters =
    "eeeeeeeeeeeettttttttaaaaaaaaoooooooiiiiiiinnnnnnnsssssshhhhhhrrrrrrddddllllcccuuummwwffggyypp"
    "bbvkjxqz";
  std::uniform_int_distribution<int> size_dist(min_size, max_size);
  std::uniform_int_distribution<std::size_t> letter_dist(0, letters.size() - 1);
  auto const size = size_dist(gen);
  std::string word;
  for (int i = 0; i < size; ++i) {
    word += letters[letter_dist(gen)];
  }
  return word;
}

synthetic_vocabulary create_vocabulary(std::mt19937& gen)
{
  constexpr int num_words  = 22000;
  constexpr int num_pieces = 6000;

  synthetic_vocabulary vocab;
  vocab.entries = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"};

  std::string const chars = "abcdefghijklmnopqrstuvwxyz0123456789.,!?'-";
  for (auto ch : chars) {
    vocab.entries.emplace_back(1, ch);
  }
  for (auto ch : chars) {
    vocab.entries.push_back(std::string("##") + ch);
  }

  // word lengths roughly following English dictionary words
  std::discrete_distribution<int> length_dist({0, 0, 2, 6, 10, 13, 14, 14, 12, 10, 8, 6, 5});
  std::set<std::string> unique_words;
  while (static_cast<int>(unique_words.size()) < num_words) {
    auto const size = length_dist(gen);
    unique_words.insert(random_word(gen, size, size));
  }
  vocab.words.assign(unique_words.begin(), unique_words.end());
  // shorter words are more frequent so they are placed first for the Zipf-like sampling
  std::shuffle(vocab.words.begin(), vocab.words.end(), gen);
  std::stable_sort(vocab.words.begin(), vocab.words.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.size() < rhs.size();
  });

  std::set<std::string> unique_pieces = {"s", "ed", "ing", "er", "ly", "tion", "est", "ment"};
  while (static_cast<int>(unique_pieces.size()) < num_pieces) {
    auto piece = random_word(gen, 2, 6);
    unique_pieces.insert(std::move(piece));
  }
  vocab.pieces.assign(unique_pieces.begin(), unique_pieces.end());

  for (auto const& word : vocab.words) {
    vocab.entries.push_back(word);
  }
  for (auto const& piece : vocab.pieces) {
    vocab.entries.push_back("##" + piece);
  }
  return vocab;
}

/**
 * @brief Creates rows of words with a mix similar to normalized English text
 *
 * - 70% whole words from the vocabulary chosen with a Zipf-like frequency
 * - 10% punctuation
 * - 15% whole words with 1-2 sub-word pieces appended (multiple tokens)
 * - 4% random out-of-vocabulary words (resolved as multiple sub-word pieces)
 * - 1% words with a character not in the vocabulary (resolves to [UNK])
 */
std::vector<std::string> create_rows(synthetic_vocabulary const& vocab,
                                     std::mt19937& gen,
                                     cudf::size_type num_rows,
                                     cudf::size_type words_per_row)
{
  std::vector<double> weights(vocab.words.size());
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), 0.9);
  }
  std::discrete_distribution<std::size_t> word_dist(weights.begin(), weights.end());
  std::uniform_int_distribution<std::size_t> piece_dist(0, vocab.pieces.size() - 1);
  std::uniform_int_distribution<int> pieces_count_dist(1, 2);
  std::uniform_int_distribution<int> category_dist(0, 99);
  std::string const punctuation = ".,!?'-";
  std::uniform_int_distribution<std::size_t> punct_dist(0, punctuation.size() - 1);

  std::vector<std::string> rows(num_rows);
  for (auto& row : rows) {
    for (cudf::size_type w = 0; w < words_per_row; ++w) {
      if (w > 0) { row += ' '; }
      auto const category = category_dist(gen);
      if (category < 70) {
        row += vocab.words[word_dist(gen)];
      } else if (category < 80) {
        row += punctuation[punct_dist(gen)];
      } else if (category < 95) {
        row += vocab.words[word_dist(gen)];
        auto const count = pieces_count_dist(gen);
        for (int p = 0; p < count; ++p) {
          row += vocab.pieces[piece_dist(gen)];
        }
      } else if (category < 99) {
        row += random_word(gen, 6, 12);
      } else {
        auto word             = random_word(gen, 3, 8);
        word[word.size() / 2] = '~';
        row += word;
      }
    }
  }
  return rows;
}

}  // namespace

static void bench_wordpiece_tokenizer(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_words = static_cast<cudf::size_type>(state.get_int64("max_words"));
  auto const num_words = static_cast<cudf::size_type>(state.get_int64("num_words"));

  if (static_cast<int64_t>(num_rows) * num_words > (int64_t{1} << 27)) {
    state.skip("Skip benchmarks greater than 128M words");
    return;
  }

  std::mt19937 gen(0);
  auto const synthetic_vocab = create_vocabulary(gen);
  auto const vocabulary      = cudf::test::strings_column_wrapper(synthetic_vocab.entries.begin(),
                                                             synthetic_vocab.entries.end());
  auto const vocab = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));

  // create a pool of unique rows and then randomly sample from it to build the input
  auto const pool_size       = std::min(num_rows, cudf::size_type{4096});
  auto const h_rows          = create_rows(synthetic_vocab, gen, pool_size, num_words);
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
    // the number of output tokens depends on the data so compute it before timing
    auto const result = nvtext::wordpiece_tokenize(input, *vocab, max_words);
    state.add_global_memory_writes<nvbench::int32_t>(result->child(1).size());
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::wordpiece_tokenize(input, *vocab, max_words);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_wordpiece_tokenizer)
  .set_name("wordpiece_tokenize")
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("num_words", {32, 256, 2048})
  .add_int64_axis("max_words", {0, 20, 40});
