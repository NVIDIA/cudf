/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file bpe_data_generator.hpp
 * @brief Host-only utilities for generating realistic byte-pair-encoding test data
 *
 * Generates natural-language-like text (Zipf distributed words, punctuation, numbers,
 * some multi-byte characters, optional code-like indentation) and trains a merges table
 * on it using the same pre-tokenization and byte-level mapping as GPT-2.
 * This avoids checking large data files (e.g. GPT-2 merges.txt) into the repository
 * while still exercising the structure of real tables and text.
 * Also includes a reference (CPU) implementation of the BPE algorithm.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cudf::test::bpe {

/**
 * @brief Returns the GPT-2 byte-to-unicode table as UTF-8 strings
 *
 * Printable bytes map to themselves; others (including space and newline)
 * map to code points starting at U+0100 (e.g. space -> 'Ġ', newline -> 'Ċ').
 */
inline std::vector<std::string> const& gpt2_byte_table()
{
  static std::vector<std::string> const table = [] {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) {
      bs.push_back(b);
    }
    for (int b = 0xA1; b <= 0xAC; ++b) {
      bs.push_back(b);
    }
    for (int b = 0xAE; b <= 0xFF; ++b) {
      bs.push_back(b);
    }
    std::vector<int> cs(bs);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
        bs.push_back(b);
        cs.push_back(256 + n++);
      }
    }
    std::vector<std::string> result(256);
    for (std::size_t i = 0; i < bs.size(); ++i) {
      auto const c = cs[i];
      std::string u;
      if (c < 0x80) {
        u += static_cast<char>(c);
      } else {
        u += static_cast<char>(0xC0 | (c >> 6));
        u += static_cast<char>(0x80 | (c & 0x3F));
      }
      result[bs[i]] = u;
    }
    return result;
  }();
  return table;
}

/**
 * @brief Maps each byte of the input to its GPT-2 byte-level unicode character
 */
inline std::string gpt2_byte_map(std::string_view input)
{
  auto const& table = gpt2_byte_table();
  std::string result;
  result.reserve(input.size() * 2);
  for (unsigned char c : input) {
    result += table[c];
  }
  return result;
}

/**
 * @brief Returns the number of bytes in the UTF-8 character starting with `c`
 */
inline int utf8_char_bytes(unsigned char c)
{
  return c < 0x80 ? 1 : (c >> 5) == 0x06 ? 2 : (c >> 4) == 0x0E ? 3 : 4;
}

/**
 * @brief Splits the input into individual UTF-8 characters
 */
inline std::vector<std::string_view> utf8_chars(std::string_view s)
{
  std::vector<std::string_view> result;
  for (std::size_t i = 0; i < s.size();) {
    auto const n = std::min<std::size_t>(utf8_char_bytes(s[i]), s.size() - i);
    result.push_back(s.substr(i, n));
    i += n;
  }
  return result;
}

/**
 * @brief Generates natural-language-like text
 *
 * Words are built from syllables and drawn with a Zipf-Mandelbrot distribution
 * so that short words are frequent and the vocabulary has a long tail.
 */
class text_generator {
 public:
  /**
   * @param seed Random seed; output is deterministic for a given seed
   * @param vocab_size Number of distinct words
   * @param multibyte_rate Fraction of words containing multi-byte characters
   */
  explicit text_generator(uint32_t seed = 1, int vocab_size = 20000, double multibyte_rate = 0.02)
    : _rng(seed)
  {
    static char const* const onsets[] = {"",   "",   "b",  "c",  "d",  "f",  "g",  "h",  "j",
                                         "k",  "l",  "m",  "n",  "p",  "r",  "s",  "t",  "v",
                                         "w",  "y",  "z",  "th", "st", "br", "ch", "sh", "pl",
                                         "gr", "tr", "wh", "qu", "cl", "fr", "sp", "str"};
    static char const* const vowels[] = {
      "a", "e", "i", "o", "u", "ea", "ou", "ai", "ee", "oo", "io", "y", "a", "e", "i", "o"};
    static char const* const codas[]   = {"",   "",   "",   "",   "n", "r", "s", "t",  "l",  "nd",
                                          "st", "ng", "ck", "rt", "m", "d", "x", "ll", "ss", "th"};
    static char const* const accents[] = {"é", "è", "ü", "ñ", "ö", "ç", "á", "í"};
    std::uniform_int_distribution<std::size_t> onset(0, std::size(onsets) - 1);
    std::uniform_int_distribution<std::size_t> vowel(0, std::size(vowels) - 1);
    std::uniform_int_distribution<std::size_t> coda(0, std::size(codas) - 1);
    std::uniform_int_distribution<std::size_t> accent(0, std::size(accents) - 1);
    std::uniform_int_distribution<int> cjk(0, 799);
    std::discrete_distribution<int> syllables({0, 35, 35, 20, 8, 2});
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    std::unordered_set<std::string> seen;
    std::vector<std::pair<double, std::string>> words;
    while (static_cast<int>(words.size()) < vocab_size) {
      std::string w;
      auto const kind = unit(_rng);
      if (kind < multibyte_rate / 2) {
        // CJK word: 1-4 characters from U+4E00..
        auto const n = 1 + cjk(_rng) % 4;
        for (int i = 0; i < n; ++i) {
          auto const cp = 0x4E00 + cjk(_rng);
          w += static_cast<char>(0xE0 | (cp >> 12));
          w += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
          w += static_cast<char>(0x80 | (cp & 0x3F));
        }
      } else {
        auto const n = syllables(_rng);
        for (int i = 0; i < n; ++i) {
          w += onsets[onset(_rng)];
          w += (kind < multibyte_rate && i == 0) ? accents[accent(_rng)] : vowels[vowel(_rng)];
          w += codas[coda(_rng)];
        }
      }
      if (w.empty() || !seen.insert(w).second) { continue; }
      // frequent words tend to be short
      std::normal_distribution<double> noise(0.0, 2.0);
      words.emplace_back(static_cast<double>(w.size()) + noise(_rng), w);
    }
    std::sort(words.begin(), words.end());
    double total = 0;
    for (std::size_t r = 0; r < words.size(); ++r) {
      total += 1.0 / std::pow(static_cast<double>(r) + 2.7, 1.07);
      _cdf.push_back(total);
      _vocab.push_back(std::move(words[r].second));
    }
  }

  /**
   * @brief Generate text of at least `bytes` bytes
   *
   * @param bytes Minimum number of bytes to generate
   * @param code_like Adds indentation, short lines, and code punctuation
   */
  std::string generate(std::size_t bytes, bool code_like = false)
  {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> sentence_len(3, 25);
    std::uniform_int_distribution<int> sentences(1, 6);
    std::uniform_int_distribution<int> digits(1, 9999);
    std::uniform_int_distribution<int> indent(0, 4);
    std::uniform_int_distribution<int> line_len(2, 10);
    static char const* const code_punct[] = {"(", ")", ";", "{", "}", " =", "::", "->", "_", ","};
    std::uniform_int_distribution<std::size_t> punct(0, std::size(code_punct) - 1);

    std::string out;
    out.reserve(bytes + 1024);
    while (out.size() < bytes) {
      auto const nsent = sentences(_rng);
      for (int s = 0; s < nsent; ++s) {
        auto const nwords = sentence_len(_rng);
        auto line_words   = line_len(_rng);
        for (int w = 0; w < nwords; ++w) {
          if (code_like && --line_words == 0) {
            out += '\n';
            out.append(4 * indent(_rng), ' ');
            line_words = line_len(_rng);
          } else if (w > 0 || s > 0) {
            out += ' ';
          }
          auto const p = unit(_rng);
          std::string word;
          if (p < 0.015) {
            word = std::to_string(digits(_rng));
          } else {
            word = next_word();
            if (w == 0 && word[0] >= 'a' && word[0] <= 'z') { word[0] = word[0] - 'a' + 'A'; }
            if (p > 0.98) { word = "\"" + word + "\""; }
          }
          out += word;
          auto const q = unit(_rng);
          if (code_like && q < 0.15) {
            out += code_punct[punct(_rng)];
          } else if (q < 0.07) {
            out += ',';
          } else if (q < 0.08) {
            out += ';';
          }
        }
        auto const e = unit(_rng);
        out += e < 0.8 ? '.' : (e < 0.9 ? '?' : '!');
      }
      out += unit(_rng) < 0.8 ? "\n\n" : "\n";
    }
    return out;
  }

 private:
  std::string const& next_word()
  {
    std::uniform_real_distribution<double> unit(0.0, _cdf.back());
    auto const it = std::lower_bound(_cdf.begin(), _cdf.end(), unit(_rng));
    return _vocab[std::min<std::size_t>(std::distance(_cdf.begin(), it), _vocab.size() - 1)];
  }

  std::mt19937 _rng;
  std::vector<std::string> _vocab;
  std::vector<double> _cdf;
};

/**
 * @brief Splits raw text into GPT-2 style pre-tokens
 *
 * Approximates the GPT-2 regex: ` ?letters+| ?digits+| ?other+|\s+(?!\S)|\s+`
 * Bytes >= 0x80 are treated as letters.
 */
inline std::vector<std::string_view> pretokenize(std::string_view s)
{
  auto char_class = [](unsigned char c) {
    if (c == ' ' || c == '\n' || c == '\t' || c == '\r') { return 0; }
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80) { return 1; }
    if (c >= '0' && c <= '9') { return 2; }
    return 3;
  };
  std::vector<std::string_view> result;
  std::size_t const n = s.size();
  std::size_t i       = 0;
  while (i < n) {
    auto j = i;
    if (s[i] == ' ' && i + 1 < n && char_class(s[i + 1]) != 0) {
      auto const cls = char_class(s[i + 1]);
      j              = i + 1;
      while (j < n && char_class(s[j]) == cls) {
        ++j;
      }
    } else if (char_class(s[i]) == 0) {
      while (j < n && char_class(s[j]) == 0) {
        ++j;
      }
      // leave the last space to prefix the next word
      if (j < n && j - i > 1 && s[j - 1] == ' ') { --j; }
    } else {
      auto const cls = char_class(s[i]);
      while (j < n && char_class(s[j]) == cls) {
        ++j;
      }
    }
    result.push_back(s.substr(i, j - i));
    i = j;
  }
  return result;
}

/**
 * @brief Trains a BPE merges table on the given raw text
 *
 * Uses GPT-2 pre-tokenization and byte-level mapping so the returned
 * table is in the same form as GPT-2's merges.txt (e.g. "Ġ t", "Ġt he").
 *
 * @param raw_text Training text
 * @param num_merges Maximum number of merge pairs to return
 * @return Merge pairs as "left right" strings ordered by rank
 */
inline std::vector<std::string> train_merges(std::string_view raw_text, int num_merges)
{
  std::unordered_map<std::string, int64_t> counts;
  for (auto const& p : pretokenize(raw_text)) {
    counts[gpt2_byte_map(p)]++;
  }
  std::vector<std::pair<std::string, int64_t>> sorted_counts(counts.begin(), counts.end());
  std::sort(sorted_counts.begin(), sorted_counts.end());

  std::vector<std::string> tokens;
  std::unordered_map<std::string, int> token_ids;
  auto get_id = [&](std::string const& t) {
    auto const [it, inserted] = token_ids.try_emplace(t, static_cast<int>(tokens.size()));
    if (inserted) { tokens.push_back(t); }
    return it->second;
  };
  auto key = [](int a, int b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
  };

  std::vector<std::vector<int>> words;
  std::vector<int64_t> freqs;
  std::unordered_map<uint64_t, int64_t> pair_counts;
  std::unordered_map<uint64_t, std::vector<int>> where;
  for (auto const& [w, c] : sorted_counts) {
    std::vector<int> symbols;
    for (auto ch : utf8_chars(w)) {
      symbols.push_back(get_id(std::string(ch)));
    }
    auto const idx = static_cast<int>(words.size());
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto const k = key(symbols[i], symbols[i + 1]);
      pair_counts[k] += c;
      where[k].push_back(idx);
    }
    words.push_back(std::move(symbols));
    freqs.push_back(c);
  }

  std::priority_queue<std::pair<int64_t, uint64_t>> heap;
  for (auto const& [k, c] : pair_counts) {
    heap.emplace(c, k);
  }

  std::vector<std::string> merges;
  while (static_cast<int>(merges.size()) < num_merges && !heap.empty()) {
    auto const [c, k] = heap.top();
    heap.pop();
    auto const pc = pair_counts.find(k);
    if (pc == pair_counts.end() || pc->second != c || c <= 0) { continue; }  // stale
    auto const a      = static_cast<int>(k >> 32);
    auto const b      = static_cast<int>(k & 0xFFFFFFFFu);
    auto const merged = get_id(tokens[a] + tokens[b]);
    merges.push_back(tokens[a] + " " + tokens[b]);

    auto ws = std::move(where[k]);
    std::sort(ws.begin(), ws.end());
    ws.erase(std::unique(ws.begin(), ws.end()), ws.end());
    std::unordered_set<uint64_t> changed;
    for (auto const w : ws) {
      auto& s      = words[w];
      auto const f = freqs[w];
      bool found   = false;
      for (std::size_t i = 0; i + 1 < s.size() && !found; ++i) {
        found = (s[i] == a && s[i + 1] == b);
      }
      if (!found) { continue; }
      for (std::size_t i = 0; i + 1 < s.size(); ++i) {
        auto const k2 = key(s[i], s[i + 1]);
        pair_counts[k2] -= f;
        changed.insert(k2);
      }
      std::vector<int> next;
      for (std::size_t i = 0; i < s.size();) {
        if (i + 1 < s.size() && s[i] == a && s[i + 1] == b) {
          next.push_back(merged);
          i += 2;
        } else {
          next.push_back(s[i++]);
        }
      }
      s = std::move(next);
      for (std::size_t i = 0; i + 1 < s.size(); ++i) {
        auto const k2 = key(s[i], s[i + 1]);
        pair_counts[k2] += f;
        where[k2].push_back(w);
        changed.insert(k2);
      }
    }
    for (auto const k2 : changed) {
      if (auto const v = pair_counts[k2]; v > 0) { heap.emplace(v, k2); }
    }
  }
  return merges;
}

/**
 * @brief Reference (CPU) byte-pair-encoding
 *
 * Implements the GPT-2/HuggingFace semantics: repeatedly find the lowest ranked
 * adjacent pair and merge all of its non-overlapping occurrences left-to-right.
 * Runs in O(n log n) per string using a heap over pair positions.
 */
class reference_encoder {
 public:
  explicit reference_encoder(std::vector<std::string> const& merges)
  {
    for (std::size_t i = 0; i < merges.size(); ++i) {
      _ranks.emplace(merges[i], static_cast<int>(i));
    }
  }

  std::string encode(std::string_view input, char separator = ' ') const
  {
    auto const chars = utf8_chars(input);
    auto const n     = static_cast<int>(chars.size());
    if (n == 0) { return {}; }
    std::vector<int> start(n), next(n), prev(n);  // token linked list by char index
    std::vector<std::size_t> offsets(n + 1);
    for (int i = 0; i < n; ++i) {
      offsets[i] = static_cast<std::size_t>(chars[i].data() - input.data());
      next[i]    = i + 1;
      prev[i]    = i - 1;
    }
    offsets[n] = input.size();
    auto token = [&](int i) { return input.substr(offsets[i], offsets[next[i]] - offsets[i]); };
    auto rank  = [&](int i) {  // rank of (token before i, token at i)
      if (i <= 0 || i >= n || prev[i] < 0) { return -1; }
      std::string key(token(prev[i]));
      key += ' ';
      key += token(i);
      auto const f = _ranks.find(key);
      return f == _ranks.end() ? -1 : f->second;
    };
    using entry = std::pair<int, int>;  // (rank, position)
    std::priority_queue<entry, std::vector<entry>, std::greater<>> heap;
    std::vector<bool> alive(n, true);  // token starts
    for (int i = 1; i < n; ++i) {
      if (auto const r = rank(i); r >= 0) { heap.emplace(r, i); }
    }
    while (!heap.empty()) {
      auto const r = heap.top().first;
      std::vector<int> positions;
      while (!heap.empty() && heap.top().first == r) {
        positions.push_back(heap.top().second);
        heap.pop();
      }
      std::sort(positions.begin(), positions.end());
      positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
      std::vector<int> touched;
      for (auto const p : positions) {
        // skip if no longer a token start or if the pair changed earlier in this round
        if (!alive[p] || rank(p) != r) { continue; }
        auto const q = prev[p];
        auto const e = next[p];
        next[q]      = e;
        if (e < n) { prev[e] = q; }
        alive[p] = false;
        touched.push_back(q);
        if (e < n) { touched.push_back(e); }
      }
      for (auto const t : touched) {
        if (alive[t]) {
          if (auto const r2 = rank(t); r2 >= 0) { heap.emplace(r2, t); }
        }
      }
    }
    std::string result;
    for (int i = 0; i < n; i = next[i]) {
      if (i > 0) { result += separator; }
      result += token(i);
    }
    return result;
  }

 private:
  std::unordered_map<std::string, int> _ranks;
};

}  // namespace cudf::test::bpe
