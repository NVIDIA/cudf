/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>

#include <cuda/std/utility>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/logical.h>

namespace nvtext {
namespace detail {
using string_index_pair = cuda::std::pair<char const*, cudf::size_type>;
using position_pair     = cuda::std::pair<cudf::size_type, cudf::size_type>;

/**
 * @brief Tokenizer class that use multi-character delimiters.
 *
 * This is common code for tokenize, token-counters, normalize functions.
 * If an empty delimiter string is specified, then whitespace
 * (code-point <= ' ') is used to identify tokens.
 *
 * After instantiating this object, use the `next_token()` method
 * to parse tokens and the `token_byte_positions()` to retrieve the
 * current token's byte offsets within the string.
 */
struct characters_tokenizer {
  /**
   * @brief Constructor for characters_tokenizer.
   *
   * @param d_str The string to tokenize.
   * @param d_delimiter The (optional) delimiter to locate tokens.
   */
  __device__ characters_tokenizer(cudf::string_view const& d_str,
                                  cudf::string_view const& d_delimiter = cudf::string_view{})
    : d_str{d_str},
      d_delimiter{d_delimiter},
      spaces{true},
      current_position{0},
      start_position(0),
      end_position(d_str.size_bytes())
  {
  }

  /**
   * @brief Return true if the given character is a delimiter.
   *
   * For empty delimiter, whitespace code-point is checked.
   *
   * @param chr The character to test.
   * @return true if the character is a delimiter
   */
  __device__ bool is_delimiter(cudf::char_utf8 chr) const
  {
    return d_delimiter.empty() ? (chr <= ' ') :  // whitespace check
             thrust::any_of(thrust::seq,
                            d_delimiter.begin(),
                            d_delimiter.end(),
                            [chr] __device__(cudf::char_utf8 c) { return c == chr; });
  }

  /**
   * @brief Identifies the bounds of the next token in the given
   * string at the specified iterator position.
   *
   * For empty delimiter, whitespace code-point is checked.
   * Starting at the current_position, a token
   * start position is identified when a delimiter is
   * not found. Once found, the end position is identified
   * when a delimiter or the end of the string is found.
   *
   * @return true if a token has been found
   */
  __device__ bool next_token()
  {
    auto const src_ptr = d_str.data();
    if (current_position >= d_str.size_bytes()) { return false; }
    if (current_position != 0) {  // skip these 2 lines the first time through
      current_position += cudf::strings::detail::bytes_in_char_utf8(src_ptr[current_position]);
      start_position = current_position;
    }
    if (start_position >= d_str.size_bytes()) { return false; }
    // continue search for the next token
    end_position = d_str.size_bytes();
    while (current_position < d_str.size_bytes()) {
      cudf::char_utf8 ch   = 0;
      auto const chr_width = cudf::strings::detail::to_char_utf8(src_ptr + current_position, ch);
      if (chr_width == 0) {
        current_position++;
        continue;
      }
      if (spaces == is_delimiter(ch)) {
        current_position += chr_width;
        if (spaces) {
          start_position = current_position;
        } else {
          end_position = current_position;
        }
        continue;
      }
      spaces = !spaces;
      if (spaces) {
        end_position = current_position;
        break;
      }
      current_position += chr_width;
    }
    return start_position < end_position;
  }

  /**
   * @brief Returns the byte offsets for the current token
   * within this string.
   *
   * @return Byte positions of the current token.
   */
  __device__ position_pair token_byte_positions() const
  { return position_pair{start_position, end_position}; }

 private:
  cudf::string_view const d_str;        ///< string to tokenize
  cudf::string_view const d_delimiter;  ///< delimiter characters
  bool spaces;                          ///< true if current position is delimiter
  cudf::size_type current_position;     ///< current position in d_str
  cudf::size_type start_position;       ///< starting byte position of token found
  cudf::size_type end_position;         ///< ending byte position (exclusive) of token found
};

/**
 * @brief Tokenizing function for multi-character delimiter.
 *
 * The first pass simply counts the tokens so the size of the output
 * vector can be calculated. The second pass places the token
 * positions into the d_tokens vector.
 */
struct strings_tokenizer {
  cudf::column_device_view const d_strings;    ///< strings to tokenize
  cudf::string_view const d_delimiter;         ///< delimiter characters to tokenize around
  cudf::detail::input_offsetalator d_offsets;  ///< offsets into the d_tokens vector for each string
  string_index_pair* d_tokens{};               ///< token positions in device memory

  /**
   * @brief Identifies the token positions within each string.
   *
   * This counts the tokens in each string and also places the token positions
   * into the d_tokens member.
   *
   * @param idx Index of the string to tokenize in the d_strings column.
   * @return The number of tokens for this string.
   */
  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto d_str = d_strings.element<cudf::string_view>(idx);
    // create tokenizer for this string
    characters_tokenizer tokenizer(d_str, d_delimiter);
    string_index_pair* d_str_tokens = d_tokens ? d_tokens + d_offsets[idx] : nullptr;
    cudf::size_type token_idx       = 0;
    while (tokenizer.next_token()) {
      if (d_str_tokens) {
        auto token_pos = tokenizer.token_byte_positions();
        d_str_tokens[token_idx] =
          string_index_pair{d_str.data() + token_pos.first, (token_pos.second - token_pos.first)};
      }
      ++token_idx;
    }
    return token_idx;  // number of tokens found
  }
};

// delimiters' iterator = delimiterator
using delimiterator = cudf::column_device_view::const_iterator<cudf::string_view>;

/**
 * @brief Tokenizes strings using multiple string delimiters.
 *
 * One or more strings are used as delimiters to identify tokens inside
 * each string of a given strings column.
 */
struct multi_delimiter_strings_tokenizer {
  cudf::column_device_view const d_strings;    ///< strings column to tokenize
  delimiterator delimiters_begin;              ///< first delimiter
  delimiterator delimiters_end;                ///< last delimiter
  cudf::detail::input_offsetalator d_offsets;  ///< offsets into the d_tokens output vector
  string_index_pair* d_tokens{};               ///< token positions found for each string

  /**
   * @brief Identifies the token positions within each string.
   *
   * This counts the tokens in each string and also places the token positions
   * into the d_tokens member.
   *
   * @param idx Index of the string to tokenize in the d_strings column.
   * @return The number of tokens for this string.
   */
  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    cudf::string_view d_str   = d_strings.element<cudf::string_view>(idx);
    auto d_str_tokens         = d_tokens ? d_tokens + d_offsets[idx] : nullptr;
    auto data_ptr             = d_str.data();
    cudf::size_type last_pos  = 0;
    cudf::size_type token_idx = 0;
    // check for delimiters at each character position
    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto curr_ptr = data_ptr + itr.byte_offset();
      cudf::string_view sub_str(
        curr_ptr, static_cast<cudf::size_type>(data_ptr + d_str.size_bytes() - curr_ptr));
      // look for delimiter at current position
      auto itr_find = thrust::find_if(
        thrust::seq,
        delimiters_begin,
        delimiters_end,
        [sub_str] __device__(cudf::string_view const& d_delim) {
          return !d_delim.empty() && (d_delim.size_bytes() <= sub_str.size_bytes()) &&
                 d_delim.compare(sub_str.data(), d_delim.size_bytes()) == 0;
        });
      if (itr_find != delimiters_end) {  // found delimiter
        auto token_size = static_cast<cudf::size_type>((curr_ptr - data_ptr) - last_pos);
        if (token_size > 0)  // we only care about non-zero sized tokens
        {
          if (d_str_tokens)
            d_str_tokens[token_idx] = string_index_pair{data_ptr + last_pos, token_size};
          ++token_idx;
        }
        last_pos = (curr_ptr - data_ptr) + (*itr_find).size_bytes();  // point past delimiter
        itr += (*itr_find).length() - 1;
      }
    }
    if (last_pos < d_str.size_bytes())  // left-over tokens
    {
      if (d_str_tokens)
        d_str_tokens[token_idx] =
          string_index_pair{data_ptr + last_pos, d_str.size_bytes() - last_pos};
      ++token_idx;
    }
    return token_idx;  // this is the number of tokens found for this string
  }
};

}  // namespace detail
}  // namespace nvtext
