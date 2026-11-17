/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/strings/string_view.cuh>

#include <cuda/std/limits>

namespace cudf {
namespace strings {
namespace detail {

__device__ constexpr bool is_whitespace(char_utf8 ch) { return ch <= ' '; }

/**
 * @brief Count tokens delimited by whitespace
 *
 * @param d_str String to tokenize
 * @param max_tokens Maximum number of tokens to count
 * @return Number of tokens delimited by whitespace
 */
__device__ inline size_type count_tokens_whitespace(
  string_view d_str, size_type const max_tokens = cuda::std::numeric_limits<size_type>::max())
{
  auto token_count = size_type{0};
  auto spaces      = true;
  auto itr         = d_str.data();
  auto const end   = itr + d_str.size_bytes();
  while (itr < end && token_count < max_tokens) {
    cudf::char_utf8 ch   = 0;
    auto const chr_width = cudf::strings::detail::to_char_utf8(itr, ch);
    if (spaces == is_whitespace(ch)) {
      itr += chr_width;
    } else {
      token_count += static_cast<size_type>(spaces);
      spaces = !spaces;
    }
  }
  return token_count;
}

// JIT has trouble including thrust/pair.h
struct position_pair {
  size_type first;
  size_type second;
};

/**
 * @brief Instantiated for each string to manage navigating tokens from
 * the beginning or the end of that string.
 */
struct whitespace_string_tokenizer {
  /**
   * @brief Identifies the position range of the next token in the given
   * string at the specified iterator position.
   *
   * Tokens are delimited by one or more whitespace characters.
   *
   * @return true if a token has been found
   */
  __device__ bool next_token()
  {
    if (start_position >= d_str.size_bytes()) { return false; }
    auto const src_ptr = d_str.data();
    if (current_position != 0) {
      current_position += cudf::strings::detail::bytes_in_char_utf8(src_ptr[current_position]);
      start_position = current_position;
    }
    if (start_position >= d_str.size_bytes()) { return false; }
    // continue search for the next token
    end_position = d_str.size_bytes();
    while (current_position < d_str.size_bytes()) {
      cudf::char_utf8 ch   = 0;
      auto const chr_width = cudf::strings::detail::to_char_utf8(src_ptr + current_position, ch);
      if (spaces == is_whitespace(ch)) {
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
   * @brief Identifies the position range of the previous token in the given
   * string at the specified iterator position.
   *
   * Tokens are delimited by one or more whitespace characters.
   *
   * @return true if a token has been found
   */
  __device__ bool prev_token()
  {
    end_position = start_position - 1;
    --itr;
    if (end_position <= 0) return false;
    // continue search for the next token
    start_position = 0;
    for (; itr >= d_str.begin(); --itr) {
      if (spaces == (*itr <= ' ')) {
        if (spaces)
          end_position = itr.byte_offset();
        else
          start_position = itr.byte_offset();
        continue;
      }
      spaces = !spaces;
      if (spaces) {
        start_position = (itr + 1).byte_offset();
        break;
      }
    }
    return start_position < end_position;
  }

  __device__ position_pair get_token() const { return position_pair{start_position, end_position}; }

  __device__ whitespace_string_tokenizer(string_view const& d_str, bool reverse = false)
    : d_str{d_str},
      spaces(true),
      start_position{reverse ? d_str.size_bytes() + 1 : 0},
      end_position{d_str.size_bytes()},
      itr{reverse ? d_str.end() : d_str.begin()},
      current_position{0}
  {
  }

 private:
  string_view const d_str;
  bool spaces;  // true if current position is whitespace
  cudf::string_view::const_iterator itr;
  size_type start_position;
  size_type end_position;
  size_type current_position;
};

}  // namespace detail
}  // namespace strings
}  // namespace cudf
