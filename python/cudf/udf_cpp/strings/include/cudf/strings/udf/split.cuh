/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "udf_string.cuh"

#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace udf {
namespace detail {

/**
 * @brief Split string using given string
 *
 * The caller must allocate an array of cudf::string_view to be filled
 * in by this function. This function can be called with a `result=nullptr`
 * to compute the number of tokens.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{"the best  of times ", 19};
 * auto tgt = cudf::string_view{" ", 1};
 * auto token_count = split(d_str, tgt, nullptr);
 * auto result = new cudf::string_view[token_count];
 * split(d_str, tgt, result);
 * // result is array like ["the", "best", "", "of", "times", ""]
 * @endcode
 *
 * @param d_str String to split
 * @param tgt String to split on
 * @param result Empty array to populate with output objects.
 *               Pass `nullptr` to just get the token count.
 * @return Number of tokens returned
 */
__device__ inline cudf::size_type split(cudf::string_view const d_str,
                                        cudf::string_view const tgt,
                                        cudf::string_view* result)
{
  auto const nchars     = d_str.length();
  cudf::size_type count = 0;

  cudf::size_type last_pos = 0;
  while (last_pos <= nchars) {
    cudf::size_type const pos = d_str.find(tgt, last_pos);
    auto const length         = (pos < 0 ? nchars : pos) - last_pos;
    if (result) { *result++ = d_str.substr(last_pos, length); }
    last_pos = pos + tgt.length();
    ++count;
    if (pos < 0) { break; }
  }

  return count;
}
}  // namespace detail

/**
 * @brief Count tokens in a string without performing the split
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{"the best  of times ", 19};
 * auto tgt = cudf::string_view{" ", 1};
 * auto token_count = count_tokens(d_str, tgt);
 * // token_count is 6
 * @endcode
 *
 * @param d_str String to split
 * @param tgt String to split on
 * @return Number of tokens returned
 */
__device__ inline cudf::size_type count_tokens(cudf::string_view const d_str,
                                               cudf::string_view const tgt)
{ return detail::split(d_str, tgt, nullptr); }

/**
 * @brief Split string using given string
 *
 * The caller must allocate an array of cudf::string_view to be filled
 * in by this function.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{"the best  of times ", 19};
 * auto tgt = cudf::string_view{" ", 1};
 * auto token_count = count_tokens(d_str, tgt);
 * auto result = new cudf::string_view[token_count];
 * split(d_str, tgt, result);
 * // result is array like ["the", "best", "", "of", "times", ""]
 * @endcode
 *
 * @param d_str String to split
 * @param tgt String to split on
 * @param result Empty array to populate with output objects.
 * @return Number of tokens returned
 */
__device__ inline cudf::size_type split(cudf::string_view const d_str,
                                        cudf::string_view const tgt,
                                        cudf::string_view* result)
{ return detail::split(d_str, tgt, result); }

/**
 * @brief Split string using given target array
 *
 * @param d_str String to split
 * @param tgt Character array encoded in UTF-8 used for identifying split points
 * @param bytes Number of bytes to read from `tgt`
 * @param result Empty array to populate with output objects
 * @return Number of tokens returned
 */
__device__ inline cudf::size_type split(cudf::string_view const d_str,
                                        char const* tgt,
                                        cudf::size_type bytes,
                                        cudf::string_view* result)
{ return detail::split(d_str, cudf::string_view{tgt, bytes}, result); }

/**
 * @brief Split string using given target array
 *
 * @param d_str String to split
 * @param tgt Null-terminated character array encoded in UTF-8 used for identifying split points
 * @param result Empty array to populate with output objects
 * @return Number of tokens returned
 */
__device__ inline cudf::size_type split(cudf::string_view const d_str,
                                        char const* tgt,
                                        cudf::string_view* result)
{ return split(d_str, tgt, detail::bytes_in_null_terminated_string(tgt), result); }

namespace detail {
/**
 * @brief Split string on whitespace
 *
 * The caller must allocate an array of cudf::string_view to be filled
 * in by this function. This function can be called with a `result=nullptr`
 * to compute the number of tokens.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{"the best  of times ", 19};
 * auto token_count = split(d_str, nullptr);
 * auto result = new cudf::string_view[token_count];
 * split(d_str, result);
 * // result is array like ["the", "best", "of", "times"]
 * @endcode
 *
 * @param d_str String to split
 * @param result Empty array to populate with output objects.
 *               Pass `nullptr` to just get the token count.
 * @return Number of tokens returned
 */
__device__ inline cudf::size_type split(cudf::string_view const d_str, cudf::string_view* result)
{
  cudf::strings::detail::whitespace_string_tokenizer tokenizer{d_str};
  cudf::size_type count = 0;
  while (tokenizer.next_token()) {
    auto token = tokenizer.get_token();
    if (result) { *result++ = d_str.substr(token.first, token.second - token.first); }
    ++count;
  }
  return count;
}
}  // namespace detail

/**
 * @brief Count tokens in a string without performing the split on whitespace
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{"the best  of times ", 19};
 * auto token_count = count_tokens(d_str);
 * // token_count is 4
 * @endcode
 *
 * @param d_str String to split
 * @return Number of tokens returned
 */
__device__ inline cudf::size_type count_tokens(cudf::string_view const d_str)
{ return detail::split(d_str, nullptr); }

/**
 * @brief Split string on whitespace
 *
 * This will create tokens by splitting on one or more consecutive whitespace characters
 * found in `d_str`.
 *
 * @param d_str String to split
 * @param result Empty array to populate with output objects.
 * @return Number of tokens returned
 */
__device__ inline cudf::size_type split(cudf::string_view const d_str, cudf::string_view* result)
{ return detail::split(d_str, result); }

/**
 * @brief Join an array of strings with a separator
 *
 * @code{.cpp}
 * auto separator = cudf::string_view{"::", 2};
 * cudf::string_view input[] = {
 *   cudf::string_view{"hello", 5},
 *   cudf::string_view{"goodbye", 7},
 *   cudf::string_view{"world", 5} };
 *
 * auto result = join(separator, input, 3);
 * // result is "hello::goodbye::world"
 * @endcode
 *
 * @param separator Separator string
 * @param input An array of strings to join
 * @param count Number of elements in `input`
 * @return New string
 */
__device__ inline udf_string join(cudf::string_view const separator,
                                  cudf::string_view* input,
                                  cudf::size_type count)
{
  udf_string result{""};
  while (count-- > 0) {
    result += *input++;
    if (count > 0) { result += separator; }
  }
  return result;
}

/**
 * @brief Join an array of strings with a separator
 *
 * @param separator Null-terminated UTF-8 string
 * @param bytes Number of bytes to read from `separator`
 * @param input An array of strings to join
 * @param count Number of elements in `input`
 * @return New string
 */
__device__ inline udf_string join(char const* separator,
                                  cudf::size_type bytes,
                                  cudf::string_view* input,
                                  cudf::size_type count)
{ return join(cudf::string_view{separator, bytes}, input, count); }

/**
 * @brief Join an array of strings with a separator
 *
 * @param separator Null-terminated UTF-8 string
 * @param input An array of strings to join
 * @param count Number of elements in `input`
 * @return New string
 */
__device__ inline udf_string join(char const* separator,
                                  cudf::string_view* input,
                                  cudf::size_type count)
{ return join(separator, detail::bytes_in_null_terminated_string(separator), input, count); }

}  // namespace udf
}  // namespace strings
}  // namespace cudf
