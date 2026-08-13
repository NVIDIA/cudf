/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT nvtext {
/**
 * @addtogroup nvtext_tokenize
 * @{
 * @file
 * @brief APIs for tokenizing and counting tokens in strings columns.
 */

/**
 * @brief Returns a single column of strings by tokenizing the input strings
 * column using the provided characters as delimiters.
 *
 * The `delimiter` may be zero or more characters. If the `delimiter` is empty,
 * whitespace (character code-point <= ' ') is used for identifying tokens.
 * Also, any consecutive delimiters found in a string are ignored.
 * This means only non-empty tokens are returned.
 *
 * Tokens are found by locating delimiter(s) starting at the beginning of each string.
 * As each string is tokenized, the tokens are appended using input column row order
 * to build the output column. That is, tokens found in input row[i] will be placed in
 * the output column directly before tokens found in input row[i+1].
 *
 * @code{.pseudo}
 * Example:
 * s = ["a", "b c", "d  e f "]
 * t = tokenize(s)
 * t is now ["a", "b", "c", "d", "e", "f"]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @param input Strings column to tokenize
 * @param delimiter UTF-8 characters used to separate each string into tokens.
 *                  The default of empty string will separate tokens using whitespace.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings columns of tokens
 */
std::unique_ptr<cudf::column> tokenize(
  cudf::strings_column_view const& input,
  cudf::string_scalar const& delimiter = cudf::string_scalar{""},
  rmm::cuda_stream_view stream         = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr    = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a single column of strings by tokenizing the input strings
 * column using multiple strings as delimiters.
 *
 * Tokens are found by locating delimiter(s) starting at the beginning of each string.
 * Any consecutive delimiters found in a string are ignored.
 * This means only non-empty tokens are returned.
 *
 * As each string is tokenized, the tokens are appended using input column row order
 * to build the output column. That is, tokens found in input row[i] will be placed in
 * the output column directly before tokens found in input row[i+1].
 *
 * @code{.pseudo}
 * Example:
 * s = ["a", "b c", "d.e:f;"]
 * d = [".", ":", ";"]
 * t = tokenize(s,d)
 * t is now ["a", "b c", "d", "e", "f"]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @throw cudf::logic_error if the delimiters column is empty or contains nulls.
 *
 * @param input Strings column to tokenize
 * @param delimiters Strings used to separate individual strings into tokens
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings columns of tokens
 */
std::unique_ptr<cudf::column> tokenize(
  cudf::strings_column_view const& input,
  cudf::strings_column_view const& delimiters,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the number of tokens in each string of a strings column.
 *
 * The `delimiter` may be zero or more characters. If the `delimiter` is empty,
 * whitespace (character code-point <= ' ') is used for identifying tokens.
 * Also, any consecutive delimiters found in a string are ignored.
 * This means that only empty strings or null rows will result in a token count of 0.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a", "b c", " ", "d e f"]
 * t = count_tokens(s)
 * t is now [1, 2, 0, 3]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 * The number of tokens for a null element is set to 0 in the output column.
 *
 * @param input Strings column to count tokens
 * @param delimiter Strings used to separate each string into tokens.
 *                  The default of empty string will separate tokens using whitespace.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of token counts
 */
std::unique_ptr<cudf::column> count_tokens(
  cudf::strings_column_view const& input,
  cudf::string_scalar const& delimiter = cudf::string_scalar{""},
  rmm::cuda_stream_view stream         = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr    = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the number of tokens in each string of a strings column
 * by using multiple strings delimiters to identify tokens in each string.
 *
 * Also, any consecutive delimiters found in a string are ignored.
 * This means that only empty strings or null rows will result in a token count of 0.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a", "b c", "d.e:f;"]
 * d = [".", ":", ";"]
 * t = count_tokens(s,d)
 * t is now [1, 1, 3]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 * The number of tokens for a null element is set to 0 in the output column.
 *
 * @throw cudf::logic_error if the delimiters column is empty or contains nulls
 *
 * @param input Strings column to count tokens
 * @param delimiters Strings used to separate each string into tokens
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column of token counts
 */
std::unique_ptr<cudf::column> count_tokens(
  cudf::strings_column_view const& input,
  cudf::strings_column_view const& delimiters,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a single column of strings by converting each character to a string.
 *
 * Each string is converted to multiple strings -- one for each character.
 * Note that a character maybe more than one byte.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello world", null, "goodbye"]
 * t = character_tokenize(s)
 * t is now ["h","e","l","l","o"," ","w","o","r","l","d","g","o","o","d","b","y","e"]
 * @endcode
 *
 * @throw std::invalid_argument if `input` contains nulls
 * @throw std::overflow_error if the output would produce more than max size_type rows
 *
 * @param input Strings column to tokenize
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings columns of tokens
 */
std::unique_ptr<cudf::column> character_tokenize(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a strings column from a strings column of tokens and an
 * associated column of row ids.
 *
 * Multiple tokens from the input column may be combined into a single row (string)
 * in the output column. The tokens are concatenated along with the `separator` string
 * in the order in which they appear in the `row_indices` column.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "world", "one", "two", "three"]
 * r = [0, 0, 1, 1, 1]
 * s1 = detokenize(s,r)
 * s1 is now ["hello world", "one two three"]
 * r = [0, 2, 1, 1, 0]
 * s2 = detokenize(s,r)
 * s2 is now ["hello three", "one two", "world"]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 * The values in `row_indices` are expected to have positive, sequential
 * values without any missing row indices otherwise the output is undefined.
 *
 * @throw cudf::logic_error is `separator` is invalid
 * @throw cudf::logic_error if `row_indices.size() != strings.size()`
 * @throw cudf::logic_error if `row_indices` contains nulls
 *
 * @param input Strings column to detokenize
 * @param row_indices The relative output row index assigned for each token in the input column
 * @param separator String to append after concatenating each token to the proper output row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings columns of tokens
 */
std::unique_ptr<cudf::column> detokenize(
  cudf::strings_column_view const& input,
  cudf::column_view const& row_indices,
  cudf::string_scalar const& separator = cudf::string_scalar(" "),
  rmm::cuda_stream_view stream         = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr    = cudf::get_current_device_resource_ref());

/**
 * @brief Vocabulary object to be used with nvtext::tokenize_with_vocabulary
 *
 * Use nvtext::load_vocabulary to create this object.
 */
struct tokenize_vocabulary {
  /**
   * @brief Vocabulary object constructor
   *
   * Token ids are the row indices within the vocabulary column.
   * Each vocabulary entry is expected to be unique otherwise the behavior is undefined.
   *
   * @throw cudf::logic_error if `vocabulary` contains nulls or is empty
   *
   * @param input Strings for the vocabulary
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   */
  tokenize_vocabulary(cudf::strings_column_view const& input,
                      rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
  ~tokenize_vocabulary();

  struct tokenize_vocabulary_impl;
  tokenize_vocabulary_impl* _impl{};
};

/**
 * @brief Create a tokenize_vocabulary object from a strings column
 *
 * Token ids are the row indices within the vocabulary column.
 * Each vocabulary entry is expected to be unique otherwise the behavior is undefined.
 *
 * @throw cudf::logic_error if `vocabulary` contains nulls or is empty
 *
 * @param input Strings for the vocabulary
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Object to be used with nvtext::tokenize_with_vocabulary
 */
std::unique_ptr<tokenize_vocabulary> load_vocabulary(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the token ids for the input string by looking up each delimited
 * token in the given vocabulary
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello world", "hello there", "there there world", "watch out world"]
 * v = load_vocabulary(["hello", "there", "world"])
 * r = tokenize_with_vocabulary(s,v)
 * r is now [[0,2], [0,1], [1,1,2], [-1,-1,2]]
 * @endcode
 *
 * Any null row entry results in a corresponding null entry in the output
 *
 * @throw cudf::logic_error if `delimiter` is invalid
 *
 * @param input Strings column to tokenize
 * @param vocabulary Used to lookup tokens within `input`
 * @param delimiter Used to identify tokens within `input`
 * @param default_id The token id to be used for tokens not found in the `vocabulary`;
 *                   Default is -1
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Lists column of token ids
 */
std::unique_ptr<cudf::column> tokenize_with_vocabulary(
  cudf::strings_column_view const& input,
  tokenize_vocabulary const& vocabulary,
  cudf::string_scalar const& delimiter,
  cudf::size_type default_id        = -1,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of tokenize group
}  // namespace CUDF_EXPORT nvtext
