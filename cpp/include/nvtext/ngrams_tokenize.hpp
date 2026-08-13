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
 * @addtogroup nvtext_ngrams
 * @{
 * @file
 * @brief APIs for tokenizing strings and generating ngrams from the resulting tokens.
 */

/**
 * @brief Returns a single column of strings by tokenizing the input strings
 * column and then producing ngrams of each string.
 *
 * An ngram is a grouping of 2 or more tokens with a separator. For example,
 * generating bigrams groups all adjacent pairs of tokens for a string.
 *
 * ```
 * ["a bb ccc"] can be tokenized to ["a", "bb", "ccc"]
 * bigrams would generate ["a_bb", "bb_ccc"] and trigrams would generate ["a_bb_ccc"]
 * ```
 *
 * The `delimiter` is used for tokenizing and may be zero or more characters.
 * If the `delimiter` is empty, whitespace (character code-point <= ' ') is used
 * for identifying tokens.
 *
 * Once tokens are identified, ngrams are produced by joining the tokens
 * with the specified separator. The generated ngrams use the tokens for each
 * string and not across strings in adjacent rows.
 * Any input string that contains fewer tokens than the specified ngrams value is
 * skipped and will not contribute to the output. Therefore, a bigram of a single
 * token is ignored as well as a trigram of 2 or less tokens.
 *
 * Tokens are found by locating delimiter(s) starting at the beginning of each string.
 * As each string is tokenized, the ngrams are generated using input column row order
 * to build the output column. That is, ngrams created in input row[i] will be placed in
 * the output column directly before ngrams created in input row[i+1].
 *
 * The size of the output column will be the total number of ngrams generated from
 * the input strings column.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a b c", "d e", "f g h i", "j"]
 * t = ngrams_tokenize(s, 2, " ", "_")
 * t is now ["a_b", "b_c", "d_e", "f_g", "g_h", "h_i"]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @param input Strings column to tokenize and produce ngrams from
 * @param ngrams The ngram number to generate
 * @param delimiter UTF-8 characters used to separate each string into tokens.
 *                  An empty string will separate tokens using whitespace.
 * @param separator The string to use for separating ngram tokens
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings columns of tokens
 */
std::unique_ptr<cudf::column> ngrams_tokenize(
  cudf::strings_column_view const& input,
  cudf::size_type ngrams,
  cudf::string_scalar const& delimiter,
  cudf::string_scalar const& separator,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
