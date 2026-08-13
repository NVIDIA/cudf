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
 * @brief APIs for generating character and word ngrams from strings columns.
 */

/**
 * @brief Returns a single column of strings by generating ngrams from
 * a strings column.
 *
 * An ngram is a grouping of 2 or more strings with a separator. For example,
 * generating bigrams groups all adjacent pairs of strings.
 *
 * ```
 * ["a", "bb", "ccc"] would generate bigrams as ["a_bb", "bb_ccc"]
 * and trigrams as ["a_bb_ccc"]
 * ```
 *
 * The size of the output column will be the total number of ngrams generated from
 * the input strings column.
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @throw cudf::logic_error if `ngrams < 2`
 * @throw cudf::logic_error if `separator` is invalid
 * @throw cudf::logic_error if there are not enough strings to generate any ngrams
 *
 * @param input Strings column to tokenize and produce ngrams from
 * @param ngrams The ngram number to generate
 * @param separator The string to use for separating ngram tokens
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings columns of tokens
 */
std::unique_ptr<cudf::column> generate_ngrams(
  cudf::strings_column_view const& input,
  cudf::size_type ngrams,
  cudf::string_scalar const& separator,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generates ngrams of characters within each string
 *
 * Each character of a string is used to build ngrams for the output row.
 * Ngrams are not created across strings.
 *
 * ```
 * ["ab", "cde", "fgh"] would generate bigrams as
 * [["ab"], ["cd", "de"], ["fg", "gh"]]
 * ```
 *
 * All null row entries are ignored and the corresponding output row will be empty.
 *
 * @throw std::invalid_argument if `ngrams < 2`
 * @throw cudf::logic_error if there are not enough characters to generate any ngrams
 *
 * @param input Strings column to produce ngrams from
 * @param ngrams The ngram number to generate.
 *               Default is 2 = bigram.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Lists column of strings
 */
std::unique_ptr<cudf::column> generate_character_ngrams(
  cudf::strings_column_view const& input,
  cudf::size_type ngrams            = 2,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Hashes ngrams of characters within each string
 *
 * Each character of a string used to build the ngrams and ngrams are not
 * produced across adjacent strings rows.
 *
 * ```
 * "abcdefg" would generate ngrams=5 as ["abcde", "bcdef" "cdefg"]
 * ```
 *
 * The ngrams for each string are hashed and returned in a list column where
 * the offsets specify rows of hash values for each string.
 *
 * The size of the child column will be the total number of ngrams generated from
 * the input strings column.
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * The hash algorithm uses MurmurHash32 on each ngram.
 *
 * @throw cudf::logic_error if `ngrams < 2`
 * @throw cudf::logic_error if there are not enough characters to generate any ngrams
 *
 * @param input Strings column to produce ngrams from
 * @param ngrams The ngram number to generate. Default is 5.
 * @param seed The seed value to use with the hash algorithm. Default is 0.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return A lists column of hash values
 */
std::unique_ptr<cudf::column> hash_character_ngrams(
  cudf::strings_column_view const& input,
  cudf::size_type ngrams            = 5,
  uint32_t seed                     = 0,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
