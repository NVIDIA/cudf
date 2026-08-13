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
 * @addtogroup nvtext_stemmer
 * @{
 * @file
 * @brief APIs for stemming words using the Porter stemming algorithm measure.
 */

/**
 * @brief Used for specifying letter type to check.
 */
enum class letter_type {
  CONSONANT,  ///< Letter is a consonant
  VOWEL       ///< Letter is not a consonant
};

/**
 * @brief Returns boolean column indicating if `character_index` of the input strings
 * is a consonant or vowel.
 *
 * Determining consonants and vowels is described in the following
 * paper: https://tartarus.org/martin/PorterStemmer/def.txt
 *
 * Each string in the input column is expected to contain a single, lower-cased
 * word (or subword) with no punctuation and no whitespace otherwise the
 * measure value for that string is undefined.
 *
 * Also, the algorithm only works with English words.
 *
 * @code{.pseudo}
 * Example:
 * st = ["trouble", "toy", "syzygy"]
 * b1 = is_letter(st, VOWEL, 1)
 * b1 is now [false, true, true]
 * @endcode
 *
 * A negative index value will check the character starting from the end
 * of each string. That is, for `character_index < 0` the letter checked for string
 * `input[i]` is at position `input[i].length + index`.
 *
 * @code{.pseudo}
 * Example:
 * st = ["trouble", "toy", "syzygy"]
 * b2 = is_letter(st, CONSONANT, -1) // last letter checked in each string
 * b2 is now [false, true, false]
 * @endcode
 *
 * A null input element at row `i` produces a corresponding null entry
 * for row `i` in the output column.
 *
 * @param input Strings column of words to measure
 * @param ltype Specify letter type to check
 * @param character_index The character position to check in each string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New BOOL column
 */
std::unique_ptr<cudf::column> is_letter(
  cudf::strings_column_view const& input,
  letter_type ltype,
  cudf::size_type character_index,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns boolean column indicating if character at `indices[i]` of `input[i]`
 * is a consonant or vowel.
 *
 * Determining consonants and vowels is described in the following
 * paper: https://tartarus.org/martin/PorterStemmer/def.txt
 *
 * Each string in the input column is expected to contain a single, lower-cased
 * word (or subword) with no punctuation and no whitespace otherwise the
 * measure value for that string is undefined.
 *
 * Also, the algorithm only works with English words.
 *
 * @code{.pseudo}
 * Example:
 * st = ["trouble", "toy", "syzygy"]
 * ix = [3, 1, 4]
 * b1 = is_letter(st, VOWEL, ix)
 * b1 is now [true, true, false]
 * @endcode
 *
 * A negative index value will check the character starting from the end
 * of each string. That is, for `character_index < 0` the letter checked for string
 * `strings[i]` is at position `strings[i].length + indices[i]`.
 *
 * @code{.pseudo}
 * Example:
 * st = ["trouble", "toy", "syzygy"]
 * ix = [3, -2, 4] // 2nd to last character in st[1] is checked
 * b2 = is_letter(st, CONSONANT, ix)
 * b2 is now [false, false, true]
 * @endcode
 *
 * A null input element at row `i` produces a corresponding null entry
 * for row `i` in the output column.
 *
 * @throw cudf::logic_error if `indices.size() != input.size()`
 * @throw cudf::logic_error if `indices` contain nulls.
 *
 * @param input Strings column of words to measure
 * @param ltype Specify letter type to check
 * @param indices The character positions to check in each string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New BOOL column
 */
std::unique_ptr<cudf::column> is_letter(
  cudf::strings_column_view const& input,
  letter_type ltype,
  cudf::column_view const& indices,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the Porter Stemmer measurements of a strings column.
 *
 * Porter stemming is used to normalize words by removing plural and tense endings
 * from words in English. The stemming measurement involves counting consonant/vowel
 * patterns within a string.
 * Reference paper: https://tartarus.org/martin/PorterStemmer/def.txt
 *
 * Each string in the input column is expected to contain a single, lower-cased
 * word (or subword) with no punctuation and no whitespace otherwise the
 * measure value for that string is undefined.
 *
 * Also, the algorithm only works with English words.
 *
 * @code{.pseudo}
 * Example:
 * st = ["tr", "troubles", "trouble"]
 * m = porter_stemmer_measure(st)
 * m is now [0,2,1]
 * @endcode
 *
 * A null input element at row `i` produces a corresponding null entry
 * for row `i` in the output column.
 *
 * @param input Strings column of words to measure
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return New INT32 column of measure values
 */
std::unique_ptr<cudf::column> porter_stemmer_measure(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
