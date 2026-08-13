/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief APIs for WordPiece tokenization of strings columns using a vocabulary.
 */

/**
 * @brief Vocabulary object to be used with nvtext::wordpiece_tokenizer
 *
 * Use nvtext::load_wordpiece_vocabulary to create this object.
 */
struct wordpiece_vocabulary {
  /**
   * @brief Vocabulary object constructor
   *
   * Token ids are the row indices within the vocabulary column.
   * Each vocabulary entry is expected to be unique otherwise the behavior is undefined.
   *
   * @throw std::invalid_argument if `vocabulary` contains nulls or is empty
   *
   * @param input Strings for the vocabulary
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   */
  wordpiece_vocabulary(cudf::strings_column_view const& input,
                       rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
  ~wordpiece_vocabulary();

  struct wordpiece_vocabulary_impl;
  std::unique_ptr<wordpiece_vocabulary_impl> _impl;
};

/**
 * @brief Create a tokenize_vocabulary object from a strings column
 *
 * Token ids are the row indices within the vocabulary column.
 * Each vocabulary entry is expected to be unique otherwise the behavior is undefined.
 *
 * @throw std::invalid_argument if `vocabulary` contains nulls or is empty
 *
 * @param input Strings for the vocabulary
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Object to be used with nvtext::wordpiece_tokenize
 */
std::unique_ptr<wordpiece_vocabulary> load_wordpiece_vocabulary(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the token ids for the input string a wordpiece tokenizer
 * algorithm with the given vocabulary
 *
 * Example:
 * @code{.pseudo}
 * vocabulary = ["[UNK]", "a", "have", "I", "new", "GP", "##U", "!"]
 * v = load_wordpiece_vocabulary(vocabulary)
 * input = ["I have a new GPU now !"]
 * t = wordpiece_tokenize(i,v)
 * t is now [[3, 2, 1, 4, 5, 6, 0, 7]]
 * @endcode
 *
 * The `max_words_per_row` also optionally limits the output by only processing
 * a maximum number of words per row. Here a word is defined as consecutive
 * sequence of characters delimited by space character(s).
 *
 * Example:
 * @code{.pseudo}
 * vocabulary = ["[UNK]", "a", "have", "I", "new", "GP", "##U", "!"]
 * v = load_wordpiece_vocabulary(vocabulary)
 * input = ["I have  a new GPU now !"]
 * t4 = wordpiece_tokenize(i,v,4)
 * t4 is now [[3, 2, 1, 4]]
 * t5 = wordpiece_tokenize(i,v,5)
 * t5 is now [[3, 2, 1, 4, 5, 6]]
 * @endcode
 *
 * Any null row entry results in a corresponding null entry in the output.
 *
 * @throw std::invalid_argument If `max_words_per_row` is less than 0.
 *
 * @param input Strings column to tokenize
 * @param vocabulary Used to lookup tokens within `input`
 * @param max_words_per_row Maximum number of words to tokenize for each row.
 *                          Default 0 tokenizes all words.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Lists column of token ids
 */
std::unique_ptr<cudf::column> wordpiece_tokenize(
  cudf::strings_column_view const& input,
  wordpiece_vocabulary const& vocabulary,
  cudf::size_type max_words_per_row = 0,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of tokenize group
}  // namespace CUDF_EXPORT nvtext
