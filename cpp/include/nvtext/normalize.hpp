/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

//! NVText APIs
namespace CUDF_EXPORT nvtext {
/**
 * @addtogroup nvtext_normalize
 * @{
 * @file
 * @brief APIs for normalizing whitespace and characters within strings columns.
 */

/**
 * @brief Returns a new strings column by normalizing the whitespace in each
 * string in the input column.
 *
 * Normalizing a string replaces any number of whitespace character
 * (character code-point <= ' ') runs with a single space ' ' and
 * trims whitespace from the beginning and end of the string.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a b", "  c  d\n", "e \t f "]
 * t = normalize_spaces(s)
 * t is now ["a b","c d","e f"]
 * @endcode
 *
 * A null input element at row `i` produces a corresponding null entry
 * for row `i` in the output column.
 *
 * @param input Strings column to normalize
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return New strings columns of normalized strings.
 */
std::unique_ptr<cudf::column> normalize_spaces(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Normalizer object to be used with nvtext::normalize_characters
 *
 * Use nvtext::create_normalizer to create this object.
 *
 * This normalizer includes:
 *
 * - adding padding around punctuation (unicode category starts with "P")
 *   as well as certain ASCII symbols like "^" and "$"
 * - adding padding around the [CJK Unicode block
 * characters](https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block))
 * - changing whitespace (e.g. `"\t", "\n", "\r"`) to just space `" "`
 * - removing control characters (unicode categories "Cc" and "Cf")
 *
 * The padding process adds a single space before and after the character.
 * Details on _unicode category_ can be found here:
 * https://unicodebook.readthedocs.io/unicode.html#categories
 *
 * If `do_lower_case = true`, lower-casing also removes any accents. The
 * accents cannot be removed from upper-case characters without lower-casing
 * and lower-casing cannot be performed without also removing accents.
 * However, if the accented character is already lower-case, then only the
 * accent is removed.
 *
 * If `special_tokens` are included the padding after `[` and before `]` is not
 * inserted if the characters between them match one of the given tokens.
 * Also, the `special_tokens` are expected to include the `[]` characters
 * at the beginning of and end of each string appropriately.
 */
struct character_normalizer {
  /**
   * @brief Normalizer object constructor
   *
   * This initializes and holds the character normalizing tables and settings.
   * The special tokens are expected to all upper case regardless of the
   * `do_lower_case` flag.
   *
   * @param do_lower_case If true, upper-case characters are converted to
   *        lower-case and accents are stripped from those characters.
   *        If false, accented and upper-case characters are not transformed.
   * @param special_tokens Each row is a token including the `[]` brackets.
   *        For example: `[BOS]`, `[EOS]`, `[UNK]`, `[SEP]`, `[PAD]`, `[CLS]`, `[MASK]`
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   */
  character_normalizer(bool do_lower_case,
                       cudf::strings_column_view const& special_tokens,
                       rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
  ~character_normalizer();

  struct character_normalizer_impl;
  std::unique_ptr<character_normalizer_impl> _impl;
};

/**
 * @brief Create a normalizer object
 *
 * Creates a normalizer object which can be reused on multiple calls to
 * nvtext::normalize_characters
 *
 * @see nvtext::character_normalizer
 *
 * @param do_lower_case If true, upper-case characters are converted to
 *        lower-case and accents are stripped from those characters.
 *        If false, accented and upper-case characters are not transformed.
 * @param special_tokens Individual tokens including `[]` brackets.
 *        Default is no special tokens.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Object to be used with nvtext::normalize_characters
 */
std::unique_ptr<character_normalizer> create_character_normalizer(
  bool do_lower_case,
  cudf::strings_column_view const& special_tokens = cudf::strings_column_view(cudf::column_view{
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0}),
  rmm::cuda_stream_view stream                    = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr               = cudf::get_current_device_resource_ref());

/**
 * @brief Normalizes the text in input strings column
 *
 * @see nvtext::character_normalizer for details on the normalizer behavior
 *
 * @code{.pseudo}
 * cn = create_character_normalizer(true)
 * s = ["éâîô\teaio", "ĂĆĖÑÜ", "ACENU", "$24.08", "[a,bb]"]
 * s1 = normalize_characters(s,cn)
 * s1 is now ["eaio eaio", "acenu", "acenu", " $ 24 . 08", " [ a , bb ] "]
 *
 * cn = create_character_normalizer(false)
 * s2 = normalize_characters(s,cn)
 * s2 is now ["éâîô eaio", "ĂĆĖÑÜ", "ACENU", " $ 24 . 08", " [ a , bb ] "]
 * @endcode
 *
 * A null input element at row `i` produces a corresponding null entry
 * for row `i` in the output column.
 *
 * @param input The input strings to normalize
 * @param normalizer Normalizer to use for this function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resource to allocate any returned objects
 * @return Normalized strings column
 */
std::unique_ptr<cudf::column> normalize_characters(
  cudf::strings_column_view const& input,
  character_normalizer const& normalizer,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
