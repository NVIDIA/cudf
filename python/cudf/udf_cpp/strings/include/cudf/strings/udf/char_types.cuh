/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/char_types/char_types_enum.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Returns true if all characters in the string are of the type specified.
 *
 * The output will be false if the string is empty or has at least one character
 * not of the specified type. If all characters fit the type then true is returned.
 *
 * To ignore all but specific types, set the `verify_types` to those types
 * which should be checked. Otherwise, the default `ALL_TYPES` will verify all
 * characters match `types`.
 *
 * @code{.pseudo}
 * Examples:
 * s = ['ab', 'a b', 'a7', 'a B']
 * all_characters_of_type('ab', LOWER) => true
 * all_characters_of_type('a b', LOWER) => false
 * all_characters_of_type('a7b', LOWER) => false
 * all_characters_of_type('aB', LOWER) => false
 * all_characters_of_type('ab', LOWER, LOWER|UPPER) => true
 * all_characters_of_type('a b', LOWER, LOWER|UPPER) => true
 * all_characters_of_type('a7', LOWER, LOWER|UPPER) => true
 * all_characters_of_type('a B', LOWER, LOWER|UPPER) => false
 * @endcode
 *
 * @param flags_table Table of character-type flags
 * @param d_str String for this operation
 * @param types The character types to check in the string
 * @param verify_types Only verify against these character types.
 *                     Default `ALL_TYPES` means return `true`
 *                     iff all characters match `types`.
 * @return True if all characters match the type conditions
 */
__device__ inline bool all_characters_of_type(
  cudf::strings::detail::character_flags_table_type* flags_table,
  string_view d_str,
  string_character_types types,
  string_character_types verify_types = string_character_types::ALL_TYPES)
{
  bool check            = !d_str.empty();  // require at least one character
  size_type check_count = 0;
  for (auto itr = d_str.begin(); check && (itr != d_str.end()); ++itr) {
    auto code_point = cudf::strings::detail::utf8_to_codepoint(*itr);
    // lookup flags in table by code-point
    auto flag = code_point <= 0x00FFFF ? flags_table[code_point] : 0;
    if ((verify_types & flag) ||                   // should flag be verified
        (flag == 0 && verify_types == ALL_TYPES))  // special edge case
    {
      check = (types & flag) > 0;
      ++check_count;
    }
  }
  return check && (check_count > 0);
}

/**
 * @brief Returns true if all characters are alphabetic only
 *
 * @param flags_table Table required for checking character types
 * @param d_str Input string to check
 * @return True if characters alphabetic
 */
__device__ inline bool is_alpha(cudf::strings::detail::character_flags_table_type* flags_table,
                                string_view d_str)
{ return all_characters_of_type(flags_table, d_str, string_character_types::ALPHA); }

/**
 * @brief Returns true if all characters are alphanumeric only
 *
 * @param flags_table Table required for checking character types
 * @param d_str Input string to check
 * @return True if characters are alphanumeric
 */
__device__ inline bool is_alpha_numeric(
  cudf::strings::detail::character_flags_table_type* flags_table, string_view d_str)
{ return all_characters_of_type(flags_table, d_str, string_character_types::ALPHANUM); }

/**
 * @brief Returns true if all characters are numeric only
 *
 * @param flags_table Table required for checking character types
 * @param d_str Input string to check
 * @return True if characters are numeric
 */
__device__ inline bool is_numeric(cudf::strings::detail::character_flags_table_type* flags_table,
                                  string_view d_str)
{ return all_characters_of_type(flags_table, d_str, string_character_types::NUMERIC); }

/**
 * @brief Returns true if all characters are digits only
 *
 * @param flags_table Table required for checking character types
 * @param d_str Input string to check
 * @return True if characters are digits
 */
__device__ inline bool is_digit(cudf::strings::detail::character_flags_table_type* flags_table,
                                string_view d_str)
{ return all_characters_of_type(flags_table, d_str, string_character_types::DIGIT); }

/**
 * @brief Returns true if all characters are decimal only
 *
 * @param flags_table Table required for checking character types
 * @param d_str Input string to check
 * @return True if characters are decimal
 */
__device__ inline bool is_decimal(cudf::strings::detail::character_flags_table_type* flags_table,
                                  string_view d_str)
{ return all_characters_of_type(flags_table, d_str, string_character_types::DECIMAL); }

/**
 * @brief Returns true if all characters are spaces only
 *
 * @param flags_table Table required for checking character types
 * @param d_str Input string to check
 * @return True if characters spaces
 */
__device__ inline bool is_space(cudf::strings::detail::character_flags_table_type* flags_table,
                                string_view d_str)
{ return all_characters_of_type(flags_table, d_str, string_character_types::SPACE); }

/**
 * @brief Returns true if all characters are upper case only
 *
 * @param flags_table Table required for checking character types
 * @param d_str Input string to check
 * @return True if characters are upper case
 */
__device__ inline bool is_upper(cudf::strings::detail::character_flags_table_type* flags_table,
                                string_view d_str)
{
  return all_characters_of_type(
    flags_table, d_str, string_character_types::UPPER, string_character_types::CASE_TYPES);
}

/**
 * @brief Returns true if all characters are lower case only
 *
 * @param flags_table Table required for checking character types
 * @param d_str Input string to check
 * @return True if characters are lower case
 */
__device__ inline bool is_lower(cudf::strings::detail::character_flags_table_type* flags_table,
                                string_view d_str)
{
  return all_characters_of_type(
    flags_table, d_str, string_character_types::LOWER, string_character_types::CASE_TYPES);
}

/**
 * @brief Returns true if string is in title case
 *
 * @param tables The char tables required for checking characters
 * @param d_str Input string to check
 * @return True if string is in title case
 */
__device__ inline bool is_title(cudf::strings::detail::character_flags_table_type* flags_table,
                                string_view d_str)
{
  auto valid                 = false;  // requires one or more cased characters
  auto should_be_capitalized = true;   // current character should be upper-case
  for (auto const chr : d_str) {
    auto const code_point = cudf::strings::detail::utf8_to_codepoint(chr);
    auto const flag       = code_point <= 0x00FFFF ? flags_table[code_point] : 0;
    if (cudf::strings::detail::IS_UPPER_OR_LOWER(flag)) {
      if (should_be_capitalized == !cudf::strings::detail::IS_UPPER(flag)) return false;
      valid = true;
    }
    should_be_capitalized = !cudf::strings::detail::IS_UPPER_OR_LOWER(flag);
  }
  return valid;
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
