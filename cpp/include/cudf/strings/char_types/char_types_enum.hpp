/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <type_traits>

/**
 * @file
 * @brief Enum of character type bitmask values used to classify string characters
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_types
 * @{
 */

/**
 * @brief Character type values.
 * These types can be or'd to check for any combination of types.
 *
 * This cannot be turned into an enum class because or'd entries can
 * result in values that are not in the class. For example,
 * combining NUMERIC|SPACE is a valid, reasonable combination but
 * does not match to any explicitly named enumerator.
 */
enum string_character_types : uint32_t {
  DECIMAL    = 1 << 0,                             ///< all decimal characters
  NUMERIC    = 1 << 1,                             ///< all numeric characters
  DIGIT      = 1 << 2,                             ///< all digit characters
  ALPHA      = 1 << 3,                             ///< all alphabetic characters
  SPACE      = 1 << 4,                             ///< all space characters
  UPPER      = 1 << 5,                             ///< all upper case characters
  LOWER      = 1 << 6,                             ///< all lower case characters
  ALPHANUM   = DECIMAL | NUMERIC | DIGIT | ALPHA,  ///< all alphanumeric characters
  CASE_TYPES = UPPER | LOWER,                      ///< all case-able characters
  ALL_TYPES  = ALPHANUM | CASE_TYPES | SPACE       ///< all character types
};

/**
 * @brief OR operator for combining string_character_types
 *
 * @param lhs left-hand side of OR operation
 * @param rhs right-hand side of OR operation
 * @return combined string_character_types
 */
constexpr string_character_types operator|(string_character_types lhs, string_character_types rhs)
{
  return static_cast<string_character_types>(
    static_cast<std::underlying_type_t<string_character_types>>(lhs) |
    static_cast<std::underlying_type_t<string_character_types>>(rhs));
}

/**
 * @brief Compound assignment OR operator for combining string_character_types
 *
 * @param lhs left-hand side of OR operation
 * @param rhs right-hand side of OR operation
 * @return Reference to `lhs` after combining `lhs` and `rhs`
 */
constexpr string_character_types& operator|=(string_character_types& lhs,
                                             string_character_types rhs)
{
  lhs = static_cast<string_character_types>(
    static_cast<std::underlying_type_t<string_character_types>>(lhs) |
    static_cast<std::underlying_type_t<string_character_types>>(rhs));
  return lhs;
}

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
