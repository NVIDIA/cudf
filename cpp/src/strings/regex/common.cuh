/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "regcomp.h"

#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/types.hpp>

#include <cuda/std/optional>
#include <cuda/std/utility>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

namespace cudf::strings::detail {

/// Bitmask type: bit i is set when Glushkov position i is active.
using glushkov_state_t = uint64_t;

/// Maximum number of character-consuming positions (states) in the Glushkov NFA.
/// Patterns with more positions fall back to Thompson NFA automatically.
constexpr int32_t GLUSHKOV_MAX_STATES = sizeof(glushkov_state_t) * 8;

/// Maximum shift amounts for the Hyperscan-style shift-and optimization.
constexpr int32_t GLUSHKOV_MAX_SHIFTS = 8;

/// Size of the precomputed ASCII reach table (characters 0–127).
constexpr int32_t GLUSHKOV_ASCII_TABLE_SIZE = 128;

/**
 * @brief Regex class stored on the device and executed by reprog_device.
 *
 * This class holds the unique data for any regex CCLASS instruction.
 */
struct alignas(16) reclass_device {
  int32_t builtins{};
  int32_t count{};
  reclass_range const* literals{};

  __device__ inline bool is_match(char32_t const ch, uint8_t const* codepoint_flags) const
  {
    if (thrust::any_of(thrust::seq, literals, literals + count, [ch](auto literal) {
          return ((ch >= literal.first) && (ch <= literal.last));
        })) {
      return true;
    }

    if (!builtins) { return false; }
    auto const codept                = utf8_to_codepoint(ch);
    constexpr uint32_t MAX_CODEPOINT = 0x00'FFFF;
    if (codept > MAX_CODEPOINT) { return false; }
    auto const fl = codepoint_flags[codept];
    if ((builtins & CCLASS_W) && ((ch == '_') || IS_ALPHANUM(fl))) { return true; }     // \w
    if ((builtins & CCLASS_S) && IS_SPACE(fl)) { return true; }                         // \s
    if ((builtins & CCLASS_D) && IS_DIGIT(fl)) { return true; }                         // \d
    if ((builtins & NCCLASS_W) && ((ch != '\n') && (ch != '_') && !IS_ALPHANUM(fl))) {  // \W
      return true;
    }
    if ((builtins & NCCLASS_S) && !IS_SPACE(fl)) { return true; }                    // \S
    if ((builtins & NCCLASS_D) && ((ch != '\n') && !IS_DIGIT(fl))) { return true; }  // \D

    return false;
  }
};

/**
 * @brief Check for supported new-line characters
 *
 * '\n, \r, \u0085, \u2028, or \u2029'
 */
CUDF_HOST_DEVICE __forceinline__ constexpr bool is_newline(char32_t const ch)
{
  return (ch == '\n' || ch == '\r' || ch == 0x00c285 || ch == 0x00e280a8 || ch == 0x00e280a9);
}

/**
 * @brief Template type used on `find` to specify desired position values in returned match_result
 */
enum class positional : int8_t {
  BEGIN_END = 0,  /// both begin and end positions are returned
  END_ONLY  = 1,  /// only the end position is returned
};

using match_pair   = cuda::std::pair<cudf::size_type, cudf::size_type>;
using match_result = cuda::std::optional<match_pair>;

}  // namespace cudf::strings::detail
