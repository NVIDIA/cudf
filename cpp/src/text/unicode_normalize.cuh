/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <cstdint>

namespace nvtext {
namespace detail {

// Maximum Unicode codepoint value
constexpr uint32_t MAX_CODEPOINT = 0x10'FFFFu;

// Size of codepoint-indexed tables (one slot per possible codepoint)
constexpr uint32_t CODEPOINT_TABLE_SIZE = MAX_CODEPOINT + 1u;  // 1,114,112

// Size of decomp_offsets: one extra slot to hold the total count
constexpr uint32_t DECOMP_OFFSETS_SIZE = MAX_CODEPOINT + 2u;  // 1,114,114

// Maximum transitive decomposition depth in Unicode (empirically ~4, bounded at 5)
constexpr int32_t MAX_DECOMP_DEPTH = 5;

// Maximum codepoints a single input codepoint can expand to after full NFKD decomposition
constexpr int32_t MAX_DECOMP_EXPAND = 18;

// Hangul algorithmic decomposition constants (Unicode 3.1+)
constexpr uint32_t HANGUL_SBASE  = 0xAC00u;
constexpr uint32_t HANGUL_LBASE  = 0x1100u;
constexpr uint32_t HANGUL_VBASE  = 0x1161u;
constexpr uint32_t HANGUL_TBASE  = 0x11A7u;
constexpr uint32_t HANGUL_LCOUNT = 19u;
constexpr uint32_t HANGUL_VCOUNT = 21u;
constexpr uint32_t HANGUL_TCOUNT = 28u;
constexpr uint32_t HANGUL_NCOUNT = HANGUL_VCOUNT * HANGUL_TCOUNT;      // 588
constexpr uint32_t HANGUL_SCOUNT = HANGUL_LCOUNT * HANGUL_NCOUNT;      // 11172
constexpr uint32_t HANGUL_SEND   = HANGUL_SBASE + HANGUL_SCOUNT - 1u;  // 0xD7A3

// Hangul Jamo ranges for composition detection
constexpr uint32_t HANGUL_LEND   = HANGUL_LBASE + HANGUL_LCOUNT - 1u;  // 0x1112
constexpr uint32_t HANGUL_VEND   = HANGUL_VBASE + HANGUL_VCOUNT - 1u;  // 0x1175
constexpr uint32_t HANGUL_TSTART = HANGUL_TBASE + 1u;                  // 0x11A8
constexpr uint32_t HANGUL_TEND   = HANGUL_TBASE + HANGUL_TCOUNT - 1u;  // 0x11C2

/**
 * @brief Decompose a Hangul syllable codepoint into its constituent jamo.
 *
 * Writes 2 or 3 codepoints into @p out and returns the count written.
 * Caller must ensure @p out has space for 3 uint32_t values.
 */
__device__ inline int hangul_decompose(uint32_t cp, uint32_t* out)
{
  auto const sindex = cp - HANGUL_SBASE;
  auto const tindex = sindex % HANGUL_TCOUNT;
  out[0]            = HANGUL_LBASE + sindex / HANGUL_NCOUNT;
  out[1]            = HANGUL_VBASE + (sindex % HANGUL_NCOUNT) / HANGUL_TCOUNT;
  if (tindex > 0) {
    out[2] = HANGUL_TBASE + tindex;
    return 3;
  }
  return 2;
}

/**
 * @brief Attempt Hangul composition of two adjacent codepoints.
 *
 * @return Composed codepoint if composition succeeded, 0 otherwise.
 */
__device__ inline uint32_t hangul_compose(uint32_t a, uint32_t b)
{
  // L + V → LV
  if (a >= HANGUL_LBASE && a <= HANGUL_LEND && b >= HANGUL_VBASE && b <= HANGUL_VEND) {
    return HANGUL_SBASE + ((a - HANGUL_LBASE) * HANGUL_NCOUNT) +
           ((b - HANGUL_VBASE) * HANGUL_TCOUNT);
  }
  // LV + T → LVT
  if (a >= HANGUL_SBASE && a <= HANGUL_SEND && ((a - HANGUL_SBASE) % HANGUL_TCOUNT == 0) &&
      b >= HANGUL_TSTART && b <= HANGUL_TEND) {
    return a + (b - HANGUL_TBASE);
  }
  return 0;
}

/**
 * @brief Parse a single uppercase hex token from a device string.
 *
 * @param s Pointer to first character of the token
 * @param size Number of characters to parse
 * @return Parsed uint32_t codepoint value
 */
__device__ inline uint32_t hex_to_cp(char const* s, cudf::size_type size)
{
  uint32_t val = 0;
  for (cudf::size_type i = 0; i < size; ++i) {
    char const c         = s[i];
    uint32_t const digit = [c]() -> uint32_t {
      if (c >= '0' && c <= '9') { return static_cast<uint32_t>(c - '0'); }
      if (c >= 'A' && c <= 'F') { return static_cast<uint32_t>(c - 'A' + 10); }
      if (c >= 'a' && c <= 'f') { return static_cast<uint32_t>(c - 'a' + 10); }
      return 0u;
    }();
    val = (val << 4u) | digit;
  }
  return val;
}

}  // namespace detail
}  // namespace nvtext
