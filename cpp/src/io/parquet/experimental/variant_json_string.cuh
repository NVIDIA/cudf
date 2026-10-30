/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/limits>

#include <cstdint>

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental::detail {

struct string_stats {
  int64_t utf8_bytes;
  int64_t utf16_units;
};

__device__ inline int hex_digit(char c)
{
  if (c >= '0' && c <= '9') { return c - '0'; }
  if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
  return c - 'A' + 10;
}

__device__ inline uint32_t read_hex4(char const* s)
{
  return (hex_digit(s[0]) << 12) | (hex_digit(s[1]) << 8) | (hex_digit(s[2]) << 4) |
         hex_digit(s[3]);
}

/**
 * @brief Decodes the escapes of a validated JSON string body into UTF-8.
 *
 * Matches Java semantics: a surrogate escape that is not part of a valid pair becomes '?' when the
 * decoded `String` is encoded as UTF-8.
 *
 * @tparam write Whether to write the decoded bytes to `out`
 */
template <bool write>
__device__ string_stats decode_json_string(char const* begin, char const* end, uint8_t* out)
{
  int64_t num_bytes = 0;
  int64_t num_units = 0;
  auto emit         = [&](uint8_t b) {
    if constexpr (write) { out[num_bytes] = b; }
    ++num_bytes;
  };
  auto s = begin;
  while (s < end) {
    auto const c = static_cast<uint8_t>(*s);
    if (c != '\\') {
      emit(c);
      if ((c & 0xC0) != 0x80) { num_units += ((c & 0xF8) == 0xF0) ? 2 : 1; }
      ++s;
      continue;
    }
    auto const esc = s[1];
    s += 2;
    ++num_units;
    switch (esc) {
      case 'b': emit('\b'); break;
      case 'f': emit('\f'); break;
      case 'n': emit('\n'); break;
      case 'r': emit('\r'); break;
      case 't': emit('\t'); break;
      case 'u': {
        uint32_t cp = read_hex4(s);
        s += 4;
        if (cp >= 0xD800 && cp <= 0xDFFF) {
          bool paired = false;
          if (cp <= 0xDBFF && end - s >= 6 && s[0] == '\\' && s[1] == 'u') {
            auto const low = read_hex4(s + 2);
            if (low >= 0xDC00 && low <= 0xDFFF) {
              cp     = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
              paired = true;
              s += 6;
              ++num_units;
            }
          }
          if (!paired) {
            emit('?');
            break;
          }
        }
        if (cp < 0x80) {
          emit(static_cast<uint8_t>(cp));
        } else if (cp < 0x800) {
          emit(0xC0 | (cp >> 6));
          emit(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
          emit(0xE0 | (cp >> 12));
          emit(0x80 | ((cp >> 6) & 0x3F));
          emit(0x80 | (cp & 0x3F));
        } else {
          emit(0xF0 | (cp >> 18));
          emit(0x80 | ((cp >> 12) & 0x3F));
          emit(0x80 | ((cp >> 6) & 0x3F));
          emit(0x80 | (cp & 0x3F));
        }
        break;
      }
      default: emit(static_cast<uint8_t>(esc)); break;
    }
  }
  return {num_bytes, num_units};
}

/**
 * @brief Iterates over the UTF-16 code units of a UTF-8 string.
 */
struct utf16_reader {
  uint8_t const* data;
  int64_t size;
  int64_t pos      = 0;
  int32_t low_unit = -1;

  /// Returns the next code unit, or -1 at the end of the string
  __device__ int32_t next()
  {
    if (low_unit >= 0) {
      auto const unit = low_unit;
      low_unit        = -1;
      return unit;
    }
    if (pos >= size) { return -1; }
    uint32_t const c = data[pos];
    uint32_t cp;
    if (c < 0x80) {
      cp = c;
      pos += 1;
    } else if (c < 0xE0) {
      cp = ((c & 0x1F) << 6) | (data[pos + 1] & 0x3F);
      pos += 2;
    } else if (c < 0xF0) {
      cp = ((c & 0x0F) << 12) | ((data[pos + 1] & 0x3F) << 6) | (data[pos + 2] & 0x3F);
      pos += 3;
    } else {
      cp = ((c & 0x07) << 18) | ((data[pos + 1] & 0x3F) << 12) | ((data[pos + 2] & 0x3F) << 6) |
           (data[pos + 3] & 0x3F);
      pos += 4;
    }
    if (cp < 0x10000) { return static_cast<int32_t>(cp); }
    cp -= 0x10000;
    low_unit = static_cast<int32_t>(0xDC00 + (cp & 0x3FF));
    return static_cast<int32_t>(0xD800 + (cp >> 10));
  }
};

/**
 * @brief Three-way comparison in Java `String.compareTo` order (UTF-16 code units).
 */
__device__ inline int utf16_compare(uint8_t const* a,
                                    int64_t a_size,
                                    uint8_t const* b,
                                    int64_t b_size)
{
  int64_t i = 0;
  while (i < a_size && i < b_size && a[i] == b[i]) {
    ++i;
  }
  if (i == a_size || i == b_size) { return (a_size > i) - (b_size > i); }
  // Restart at the code point containing the first differing byte
  while (i > 0 && (a[i] & 0xC0) == 0x80) {
    --i;
  }
  utf16_reader ra{a + i, a_size - i};
  utf16_reader rb{b + i, b_size - i};
  while (true) {
    auto const ua = ra.next();
    auto const ub = rb.next();
    if (ua != ub) { return ua < ub ? -1 : 1; }
    if (ua < 0) { return 0; }
  }
}

constexpr uint64_t long_key_code = cuda::std::numeric_limits<uint64_t>::max();

/**
 * @brief Encodes a key of at most 8 bytes as an integer that orders like `utf16_compare`.
 *
 * UTF-8 byte order matches UTF-16 order once the leads of U+E000-U+FFFF move above the
 * supplementary leads. Zero padding requires keys without NUL bytes, and lead 0xED is excluded
 * because lone surrogates do not follow this order. Other keys return `long_key_code`.
 */
__device__ inline uint64_t short_key_code(uint8_t const* key, int64_t len)
{
  if (len > 8) { return long_key_code; }
  uint64_t code = 0;
  for (int64_t i = 0; i < 8; ++i) {
    uint8_t b = 0;
    if (i < len) {
      b = key[i];
      if (b == 0 || b == 0xED) { return long_key_code; }
      if (b == 0xEE || b == 0xEF) { b += 0xF5 - 0xEE; }
    }
    code = (code << 8) | b;
  }
  return code;
}

}  // namespace io::parquet::experimental::detail
}  // namespace CUDF_EXPORT cudf
