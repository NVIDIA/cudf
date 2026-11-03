/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

/**
 * @file
 * @brief Standalone string functions.
 */

namespace CUDF_EXPORT cudf {
namespace strings::detail {

/**
 * @brief This will return true if passed a continuation byte of a UTF-8 character.
 *
 * @param chr Any single byte from a valid UTF-8 character
 * @return true if this is not the first byte of the character
 */
CUDF_HOST_DEVICE constexpr bool is_utf8_continuation_char(unsigned char chr)
{
  // The (0xC0 & 0x80) bit pattern identifies a continuation byte of a character.
  return (chr & 0xC0) == 0x80;
}

/**
 * @brief This will return true if passed the first byte of a UTF-8 character.
 *
 * @param chr Any single byte from a valid UTF-8 character
 * @return true if this the first byte of the character
 */
CUDF_HOST_DEVICE constexpr bool is_begin_utf8_char(unsigned char chr)
{
  return not is_utf8_continuation_char(chr);
}

/**
 * @brief This will return true if the passed in byte could be the start of
 * a valid UTF-8 character.
 *
 * This differs from is_begin_utf8_char(uint8_t) in that byte may not be valid
 * UTF-8, so a more rigorous check is performed.
 *
 * @param byte The byte to be tested
 * @return true if this can be the first byte of a character
 */
CUDF_HOST_DEVICE constexpr bool is_valid_begin_utf8_char(uint8_t byte)
{
  // to be the first byte of a valid (up to 4 byte) UTF-8 char, byte must be one of:
  //  0b0vvvvvvv a 1 byte character
  //  0b110vvvvv start of a 2 byte character
  //  0b1110vvvv start of a 3 byte character
  //  0b11110vvv start of a 4 byte character
  return (byte & 0x80) == 0 || (byte & 0xE0) == 0xC0 || (byte & 0xF0) == 0xE0 ||
         (byte & 0xF8) == 0xF0;
}

/**
 * @brief Returns the number of bytes in the specified character.
 *
 * @param character Single character
 * @return Number of bytes
 */
CUDF_HOST_DEVICE constexpr size_type bytes_in_char_utf8(char_utf8 character)
{
  return 1 + static_cast<size_type>((character & 0x0000'FF00u) > 0) +
         static_cast<size_type>((character & 0x00FF'0000u) > 0) +
         static_cast<size_type>((character & 0xFF00'0000u) > 0);
}

/**
 * @brief Returns the number of bytes used to represent the provided byte.
 *
 * This could be 0 to 4 bytes. 0 is returned for intermediate bytes within a
 * single character. For example, for the two-byte 0xC3A8 single character,
 * the first byte would return 2 and the second byte would return 0.
 *
 * @param byte Byte from an encoded character.
 * @return Number of bytes.
 */
CUDF_HOST_DEVICE constexpr size_type bytes_in_utf8_byte(uint8_t byte)
{
  return 1 + static_cast<size_type>((byte & 0xF0) == 0xF0)  // 4-byte character prefix
         + static_cast<size_type>((byte & 0xE0) == 0xE0)    // 3-byte character prefix
         + static_cast<size_type>((byte & 0xC0) == 0xC0)    // 2-byte character prefix
         - static_cast<size_type>((byte & 0xC0) == 0x80);   // intermediate byte
}

/**
 * @brief Convert a char array into a char_utf8 value.
 *
 * @param str String containing encoded char bytes.
 * @param[out] character Single char_utf8 value.
 * @return The number of bytes in the character
 */
CUDF_HOST_DEVICE constexpr size_type to_char_utf8(char const* str, char_utf8& character)
{
  size_type const chr_width = bytes_in_utf8_byte(static_cast<uint8_t>(*str));

  character = static_cast<char_utf8>(*str++) & 0xFF;
  if (chr_width > 1) {
    character = character << 8;
    character |= (static_cast<char_utf8>(*str++) & 0xFF);  // << 8;
    if (chr_width > 2) {
      character = character << 8;
      character |= (static_cast<char_utf8>(*str++) & 0xFF);  // << 16;
      if (chr_width > 3) {
        character = character << 8;
        character |= (static_cast<char_utf8>(*str++) & 0xFF);  // << 24;
      }
    }
  }
  return chr_width;
}

/**
 * @brief Place a char_utf8 value into a char array.
 *
 * @param character Single character
 * @param[out] str Output array.
 * @return The number of bytes in the character
 */
CUDF_HOST_DEVICE constexpr inline size_type from_char_utf8(char_utf8 character, char* str)
{
  size_type const chr_width = bytes_in_char_utf8(character);
  for (size_type idx = 0; idx < chr_width; ++idx) {
    str[chr_width - idx - 1] = static_cast<char>(character) & 0xFF;
    character                = character >> 8;
  }
  return chr_width;
}

/**
 * @brief Converts a single UTF-8 character into a code-point value that
 * can be used for lookup in the character flags or the character case tables.
 *
 * Note that this could return a 3-byte value outside the supported lookup
 * tables' ranges.
 *
 * @param utf8_char Single UTF-8 character to convert.
 * @return Code-point for the UTF-8 character.
 */
CUDF_HOST_DEVICE constexpr uint32_t utf8_to_codepoint(cudf::char_utf8 utf8_char)
{
  uint32_t unchr = 0;
  if (utf8_char < 0x0000'0080)  // single-byte pass thru
    unchr = utf8_char;
  else if (utf8_char < 0x0000'E000)  // two bytes
  {
    unchr = (utf8_char & 0x1F00) >> 2;  // shift and
    unchr |= (utf8_char & 0x003F);      // unmask
  } else if (utf8_char < 0x00F0'0000)   // three bytes
  {
    unchr = (utf8_char & 0x0F'0000) >> 4;   // get upper 4 bits
    unchr |= (utf8_char & 0x00'3F00) >> 2;  // shift and
    unchr |= (utf8_char & 0x00'003F);       // unmask
  } else if (utf8_char <= 0xF800'0000u)     // four bytes
  {
    unchr = (utf8_char & 0x0700'0000) >> 6;   // upper 3 bits
    unchr |= (utf8_char & 0x003F'0000) >> 4;  // next 6 bits
    unchr |= (utf8_char & 0x0000'3F00) >> 2;  // next 6 bits
    unchr |= (utf8_char & 0x0000'003F);       // unmask
  }
  return unchr;
}

/**
 * @brief Converts a character code-point value into a UTF-8 character.
 *
 * @param unchr Character code-point to convert.
 * @return Single UTF-8 character.
 */
CUDF_HOST_DEVICE constexpr cudf::char_utf8 codepoint_to_utf8(uint32_t unchr)
{
  cudf::char_utf8 utf8 = 0;
  if (unchr < 0x0000'0080)  // single byte utf8
    utf8 = unchr;
  else if (unchr < 0x0000'0800)  // double byte utf8
  {
    utf8 = (unchr << 2) & 0x1F00;  // shift bits for
    utf8 |= (unchr & 0x3F);        // utf8 encoding
    utf8 |= 0x0000'C080;
  } else if (unchr < 0x0001'0000)  // triple byte utf8
  {
    utf8 = (unchr << 4) & 0x0F'0000;   // upper 4 bits
    utf8 |= (unchr << 2) & 0x00'3F00;  // next 6 bits
    utf8 |= (unchr & 0x3F);            // last 6 bits
    utf8 |= 0x00E0'8080;
  } else if (unchr < 0x0011'0000)  // quadruple byte utf8
  {
    utf8 = (unchr << 6) & 0x0700'0000;   // upper 3 bits
    utf8 |= (unchr << 4) & 0x003F'0000;  // next 6 bits
    utf8 |= (unchr << 2) & 0x0000'3F00;  // next 6 bits
    utf8 |= (unchr & 0x3F);              // last 6 bits
    utf8 |= 0xF080'8080u;
  }
  return utf8;
}

}  // namespace strings::detail
}  // namespace CUDF_EXPORT cudf
