/*
 * SPDX-FileCopyrightText: Copyright (C) 2004-2017, 2020-2022 René Nyffenegger
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND Zlib
 */

/*
 * Portions of this file are derived from Rene Nyffenegger's codebase at
 * https://github.com/ReneNyffenegger/cpp-base64, original license text below.
 */

/*
 *  base64.cpp and base64.h
 *
 *  base64 encoding and decoding with C++.
 *  More information at
 *    https://renenyffenegger.ch/notes/development/Base64/Encoding-and-decoding-base-64-with-cpp/
 *
 *  Version: 2.rc.09 (release candidate)
 *
 *  Copyright (C) 2004-2017, 2020-2022 René Nyffenegger
 *
 *  This source code is provided 'as-is', without any express or implied
 *  warranty. In no event will the author be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, subject to the following restrictions:
 *
 *  1. The origin of this source code must not be misrepresented; you must not
 *     claim that you wrote the original source code. If you use this source code
 *     in a product, an acknowledgment in the product documentation would be
 *     appreciated but is not required.
 *
 *  2. Altered source versions must be plainly marked as such, and must not be
 *     misrepresented as being the original source code.
 *
 *  3. This notice may not be removed or altered from any source distribution.
 *
 *  René Nyffenegger rene.nyffenegger@adp-gmbh.ch
 */

/**
 * @file base64_utils.cpp
 * @brief base64 string encoding/decoding implementation
 */

// altered: applying clang-format for libcudf on this file.

#include "base64_utilities.hpp"

#include <cudf/logger.hpp>

#include <cuda/iterator>

#include <algorithm>

// altered: use cudf namespaces
namespace cudf::io::detail {

static std::string const base64_chars =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  "abcdefghijklmnopqrstuvwxyz"
  "0123456789+/";

static constexpr unsigned char trailing_char = '=';

// Function to encode input string to base64 and return the encoded string
std::string base64_encode(std::string_view string_to_encode)
{
  auto input_length = static_cast<int32_t>(string_to_encode.size());

  // altered: compute number of encoding iterations = floor(multiple of 3)
  int32_t num_iterations = (input_length / 3);
  num_iterations += (input_length % 3) ? 1 : 0;

  std::string encoded;
  size_t const encoded_length = (input_length + 2) / 3 * 4;
  encoded.reserve(encoded_length);

  // altered: modify base64 encoder loop using STL and Thrust.
  // TODO: Port this loop to thrust cooperative groups if needed for too-wide tables.
  std::for_each(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator{num_iterations}, [&](auto&& iter) {
      auto idx = iter * 3;

      encoded.push_back(base64_chars[(string_to_encode[idx] & 0xfc) >> 2]);
      // increment the index by 1
      idx += 1;

      if (idx < input_length) {
        encoded.push_back(base64_chars[((string_to_encode[idx - 1] & 0x03) << 4) +
                                       ((string_to_encode[idx] & 0xf0) >> 4)]);
        // increment the index by 1
        idx += 1;

        if (idx < input_length) {
          encoded.push_back(base64_chars[((string_to_encode[idx - 1] & 0x0f) << 2) +
                                         ((string_to_encode[idx] & 0xc0) >> 6)]);
          encoded.push_back(base64_chars[string_to_encode[idx] & 0x3f]);
        } else {
          encoded.push_back(base64_chars[(string_to_encode[idx - 1] & 0x0f) << 2]);
          encoded.push_back(trailing_char);
        }
      } else {
        encoded.push_back(base64_chars[(string_to_encode[idx - 1] & 0x03) << 4]);
        encoded.push_back(trailing_char);
        encoded.push_back(trailing_char);
      }
    });

  return encoded;
}

// base64 decode function
std::string base64_decode(std::string_view encoded_string)
{
  // altered: there must be at least 2 characters in the base64-encoded string
  if (encoded_string.size() < 2) {
    CUDF_LOG_ERROR(
      "Parquet reader encountered invalid base64-encoded string size."
      "arrow:schema not processed.");
    return std::string{};
  }

  size_t const input_length = encoded_string.length();
  std::string decoded;

  // altered: compute number of decoding iterations = floor (multiple of 4)
  int32_t num_iterations = (input_length / 4);
  num_iterations += (input_length % 4) ? 1 : 0;

  //
  // The approximate length (bytes) of the decoded string might be one or
  // two bytes smaller, depending on the amount of trailing equal signs
  // in the encoded string. This approximation is needed to reserve
  // enough space in the string to be returned.
  size_t const approx_decoded_length = input_length / 4 * 3;
  decoded.reserve(approx_decoded_length);

  //
  // Iterate over encoded input string in chunks. The size of all
  // chunks except the last one is 4 bytes.
  //
  // The last chunk might be padded with equal signs or dots
  // in order to make it 4 bytes in size as well, but this
  // is not required as per RFC 2045.
  //
  // All chunks except the last one produce three output bytes.
  //
  // The last chunk produces at least one and up to three bytes.
  //
  // altered: modify base64 encoder loop to number of iterations using STL and Thrust.
  // TODO: Port this loop to thrust cooperative groups if needed for too-wide tables.
  if (not std::all_of(
        cuda::counting_iterator<int32_t>{0},
        cuda::counting_iterator{num_iterations},
        [&](auto&& iter) {
          int32_t idx                  = iter * 4;
          size_t current_char_position = 0;
          size_t char1_position        = 0;
          size_t char2_position        = 0;

          // Check for data that is not padded with equal
          // signs (which is allowed by RFC 2045)
          if (encoded_string[idx] == '=') { return true; }

          current_char_position = base64_chars.find(encoded_string[idx]);
          char1_position        = base64_chars.find(encoded_string[idx + 1]);
          if (current_char_position == std::string::npos or char1_position == std::string::npos) {
            return false;
          }
          // Emit the first output byte that is produced in each chunk:
          decoded.push_back(static_cast<std::string::value_type>((current_char_position << 2) +
                                                                 ((char1_position & 0x30) >> 4)));

          // increment the index by 1
          idx += 1;
          // check for = padding
          if (encoded_string[idx] == '=') { return true; }

          // increment the index by 1
          idx += 1;
          // check for = padding
          if (encoded_string[idx] == '=') { return true; }

          char1_position = base64_chars.find(encoded_string[idx - 1]);
          char2_position = base64_chars.find(encoded_string[idx]);
          if (char1_position == std::string::npos or char2_position == std::string::npos) {
            return false;
          }
          // Emit a chunk's second byte (which might not be produced in the last
          // chunk).
          decoded.push_back(static_cast<std::string::value_type>(((char1_position & 0x0f) << 4) +
                                                                 ((char2_position & 0x3c) >> 2)));

          // increment the index by 1
          idx += 1;
          // check for = padding
          if (encoded_string[idx] == '=') { return true; }

          char2_position        = base64_chars.find(encoded_string[idx - 1]);
          current_char_position = base64_chars.find(encoded_string[idx]);
          if (current_char_position == std::string::npos or char2_position == std::string::npos) {
            return false;
          }
          // Emit a chunk's third byte (which might not be produced in the last
          // chunk).
          decoded.push_back(static_cast<std::string::value_type>(((char2_position & 0x03) << 6) +
                                                                 current_char_position));

          // all good, return true
          return true;
        })) {
    return std::string{};
  }

  // return the decoded string
  return decoded;
}

}  // namespace cudf::io::detail
