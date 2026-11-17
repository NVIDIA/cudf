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
 * @brief base64 string encoding/decoding utilities
 */

#pragma once

// altered: applying clang-format for libcudf on this file.

// altered: include required headers
#include <cudf/utilities/export.hpp>

#include <string>

// altered: use cudf namespaces
namespace CUDF_EXPORT cudf {
namespace io::detail {

/**
 * @brief Encodes input string to base64 and returns it
 *
 * @param string_to_encode a view of the string to be encoded in base64
 * @return the base64-encoded string
 *
 */
std::string base64_encode(std::string_view string_to_encode);

/**
 * @brief Decodes the input base64-encoded string and returns it
 *
 * @param encoded_string a view of the base64-encoded string to be decoded
 * @return the decoded string
 *
 */
std::string base64_decode(std::string_view encoded_string);

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
