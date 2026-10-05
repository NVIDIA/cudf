
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "udf_string.cuh"

#include <cudf/strings/detail/strip.cuh>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Strip characters from the beginning and/or end of the given string
 *
 * The `d_to_strip` is interpreted as an array of characters to be removed.
 * If `d_to_strip` is an empty string, whitespace characters are stripped.
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" aba ", 5};
 * auto d_to_strip = cudf::string_view{}; // empty string
 * auto result = strip(d_str, d_to_strip);
 * // result is "aba"
 * d_to_strip = cudf::string_view{" a", 2}; // space and 'a'
 * result = strip(d_str, d_to_strip);
 * // result is "b" ('a' or ' ' removed from the ends)
 * @endcode
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" aba ", 5};
 * auto d_to_strip = cudf::string_view{}; // empty string
 * auto result = strip(d_str, d_to_strip, side_type::LEFT);
 * // result is "aba "
 * d_to_strip = cudf::string_view{"a ", 2}; // 'a' and space
 * result = strip(d_str, d_to_strip, side_type::LEFT);
 * // result is "ba " ('a' or ' ' removed from the beginning)
 * @endcode
 *
 * @code{.cpp}
 * auto d_str = cudf::string_view{" aba ", 5};
 * auto d_to_strip = cudf::string_view{}; // empty string
 * auto result = strip(d_str, d_to_strip, side_type::RIGHT);
 * // result is " aba"
 * d_to_strip = cudf::string_view{" a", 2}; // space and 'a'
 * result = rstrip(d_str, d_to_strip, side_type::RIGHT);
 * // result is " ab" ('a' or ' ' removed from the end)
 * @endcode
 *
 * @param d_str String to strip characters from
 * @param d_to_strip Characters to remove
 * @param stype From where to strip the characters;
 *              Default `BOTH` indicates stripping characters from the
 *              beginning and the end of the input string `d_str`
 * @return New string with characters removed
 */
__device__ udf_string strip(cudf::string_view const d_str,
                            cudf::string_view const d_to_strip,
                            side_type stype = side_type::BOTH)
{ return udf_string{cudf::strings::detail::strip(d_str, d_to_strip, stype)}; }

}  // namespace udf
}  // namespace strings
}  // namespace cudf
