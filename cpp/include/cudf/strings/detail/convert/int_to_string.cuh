/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Converts an integer into string
 *
 * @tparam IntegerType integer type to convert from
 * @param value integer value to convert
 * @param d_buffer character buffer to store the converted string
 */
template <typename IntegerType>
__device__ inline size_type integer_to_string(IntegerType value, char* d_buffer)
{
  if (value == 0) {
    *d_buffer = '0';
    return 1;
  }
  bool const is_negative = cuda::std::is_signed<IntegerType>() ? (value < 0) : false;

  constexpr IntegerType base = 10;
  // largest 64-bit integer is 20 digits; largest 128-bit integer is 39 digits
  constexpr int MAX_DIGITS = cuda::std::numeric_limits<IntegerType>::digits10 + 1;
  char digits[MAX_DIGITS];  // place-holder for digit chars
  int digits_idx = 0;
  while (value != 0) {
    assert(digits_idx < MAX_DIGITS);
    digits[digits_idx++] = '0' + cudf::detail::absolute_value(value % base);
    // next digit
    value = value / base;
  }
  size_type const bytes = digits_idx + static_cast<size_type>(is_negative);

  char* ptr = d_buffer;
  if (is_negative) *ptr++ = '-';
  // digits are backwards, reverse the string into the output
  while (digits_idx-- > 0)
    *ptr++ = digits[digits_idx];
  return bytes;
}

/**
 * @brief Counts number of digits in a integer value including '-' sign
 *
 * @tparam IntegerType integer type of input value
 * @param value input value to count the digits of
 * @return size_type number of digits in input value
 */
template <typename IntegerType>
__device__ constexpr size_type count_digits(IntegerType value)
{
  if (value == 0) return 1;
  bool const is_negative = cuda::std::is_signed<IntegerType>() ? (value < 0) : false;
  // abs(std::numeric_limits<IntegerType>::min()) is negative;
  // for all integer types, the max() and min() values have the same number of digits
  value = (value == cuda::std::numeric_limits<IntegerType>::min())
            ? cuda::std::numeric_limits<IntegerType>::max()
            : cudf::detail::absolute_value(value);

  auto const digits = [value] {
    // largest 8-byte  unsigned value is 18446744073709551615 (20 digits)
    // largest 16-byte unsigned value is 340282366920938463463374607431768211455 (39 digits)
    auto constexpr max_digits = cuda::std::numeric_limits<IntegerType>::digits10 + 1;

    size_type digits = 1;
    __int128_t pow10 = 10;
    for (; digits < max_digits; ++digits, pow10 *= 10)
      if (value < pow10) break;
    return digits;
  }();

  return digits + static_cast<size_type>(is_negative);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
