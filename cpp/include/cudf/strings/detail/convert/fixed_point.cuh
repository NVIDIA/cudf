/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/fixed_point/temporary.hpp>

#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Return the integer component of a decimal string.
 *
 * This reads everything up to the exponent 'e' notation.
 * The return includes the integer digits and any exponent offset.
 *
 * @tparam UnsignedDecimalType The unsigned version of the desired decimal type.
 *                             Use the `cuda::std::make_unsigned_t` to create the
 *                             unsigned type from the storage type.
 *
 * @param[in,out] iter Start of characters to parse
 * @param[in] iter_end End of characters to parse
 * @param[in] decimal_pt_char The character used as the decimal point
 * @return Integer component and exponent offset.
 */
template <typename UnsignedDecimalType>
__device__ inline cuda::std::pair<UnsignedDecimalType, int32_t> parse_integer(
  char const*& iter, char const* iter_end, char const decimal_pt_char = '.')
{
  // highest value where another decimal digit cannot be appended without an overflow;
  // this preserves the most digits when scaling the final result for this type
  constexpr UnsignedDecimalType decimal_max =
    (cuda::std::numeric_limits<UnsignedDecimalType>::max() - 9L) / 10L;

  __uint128_t value  = 0;  // for checking overflow
  int32_t exp_offset = 0;
  bool decimal_found = false;

  while (iter < iter_end) {
    auto const ch = *iter++;
    if (ch == decimal_pt_char && !decimal_found) {
      decimal_found = true;
      continue;
    }
    if (ch < '0' || ch > '9') {
      --iter;
      break;
    }
    if (value > decimal_max) {
      exp_offset += static_cast<int32_t>(!decimal_found);
    } else {
      value = (value * 10) + static_cast<UnsignedDecimalType>(ch - '0');
      exp_offset -= static_cast<int32_t>(decimal_found);
    }
  }
  return {value, exp_offset};
}

/**
 * @brief Return the exponent of a decimal string.
 *
 * This should only be called after the exponent 'e' notation was detected.
 * The return is the exponent (base-10) integer and can only be
 * invalid if `check_only == true` and invalid characters are found or the
 * exponent overflows an int32.
 *
 * @tparam check_only Set to true to verify the characters are valid and the
 *         exponent value in the decimal string does not overflow int32
 * @param[in,out] iter Start of characters to parse
 *                     (points to the character after the 'E' or 'e')
 * @param[in] iter_end End of characters to parse
 * @return Integer value of the exponent
 */
template <bool check_only = false>
__device__ cuda::std::optional<int32_t> parse_exponent(char const* iter, char const* iter_end)
{
  constexpr uint32_t exponent_max =
    static_cast<uint32_t>(cuda::std::numeric_limits<int32_t>::max());

  // get optional exponent sign
  int32_t const exp_sign = [&iter] {
    auto const ch = *iter;
    if (ch != '-' && ch != '+') { return 1; }
    ++iter;
    return (ch == '-' ? -1 : 1);
  }();

  // parse exponent integer
  int32_t exp_ten = 0;
  while (iter < iter_end) {
    auto const ch = *iter++;
    if (ch < '0' || ch > '9') {
      if (check_only) { return cuda::std::nullopt; }
      break;
    }

    uint32_t exp_check = static_cast<uint32_t>(exp_ten * 10) + static_cast<uint32_t>(ch - '0');
    if (check_only && (exp_check > exponent_max)) { return cuda::std::nullopt; }  // check overflow
    exp_ten = static_cast<int32_t>(exp_check);
  }

  return exp_ten * exp_sign;
}

/**
 * @brief Converts the string in the range [iter, iter_end) into a decimal.
 *
 * @tparam DecimalType The decimal type to be returned
 * @param iter The beginning of the string
 * @param iter_end The end of the characters to parse
 * @param scale The scale to be applied
 * @return
 */
template <typename DecimalType>
__device__ DecimalType parse_decimal(char const* iter, char const* iter_end, int32_t scale)
{
  auto const sign = [&] {
    if (iter_end <= iter) { return 0; }
    if (*iter == '-') { return -1; }
    if (*iter == '+') { return 1; }
    return 0;
  }();

  // if string begins with a sign, continue with next character
  if (sign != 0) ++iter;

  using UnsignedDecimalType = cuda::std::make_unsigned_t<DecimalType>;
  auto [value, exp_offset]  = parse_integer<UnsignedDecimalType>(iter, iter_end);
  if (value == 0) { return DecimalType{0}; }

  // check for exponent
  int32_t exp_ten = 0;
  if ((iter < iter_end) && (*iter == 'e' || *iter == 'E')) {
    ++iter;
    if (iter < iter_end) { exp_ten = parse_exponent<false>(iter, iter_end).value(); }
  }
  exp_ten += exp_offset;

  // shift the output value based on the exp_ten and the scale values
  auto const shift_adjust =
    abs(scale - exp_ten) > cuda::std::numeric_limits<UnsignedDecimalType>::digits10
      ? cuda::std::numeric_limits<UnsignedDecimalType>::max()
      : numeric::detail::exp10<UnsignedDecimalType>(abs(scale - exp_ten));
  value = exp_ten < scale ? value / shift_adjust : value * shift_adjust;

  return static_cast<DecimalType>(value) * (sign == 0 ? 1 : sign);
}
}  // namespace detail
}  // namespace strings
}  // namespace cudf
