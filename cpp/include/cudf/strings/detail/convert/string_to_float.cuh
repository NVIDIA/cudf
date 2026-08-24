/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/strings/detail/convert/is_float.cuh>
#include <cudf/strings/string_view.cuh>

#include <cuda/std/cmath>
#include <cuda/std/limits>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief This function converts the given string into a
 * floating point double value.
 *
 * This will also map strings containing "NaN", "Inf", etc.
 * to the appropriate float values.
 *
 * This function will also handle scientific notation format.
 */
__device__ inline double stod(string_view const& d_str)
{
  char const* in_ptr = d_str.data();
  char const* end    = in_ptr + d_str.size_bytes();
  if (end == in_ptr) return 0.0;
  double sign{1.0};
  if (*in_ptr == '-' || *in_ptr == '+') {
    sign = (*in_ptr == '-' ? -1 : 1);
    ++in_ptr;
  }

  constexpr double infinity      = cuda::std::numeric_limits<double>::infinity();
  constexpr uint64_t max_holding = (cuda::std::numeric_limits<uint64_t>::max() - 9L) / 10L;

  // special strings: NaN, Inf
  if ((in_ptr < end) && *in_ptr > '9') {
    auto const inf_nan = string_view(in_ptr, static_cast<size_type>(end - in_ptr));
    if (is_nan_str(inf_nan)) return nan("");
    if (is_inf_str(inf_nan)) return sign * infinity;
  }

  // Parse and store the mantissa as much as we can,
  // until we are about to exceed the limit of uint64_t
  uint64_t digits = 0;
  int exp_off     = 0;
  bool decimal    = false;
  while (in_ptr < end) {
    char ch = *in_ptr;
    if (ch == '.') {
      decimal = true;
      ++in_ptr;
      continue;
    }
    if (ch < '0' || ch > '9') break;
    if (digits > max_holding)
      exp_off += (int)!decimal;
    else {
      digits = (digits * 10L) + static_cast<uint64_t>(ch - '0');
      if (digits > max_holding) {
        digits = digits / 10L;
        exp_off += (int)!decimal;
      } else
        exp_off -= (int)decimal;
    }
    ++in_ptr;
  }
  if (digits == 0) return sign * static_cast<double>(0);

  // check for exponent char
  int exp_ten  = 0;
  int exp_sign = 1;
  if (in_ptr < end) {
    char ch = *in_ptr++;
    if (ch == 'e' || ch == 'E') {
      if (in_ptr < end) {
        ch = *in_ptr;
        if (ch == '-' || ch == '+') {
          exp_sign = (ch == '-' ? -1 : 1);
          ++in_ptr;
        }
        while (in_ptr < end) {
          ch = *in_ptr++;
          if (ch < '0' || ch > '9') break;
          exp_ten = (exp_ten * 10) + (int)(ch - '0');
          // Prevent integer overflow in exp_ten. 100,000,000 is the largest
          // power of ten that can be multiplied by 10 without overflow.
          if (exp_ten >= 100'000'000) { break; }
        }
      }
    }
  }

  int const num_digits = static_cast<int>(log10(static_cast<double>(digits))) + 1;
  exp_ten *= exp_sign;
  exp_ten += exp_off;
  exp_ten += num_digits - 1;
  if (exp_ten > cuda::std::numeric_limits<double>::max_exponent10) {
    return sign > 0 ? infinity : -infinity;
  }

  double base = sign * static_cast<double>(digits);

  exp_ten += 1 - num_digits;
  // If 10^exp_ten would result in a subnormal value, the base and
  // exponent should be adjusted so that 10^exp_ten is a normal value
  auto const subnormal_shift = cuda::std::numeric_limits<double>::min_exponent10 - exp_ten;
  if (subnormal_shift > 0) {
    // Handle subnormal values. Ensure that both base and exponent are
    // normal values before computing their product.
    base = base / exp10(static_cast<double>(num_digits - 1 + subnormal_shift));
    exp_ten += num_digits - 1;  // adjust exponent
    auto const exponent = exp10(static_cast<double>(exp_ten + subnormal_shift));
    return base * exponent;
  }

  double const exponent = exp10(static_cast<double>(cuda::std::abs(exp_ten)));
  return exp_ten < 0 ? base / exponent : base * exponent;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
