/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/detail/convert/int_to_string.cuh>

#include <cuda/std/functional>

namespace cudf::strings::detail {

/**
 * @brief Returns the number of digits in the given fixed point number.
 *
 * @param value The value of the fixed point number
 * @param scale The scale of the fixed point number
 * @return int32_t The number of digits required to represent the fixed point number
 */
__device__ inline int32_t fixed_point_string_size(__int128_t const& value, int32_t scale)
{
  if (scale >= 0) return count_digits(value) + scale;

  auto const abs_value = numeric::detail::abs(value);
  auto const exp_ten   = numeric::detail::exp10<__int128_t>(-scale);
  auto const fraction  = count_digits(abs_value % exp_ten);
  auto const num_zeros = cuda::std::max(0, static_cast<int>(-scale - fraction));
  return static_cast<int32_t>(value < 0) +    // sign if negative
         count_digits(abs_value / exp_ten) +  // integer
         1 +                                  // decimal point
         num_zeros +                          // zeros padding
         fraction;                            // size of fraction
}

/**
 * @brief Converts the given fixed point number to a string.
 *
 * Caller is responsible for ensuring that the output buffer is large enough. The required output
 * buffer size can be obtained by calling `fixed_point_string_size`.
 *
 * @param value The value of the fixed point number
 * @param scale The scale of the fixed point number
 * @param out_ptr The pointer to the output string
 */
__device__ inline void fixed_point_to_string(__int128_t const& value, int32_t scale, char* out_ptr)
{
  if (scale >= 0) {
    out_ptr += integer_to_string(value, out_ptr);
    thrust::generate_n(thrust::seq, out_ptr, scale, []() { return '0'; });  // add zeros
    return;
  }

  // scale < 0
  // write format:   [-]integer.fraction
  // where integer  = abs(value) / (10^abs(scale))
  //       fraction = abs(value) % (10^abs(scale))
  if (value < 0) *out_ptr++ = '-';  // add sign
  auto const abs_value = numeric::detail::abs(value);
  auto const exp_ten   = numeric::detail::exp10<__int128_t>(-scale);
  auto const num_zeros =
    cuda::std::max(0, static_cast<int>(-scale - count_digits(abs_value % exp_ten)));

  out_ptr += integer_to_string(abs_value / exp_ten, out_ptr);  // add the integer part
  *out_ptr++ = '.';                                            // add decimal point

  thrust::generate_n(thrust::seq, out_ptr, num_zeros, []() { return '0'; });  // add zeros
  out_ptr += num_zeros;

  integer_to_string(abs_value % exp_ten, out_ptr);  // add the fraction part
}

}  // namespace cudf::strings::detail
