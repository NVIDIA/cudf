/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/traits.hpp>

#include <cuda/std/cmath>
#include <cuda/std/limits>

namespace cudf::hashing::detail {

/**
 * Normalization of floating point NaNs, passthrough for all other values.
 */
template <typename T>
T __device__ inline normalize_nans(T const& key)
{
  if constexpr (cudf::is_floating_point<T>()) {
    if (cuda::std::isnan(key)) { return cuda::std::numeric_limits<T>::quiet_NaN(); }
  }
  return key;
}

/**
 * Normalization of floating point NaNs and zeros, passthrough for all other values.
 */
template <typename T>
T __device__ inline normalize_nans_and_zeros(T const& key)
{
  if constexpr (cudf::is_floating_point<T>()) {
    if (key == T{0.0}) { return T{0.0}; }
  }
  return normalize_nans(key);
}

__device__ inline uint32_t rotate_bits_left(uint32_t x, uint32_t r)
{
  // This function is equivalent to (x << r) | (x >> (32 - r))
  return __funnelshift_l(x, x, r);
}

__device__ inline uint64_t rotate_bits_left(uint64_t x, uint32_t r)
{
  return (x << r) | (x >> (64 - r));
}

__device__ inline uint32_t rotate_bits_right(uint32_t x, uint32_t r)
{
  // This function is equivalent to (x >> r) | (x << (32 - r))
  return __funnelshift_r(x, x, r);
}

__device__ inline uint64_t rotate_bits_right(uint64_t x, uint32_t r)
{
  return (x >> r) | (x << (64 - r));
}

// Swap the endianness of a 32 bit value
__device__ inline uint32_t swap_endian(uint32_t x)
{
  // The selector 0x0123 reverses the byte order
  return __byte_perm(x, 0, 0x0123);
}

// Swap the endianness of a 64 bit value
// There is no CUDA intrinsic for permuting bytes in 64 bit integers
__device__ inline uint64_t swap_endian(uint64_t x)
{
  // Reverse the endianness of each 32 bit section
  uint32_t low_bits  = swap_endian(static_cast<uint32_t>(x));
  uint32_t high_bits = swap_endian(static_cast<uint32_t>(x >> 32));
  // Reassemble a 64 bit result, swapping the low bits and high bits
  return (static_cast<uint64_t>(low_bits) << 32) | (static_cast<uint64_t>(high_bits));
};

/**
 * SPDX-SnippetBegin
 * SPDX-SnippetCopyrightText: Copyright (c) 2015 Barry Clark
 * SPDX-License-Identifier: MIT
 *
 * Modified GPU implementation of
 * https://johnnylee-sde.github.io/Fast-unsigned-integer-to-hex-string/
 * Copyright (c) 2015 Barry Clark
 * Licensed under the MIT license.
 * See file LICENSE for detail or copy at https://opensource.org/license/MIT
 */
__device__ inline void uint32ToLowercaseHexString(uint32_t num, char* destination)
{
  // Transform 0xABCD'1234 => 0x0000'ABCD'0000'1234 => 0x0B0A'0D0C'0201'0403
  uint64_t x = num;
  x          = ((x & 0xFFFF'0000u) << 16) | ((x & 0xFFFF));
  x          = ((x & 0x000F'0000'000Fu) << 8) | ((x & 0x00F0'0000'00F0u) >> 4) |
      ((x & 0x0F00'0000'0F00u) << 16) | ((x & 0xF000'0000'F000) << 4);

  // Calculate a mask of ascii value offsets for bytes that contain alphabetical hex digits
  uint64_t offsets = (((x + 0x0606'0606'0606'0606) >> 4) & 0x0101'0101'0101'0101) * 0x27;

  x |= 0x3030'3030'3030'3030;
  x += offsets;
  std::memcpy(destination, reinterpret_cast<uint8_t*>(&x), 8);
}
// SPDX-SnippetEnd

}  // namespace cudf::hashing::detail
