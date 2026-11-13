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
