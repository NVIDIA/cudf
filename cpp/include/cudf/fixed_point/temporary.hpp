/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

#ifndef __CUDACC_RTC__
#include <algorithm>
#include <string>
#endif

namespace CUDF_EXPORT numeric {
namespace detail {

#ifndef __CUDACC_RTC__

template <typename T>
auto to_string(T value) -> std::string
{
  if constexpr (cuda::std::is_same_v<T, __int128_t>) {
    auto s          = std::string{};
    auto const sign = value < 0;
    if (sign) {
      value += 1;  // avoid overflowing if value == _int128_t lowest
      value *= -1;
      if (value == cuda::std::numeric_limits<__int128_t>::max())
        return "-170141183460469231731687303715884105728";
      value += 1;  // can add back the one, no need to avoid overflow anymore
    }
    do {
      s.push_back("0123456789"[value % 10]);
      value /= 10;
    } while (value);
    if (sign) s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
  } else {
    return std::to_string(value);
  }
  return std::string{};  // won't ever hit here, need to suppress warning though
}

#endif

template <typename T>
CUDF_HOST_DEVICE constexpr auto abs(T value)
{
  return value >= 0 ? value : -value;
}

template <typename T>
CUDF_HOST_DEVICE inline auto min(T lhs, T rhs)
{
  return lhs < rhs ? lhs : rhs;
}

template <typename T>
CUDF_HOST_DEVICE inline auto max(T lhs, T rhs)
{
  return lhs > rhs ? lhs : rhs;
}

template <typename BaseType>
CUDF_HOST_DEVICE constexpr auto exp10(int32_t exponent)
{
  BaseType value = 1;
  while (exponent > 0)
    value *= 10, --exponent;
  return value;
}

}  // namespace detail
}  // namespace CUDF_EXPORT numeric
