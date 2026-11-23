/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/cstdint>

/**
 * @file
 * @brief Error codes for libcudf operations.
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief Integer representation of error codes, compatible with external consumers such as Numba.
 */
using error_int = int32_t;

/**
 * @brief Operation error codes. Zero means success. User-defined errors must be in
 * [USER_RESERVED_BEGIN, USER_RESERVED_END). Other values are reserved for libcudf.
 */
enum class [[nodiscard]] errc : error_int {
  SUCCESS             = 0,
  USER_RESERVED_BEGIN = 1,
  USER_RESERVED       = USER_RESERVED_BEGIN,
  USER_RESERVED_END   = 0xFF00,
  CUDF_RESERVED_BEGIN = 0xFF00,
  // Library error codes are ordered from higher to lower severity.
  DIVISION_BY_ZERO    = 0xFF00,
  ARITHMETIC_OVERFLOW = 0xFF01,
  CUDF_RESERVED_END   = 0x10000,
  RESERVED_BEGIN      = 0x10000
};

/**
 * @brief Convert an `errc` error code to a human-readable string.
 * @param error The error code to convert
 * @return A C-string representing the error code
 */
[[nodiscard]] constexpr char const* to_string(errc error)
{
  switch (error) {
    case errc::SUCCESS: return "SUCCESS";
    case errc::ARITHMETIC_OVERFLOW: return "ARITHMETIC_OVERFLOW";
    case errc::DIVISION_BY_ZERO: return "DIVISION_BY_ZERO";
    default: return "Unknown Error";
  }
}

/** @} */

}  // namespace CUDF_EXPORT cudf
