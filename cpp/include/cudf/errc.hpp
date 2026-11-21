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
using error_int = cuda::std::int32_t;

/**
 * @brief Operation error codes. Values in range [CUDF_RESERVED_BEGIN, CUDF_RESERVED_END) are
 * reserved for libcudf. User-defined error codes must be in range [`USER_RESERVED_BEGIN`,
 * `USER_RESERVED_END`) and will be interpreted by the consumer.
 */
enum class [[nodiscard]] errc : error_int {
  CUDF_RESERVED_BEGIN = 0,
  SUCCESS             = 0,
  ARITHMETIC_OVERFLOW = 1,
  DIVISION_BY_ZERO    = 2,
  CUDF_RESERVED_END   = 0xFF,
  USER_RESERVED_BEGIN = 0x100,
  USER_RESERVED_END   = 0xFFFF
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
