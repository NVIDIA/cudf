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
 * @brief An enumeration of error codes that can occur during operations.
 *
 * The last 32 codes, `USER_ERROR_0` to `USER_ERROR_31`, are reserved for user-defined functions.
 * libcudf never returns them and gives them no meaning, so a user-defined function can fail a row
 * with any of them and leave its meaning to the caller.
 */
enum class [[nodiscard]] errc : cuda::std::int8_t {
  SUCCESS             = 0,
  ARITHMETIC_OVERFLOW = 1,
  DIVISION_BY_ZERO    = 2,
  USER_ERROR_0        = 96,
  USER_ERROR_1        = 97,
  USER_ERROR_2        = 98,
  USER_ERROR_3        = 99,
  USER_ERROR_4        = 100,
  USER_ERROR_5        = 101,
  USER_ERROR_6        = 102,
  USER_ERROR_7        = 103,
  USER_ERROR_8        = 104,
  USER_ERROR_9        = 105,
  USER_ERROR_10       = 106,
  USER_ERROR_11       = 107,
  USER_ERROR_12       = 108,
  USER_ERROR_13       = 109,
  USER_ERROR_14       = 110,
  USER_ERROR_15       = 111,
  USER_ERROR_16       = 112,
  USER_ERROR_17       = 113,
  USER_ERROR_18       = 114,
  USER_ERROR_19       = 115,
  USER_ERROR_20       = 116,
  USER_ERROR_21       = 117,
  USER_ERROR_22       = 118,
  USER_ERROR_23       = 119,
  USER_ERROR_24       = 120,
  USER_ERROR_25       = 121,
  USER_ERROR_26       = 122,
  USER_ERROR_27       = 123,
  USER_ERROR_28       = 124,
  USER_ERROR_29       = 125,
  USER_ERROR_30       = 126,
  USER_ERROR_31       = 127,
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
    case errc::USER_ERROR_0: return "USER_ERROR_0";
    case errc::USER_ERROR_1: return "USER_ERROR_1";
    case errc::USER_ERROR_2: return "USER_ERROR_2";
    case errc::USER_ERROR_3: return "USER_ERROR_3";
    case errc::USER_ERROR_4: return "USER_ERROR_4";
    case errc::USER_ERROR_5: return "USER_ERROR_5";
    case errc::USER_ERROR_6: return "USER_ERROR_6";
    case errc::USER_ERROR_7: return "USER_ERROR_7";
    case errc::USER_ERROR_8: return "USER_ERROR_8";
    case errc::USER_ERROR_9: return "USER_ERROR_9";
    case errc::USER_ERROR_10: return "USER_ERROR_10";
    case errc::USER_ERROR_11: return "USER_ERROR_11";
    case errc::USER_ERROR_12: return "USER_ERROR_12";
    case errc::USER_ERROR_13: return "USER_ERROR_13";
    case errc::USER_ERROR_14: return "USER_ERROR_14";
    case errc::USER_ERROR_15: return "USER_ERROR_15";
    case errc::USER_ERROR_16: return "USER_ERROR_16";
    case errc::USER_ERROR_17: return "USER_ERROR_17";
    case errc::USER_ERROR_18: return "USER_ERROR_18";
    case errc::USER_ERROR_19: return "USER_ERROR_19";
    case errc::USER_ERROR_20: return "USER_ERROR_20";
    case errc::USER_ERROR_21: return "USER_ERROR_21";
    case errc::USER_ERROR_22: return "USER_ERROR_22";
    case errc::USER_ERROR_23: return "USER_ERROR_23";
    case errc::USER_ERROR_24: return "USER_ERROR_24";
    case errc::USER_ERROR_25: return "USER_ERROR_25";
    case errc::USER_ERROR_26: return "USER_ERROR_26";
    case errc::USER_ERROR_27: return "USER_ERROR_27";
    case errc::USER_ERROR_28: return "USER_ERROR_28";
    case errc::USER_ERROR_29: return "USER_ERROR_29";
    case errc::USER_ERROR_30: return "USER_ERROR_30";
    case errc::USER_ERROR_31: return "USER_ERROR_31";
    default: return "UNKNOWN_ERROR";
  }
}

/** @} */

}  // namespace CUDF_EXPORT cudf
