/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Column APIs for round
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup transformation_unaryops
 * @{
 */

/**
 * @brief Different rounding methods for `cudf::round`
 *
 * Info on HALF_EVEN rounding: https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even
 * Info on HALF_UP   rounding: https://en.wikipedia.org/wiki/Rounding#Rounding_half_away_from_zero
 * Note: HALF_UP means up in MAGNITUDE: Away from zero! Because of how Java and python define it
 */
enum class rounding_method : int32_t { HALF_UP, HALF_EVEN };

/**
 * @brief Rounds all the values in a column to the specified number of decimal places.
 *
 * @deprecated Use cudf::round_decimal only for integer and fixed-point (decimal) types
 *
 * `cudf::round` currently supports HALF_UP and HALF_EVEN rounding for integer, floating point and
 * `decimal32` and `decimal64` numbers. For `decimal32` and `decimal64` numbers, negated
 * `numeric::scale` is equivalent to `decimal_places`.
 *
 * Example:
 * ```
 * using namespace cudf;
 *
 * column_view a; // contains { 1.729, 17.29, 172.9, 1729 };
 *
 * auto result1 = round(a);     // { 2,   17,   173,   1729 }
 * auto result2 = round(a, 1);  // { 1.7, 17.3, 172.9, 1729 }
 * auto result3 = round(a, -1); // { 0,   20,   170,   1730 }
 *
 * column_view b; // contains { 1.5, 2.5, 1.35, 1.45, 15, 25 };
 *
 * auto result4 = round(b,  0, rounding_method::HALF_EVEN); // { 2,   2,   1,   1,   15, 25};
 * auto result5 = round(b,  1, rounding_method::HALF_EVEN); // { 1.5, 2.5, 1.4, 1.4, 15, 25};
 * auto result6 = round(b, -1, rounding_method::HALF_EVEN); // { 0,   0,   0,   0,   20, 20};
 * ```
 *
 * @param input          Column of values to be rounded
 * @param decimal_places Number of decimal places to round to (default 0). If negative, this
 * specifies the number of positions to the left of the decimal point.
 * @param method         Rounding method
 * @param stream         CUDA stream used for device memory operations and kernel launches
 * @param mr             Device memory resource used to allocate the returned column's device memory
 *
 * @return Column with each of the values rounded
 */
[[deprecated]] std::unique_ptr<column> round(
  column_view const& input,
  int32_t decimal_places            = 0,
  rounding_method method            = rounding_method::HALF_UP,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Rounds all the values in a column to the specified number of decimal places.
 *
 * Supports HALF_UP and HALF_EVEN rounding for integer and fixed-point (decimal)
 * numbers. For decimal numbers, negated `numeric::scale` is equivalent to `decimal_places`.
 *
 * Example:
 * ```
 * using namespace cudf;
 *
 * column_view a; // contains { 1.729, 17.29, 172.9, 1729 };
 *
 * round_decimal(a);     // { 2,   17,   173,   1729 }
 * round_decimal(a, 1);  // { 1.7, 17.3, 172.9, 1729 }
 * round_decimal(a, -1); // { 0,   20,   170,   1730 }
 *
 * column_view b; // contains { 1.5, 2.5, 1.35, 1.45, 15, 25 };
 *
 * round_decimal(b,  0, rounding_method::HALF_EVEN); // { 2,   2,   1,   1,   15, 25};
 * round_decimal(b,  1, rounding_method::HALF_EVEN); // { 1.5, 2.5, 1.4, 1.4, 15, 25};
 * round_decimal(b, -1, rounding_method::HALF_EVEN); // { 0,   0,   0,   0,   20, 20};
 * ```
 *
 * @param input          Column of values to be rounded
 * @param decimal_places Number of decimal places to round to (default 0).
 *                       If negative, the number of positions to the left of the decimal point.
 * @param method         Rounding method
 * @param stream         CUDA stream used for device memory operations and kernel launches
 * @param mr             Device memory resource used to allocate the returned column's device memory
 *
 * @return Column with each of the values rounded
 */
std::unique_ptr<column> round_decimal(
  column_view const& input,
  int32_t decimal_places            = 0,
  rounding_method method            = rounding_method::HALF_UP,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
