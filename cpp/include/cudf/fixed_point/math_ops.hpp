/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

/**
 * @file
 * @brief Column APIs for fixed-point math operations
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup transformation_fixed_point
 * @{
 */

/**
 * @brief Width of the intermediate value that `divide_decimal` rescales the dividend into.
 */
enum class decimal_intermediate : int32_t {
  STANDARD,  ///< The widest of the input and output types. Rows that overflow it are null.
  WIDE       ///< 256 bits. Rows are null only for a zero divisor or a quotient that doesn't fit.
};

/**
 * @brief Divides two decimal columns element-wise into a column of type `output_type`.
 *
 * Each output element is `lhs[i] / rhs[i]`, rounded to the scale of `output_type` using `method`.
 * `lhs`, `rhs` and `output_type` may each be any decimal type, with any scale. Passing
 * `lhs.type()` keeps the type and scale of `lhs`, and a scale of `min(lhs.scale(), rhs.scale())`
 * keeps the larger number of fractional digits of the two operands. This is unlike
 * `cudf::binary_operation` with `binary_operator::DIV`, which truncates instead of rounding.
 *
 * The precision of the output is the largest number of decimal digits its type can hold: 9 for
 * `DECIMAL32`, 18 for `DECIMAL64` and 38 for `DECIMAL128`.
 *
 * An output element is null if:
 * - either input element is null,
 * - `rhs[i]` is zero,
 * - the rounded quotient has more digits than the precision of `output_type`, or
 * - `intermediate` is `decimal_intermediate::STANDARD` and `lhs[i]`, rescaled to give a quotient
 *   at the scale of `output_type`, does not fit in the widest of the three types.
 *
 * With `decimal_intermediate::WIDE`, the rescaled dividend is held in 256 bits, which is always
 * enough for any quotient that fits in `output_type`. Rows that would need it are slower than
 * other rows.
 *
 * @throw cudf::logic_error if `lhs`, `rhs` or `output_type` is not a decimal type
 * @throw cudf::logic_error if `lhs` and `rhs` have different sizes
 *
 * @param lhs The dividend column
 * @param rhs The divisor column
 * @param output_type The decimal type and scale of the output
 * @param method The rounding method to use
 * @param intermediate The width of the intermediate value used to rescale the dividend
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations and the returned column
 * @return Column of type `output_type` containing the quotients
 */
std::unique_ptr<column> divide_decimal(
  column_view const& lhs,
  column_view const& rhs,
  data_type output_type,
  rounding_method method            = rounding_method::HALF_UP,
  decimal_intermediate intermediate = decimal_intermediate::STANDARD,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  cudf::memory_resources mr         = cudf::get_current_device_resource_ref());

/**
 * @brief Divides each element of a decimal column by a decimal scalar into a column of type
 * `output_type`.
 *
 * Each output element is `lhs[i] / rhs`, rounded to the scale of `output_type` using `method`.
 *
 * Types, precision and null outputs follow the same rules as the column-column overload. Every
 * output element is null if `rhs` is invalid.
 *
 * @throw cudf::logic_error if `lhs`, `rhs` or `output_type` is not a decimal type
 *
 * @param lhs The dividend column
 * @param rhs The divisor scalar
 * @param output_type The decimal type and scale of the output
 * @param method The rounding method to use
 * @param intermediate The width of the intermediate value used to rescale the dividend
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations and the returned column
 * @return Column of type `output_type` containing the quotients
 */
std::unique_ptr<column> divide_decimal(
  column_view const& lhs,
  scalar const& rhs,
  data_type output_type,
  rounding_method method            = rounding_method::HALF_UP,
  decimal_intermediate intermediate = decimal_intermediate::STANDARD,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  cudf::memory_resources mr         = cudf::get_current_device_resource_ref());

/**
 * @brief Divides a decimal scalar by each element of a decimal column into a column of type
 * `output_type`.
 *
 * Each output element is `lhs / rhs[i]`, rounded to the scale of `output_type` using `method`.
 *
 * Types, precision and null outputs follow the same rules as the column-column overload. Every
 * output element is null if `lhs` is invalid.
 *
 * @throw cudf::logic_error if `lhs`, `rhs` or `output_type` is not a decimal type
 *
 * @param lhs The dividend scalar
 * @param rhs The divisor column
 * @param output_type The decimal type and scale of the output
 * @param method The rounding method to use
 * @param intermediate The width of the intermediate value used to rescale the dividend
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations and the returned column
 * @return Column of type `output_type` containing the quotients
 */
std::unique_ptr<column> divide_decimal(
  scalar const& lhs,
  column_view const& rhs,
  data_type output_type,
  rounding_method method            = rounding_method::HALF_UP,
  decimal_intermediate intermediate = decimal_intermediate::STANDARD,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  cudf::memory_resources mr         = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
