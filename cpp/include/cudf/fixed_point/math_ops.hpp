/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
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
 * @brief Divides two decimal columns element-wise, keeping the scale of `lhs`.
 *
 * Each output element is `lhs[i] / rhs[i]`, rounded to the scale of `lhs` using `method`.
 * The output has the same type and scale as `lhs`. This is unlike `cudf::binary_operation`
 * with `binary_operator::DIV`, whose output scale is `lhs.scale() - rhs.scale()` and which
 * truncates instead of rounding.
 *
 * An output element is null if either input element is null, if `rhs[i]` is zero, or if the
 * rounded quotient does not fit in the output type. For `DECIMAL128`, it is also null if `lhs[i]`
 * rescaled by `10^(-rhs.scale())` does not fit in 128 bits.
 *
 * @throw cudf::logic_error if `lhs` or `rhs` is not a decimal type
 * @throw cudf::logic_error if `lhs` and `rhs` have different type ids
 * @throw cudf::logic_error if `lhs` and `rhs` have different sizes
 *
 * @param lhs The dividend column
 * @param rhs The divisor column
 * @param method The rounding method to use
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations and the returned column
 * @return Column containing the quotients, with the type of `lhs`
 */
std::unique_ptr<column> divide_decimal(
  column_view const& lhs,
  column_view const& rhs,
  rounding_method method    = rounding_method::HALF_UP,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/**
 * @brief Divides each element of a decimal column by a decimal scalar, keeping the scale of `lhs`.
 *
 * Each output element is `lhs[i] / rhs`, rounded to the scale of `lhs` using `method`.
 * The output has the same type and scale as `lhs`. `rhs` may have a different scale.
 *
 * Null outputs follow the same rules as the column-column overload. Every output element is null
 * if `rhs` is invalid.
 *
 * @throw cudf::logic_error if `lhs` or `rhs` is not a decimal type
 * @throw cudf::logic_error if `lhs` and `rhs` have different type ids
 *
 * @param lhs The dividend column
 * @param rhs The divisor scalar
 * @param method The rounding method to use
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations and the returned column
 * @return Column containing the quotients, with the type of `lhs`
 */
std::unique_ptr<column> divide_decimal(
  column_view const& lhs,
  scalar const& rhs,
  rounding_method method    = rounding_method::HALF_UP,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/**
 * @brief Divides a decimal scalar by each element of a decimal column, keeping the scale of `lhs`.
 *
 * Each output element is `lhs / rhs[i]`, rounded to the scale of `lhs` using `method`.
 * The output has the same type and scale as `lhs`. `rhs` may have a different scale.
 *
 * Null outputs follow the same rules as the column-column overload. Every output element is null
 * if `lhs` is invalid.
 *
 * @throw cudf::logic_error if `lhs` or `rhs` is not a decimal type
 * @throw cudf::logic_error if `lhs` and `rhs` have different type ids
 *
 * @param lhs The dividend scalar
 * @param rhs The divisor column
 * @param method The rounding method to use
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations and the returned column
 * @return Column containing the quotients, with the type of `lhs`
 */
std::unique_ptr<column> divide_decimal(
  scalar const& lhs,
  column_view const& rhs,
  rounding_method method    = rounding_method::HALF_UP,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
