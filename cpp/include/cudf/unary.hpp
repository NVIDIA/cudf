/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/detail/floating_conversion.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <memory>

/**
 * @file
 * @brief Column APIs for unary ops
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup transformation_unaryops
 * @{
 */

/**
 * @brief Types of unary operations that can be performed on data.
 */
enum class unary_operator : int32_t {
  SIN,         ///< Trigonometric sine
  COS,         ///< Trigonometric cosine
  TAN,         ///< Trigonometric tangent
  ARCSIN,      ///< Trigonometric sine inverse
  ARCCOS,      ///< Trigonometric cosine inverse
  ARCTAN,      ///< Trigonometric tangent inverse
  SINH,        ///< Hyperbolic sine
  COSH,        ///< Hyperbolic cosine
  TANH,        ///< Hyperbolic tangent
  ARCSINH,     ///< Hyperbolic sine inverse
  ARCCOSH,     ///< Hyperbolic cosine inverse
  ARCTANH,     ///< Hyperbolic tangent inverse
  EXP,         ///< Exponential (base e, Euler number)
  LOG,         ///< Natural Logarithm (base e)
  SQRT,        ///< Square-root (x^0.5)
  CBRT,        ///< Cube-root (x^(1.0/3))
  CEIL,        ///< Smallest integer value not less than arg
  FLOOR,       ///< largest integer value not greater than arg
  ABS,         ///< Absolute value
  RINT,        ///< Rounds the floating-point argument arg to an integer value
  BIT_COUNT,   ///< Count the number of bits set to 1 of an integer value
  BIT_INVERT,  ///< Bitwise Not (~)
  NOT,         ///< Logical Not (!)
  NEGATE,      ///< Unary negation (-), only for signed numeric and duration types.
};

/**
 * @brief Performs unary op on all values in column
 *
 * Note: For `decimal32` and `decimal64`, only `ABS`, `CEIL` and `FLOOR` are supported.
 *
 * @param input A `column_view` as input
 * @param op operation to perform
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns Column of same size as `input` containing result of the operation
 */
std::unique_ptr<cudf::column> unary_operation(
  cudf::column_view const& input,
  cudf::unary_operator op,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a column of `type_id::BOOL8` elements where for every element in `input` `true`
 * indicates the value is null and `false` indicates the value is valid.
 *
 * @param input A `column_view` as input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A non-nullable column of `type_id::BOOL8` elements with `true`
 * representing `null` values.
 */
std::unique_ptr<cudf::column> is_null(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a column of `type_id::BOOL8` elements where for every element in `input` `true`
 * indicates the value is valid and `false` indicates the value is null.
 *
 * @param input A `column_view` as input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A non-nullable column of `type_id::BOOL8` elements with `false`
 * representing `null` values.
 */
std::unique_ptr<cudf::column> is_valid(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Casts data from dtype specified in input to dtype specified in output.
 *
 * Supports only fixed-width types.
 *
 * @param input Input column
 * @param out_type Desired datatype of output column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns Column of same size as `input` containing result of the cast operation
 * @throw cudf::logic_error if `out_type` is not a fixed-width type
 */
std::unique_ptr<column> cast(
  column_view const& input,
  data_type out_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Check if a cast between two datatypes is supported.
 *
 * @param from source type
 * @param to   target type
 *
 * @returns true if the cast is supported.
 */
bool is_supported_cast(data_type from, data_type to) noexcept;

/**
 * @brief Creates a column of `type_id::BOOL8` elements indicating the presence of `NaN` values
 * in a column of floating point values.
 * The output element at row `i` is `true` if the element in `input` at row i is `NAN`, else `false`
 *
 * @throws cudf::logic_error if `input` is a non-floating point type
 *
 * @param input A column of floating-point elements
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A non-nullable column of `type_id::BOOL8` elements with `true` representing `NAN` values
 */
std::unique_ptr<column> is_nan(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Creates a column of `type_id::BOOL8` elements indicating the absence of `NaN` values
 * in a column of floating point values.
 * The output element at row `i` is `false` if the element in `input` at row i is `NAN`, else `true`
 *
 * @throws cudf::logic_error if `input` is a non-floating point type
 *
 * @param input A column of floating-point elements
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A non-nullable column of `type_id::BOOL8` elements with `false` representing `NAN`
 * values
 */
std::unique_ptr<column> is_not_nan(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
