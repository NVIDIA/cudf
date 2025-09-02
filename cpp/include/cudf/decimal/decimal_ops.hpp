/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup transformation_decimalops
 * @{
 * @file
 * @brief Column APIs for decimal operations with scale preservation
 */

/**
 * @brief Performs decimal division between two columns with scale preservation.
 *
 * The output contains the result of `divide_decimal(lhs[i], rhs[i])` for all `0 <= i < lhs.size()`
 * The scale of the output is preserved to match the scale of the left operand.
 *
 * @param lhs         The left operand column
 * @param rhs         The right operand column
 * @param rounding_mode The rounding mode to use
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return            Output column containing the result of the decimal division
 * @throw cudf::logic_error if @p lhs and @p rhs are different sizes
 * @throw cudf::logic_error if @p lhs and @p rhs are not decimal types
 */
std::unique_ptr<column> divide_decimal(
  column_view const& lhs,
  column_view const& rhs,
  numeric::decimal_rounding_mode rounding_mode = numeric::decimal_rounding_mode::HALF_UP,
  rmm::cuda_stream_view stream                 = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr            = cudf::get_current_device_resource_ref());

/**
 * @brief Performs decimal division between a column and a scalar with scale preservation.
 *
 * The output contains the result of `divide_decimal(lhs[i], rhs)` for all `0 <= i < lhs.size()`
 * The scale of the output is preserved to match the scale of the left operand.
 *
 * @param lhs         The left operand column
 * @param rhs         The right operand scalar
 * @param rounding_mode The rounding mode to use
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return            Output column containing the result of the decimal division
 * @throw cudf::logic_error if @p lhs is not a decimal type
 * @throw cudf::logic_error if @p rhs is not a decimal scalar
 */
std::unique_ptr<column> divide_decimal(
  column_view const& lhs,
  scalar const& rhs,
  numeric::decimal_rounding_mode rounding_mode = numeric::decimal_rounding_mode::HALF_UP,
  rmm::cuda_stream_view stream                 = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr            = cudf::get_current_device_resource_ref());

/**
 * @brief Performs decimal division between a scalar and a column with scale preservation.
 *
 * The output contains the result of `divide_decimal(lhs, rhs[i])` for all `0 <= i < rhs.size()`
 * The scale of the output is preserved to match the scale of the left operand.
 *
 * @param lhs         The left operand scalar
 * @param rhs         The right operand column
 * @param rounding_mode The rounding mode to use
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return            Output column containing the result of the decimal division
 * @throw cudf::logic_error if @p lhs is not a decimal scalar
 * @throw cudf::logic_error if @p rhs is not a decimal type
 */
std::unique_ptr<column> divide_decimal(
  scalar const& lhs,
  column_view const& rhs,
  numeric::decimal_rounding_mode rounding_mode = numeric::decimal_rounding_mode::HALF_UP,
  rmm::cuda_stream_view stream                 = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr            = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
