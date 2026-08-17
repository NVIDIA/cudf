/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/binaryop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <optional>

namespace cudf {
// Forward declarations
class column_device_view;
class mutable_column_device_view;

namespace binops {
namespace compiled {

std::unique_ptr<column> string_null_min_max(scalar const& lhs,
                                            column_view const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

std::unique_ptr<column> string_null_min_max(column_view const& lhs,
                                            scalar const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

std::unique_ptr<column> string_null_min_max(column_view const& lhs,
                                            column_view const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @brief Performs a binary operation between a string scalar and a string
 * column.
 *
 * The output contains the result of op(lhs, rhs[i]) for all 0 <= i < rhs.size()
 * The scalar is the left operand and the column elements are the right operand.
 * This distinction is significant in case of non-commutative binary operations
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands
 *
 * @param lhs         The left operand string scalar
 * @param rhs         The right operand string column
 * @param op          The binary operator
 * @param output_type The desired data type of the output column
 * @param stream      CUDA stream used for device memory operations and kernel launches.
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> Output column
 */
std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @brief Performs a binary operation between a string column and a string
 * scalar.
 *
 * The output contains the result of op(lhs[i], rhs) for all 0 <= i < lhs.size()
 * The column elements are the left operand and the scalar is the right operand.
 * This distinction is significant in case of non-commutative binary operations
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands
 *
 * @param lhs         The left operand string column
 * @param rhs         The right operand string scalar
 * @param op          The binary operator
 * @param output_type The desired data type of the output column
 * @param stream      CUDA stream used for device memory operations and kernel launches.
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> Output column
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @brief Performs a binary operation between two string columns.
 *
 * @note The sizes of @p lhs and @p rhs should be the same
 *
 * The output contains the result of op(lhs[i], rhs[i]) for all 0 <= i < lhs.size()
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands
 *
 * @param lhs         The left operand string column
 * @param rhs         The right operand string column
 * @param op          The binary operator enum
 * @param output_type The desired data type of the output column
 * @param stream      CUDA stream used for device memory operations and kernel launches.
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> Output column
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

void binary_operation(mutable_column_view& out,
                      scalar const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      cuda::stream_ref stream);
void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      scalar const& rhs,
                      binary_operator op,
                      cuda::stream_ref stream);
void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      cuda::stream_ref stream);

// Defined in util.cpp
/**
 * @brief Get the common type among all input types.
 *
 * @param out type 1
 * @param lhs type 2
 * @param rhs type 3
 * @return common type among @p out, @p lhs, @p rhs.
 */
std::optional<data_type> get_common_type(data_type out, data_type lhs, data_type rhs);
/**
 * @brief Check if input binary operation is supported for the given input and output types.
 *
 * @param out output type of the binary operation
 * @param lhs first operand type of the binary operation
 * @param rhs second operand type of the binary operation
 * @param op binary operator enum.
 * @return true if given binary operator supports given input and output types.
 */
bool is_supported_operation(data_type out, data_type lhs, data_type rhs, binary_operator op);

// Defined in individual .cu files.
/**
 * @brief Deploys single type or double type dispatcher that runs binary operation on each element
 * of @p lhs and @p rhs columns.
 *
 * This template is instantiated for each binary operator.
 *
 * @tparam BinaryOperator Binary operator functor
 * @param out mutable view of output column
 * @param lhs view of left operand column
 * @param rhs view of right operand column
 * @param is_lhs_scalar true if @p lhs is a single element column representing a scalar
 * @param is_rhs_scalar true if @p rhs is a single element column representing a scalar
 * @param stream CUDA stream used for device memory operations
 */
template <class BinaryOperator>
void apply_binary_op(mutable_column_view& out,
                     column_view const& lhs,
                     column_view const& rhs,
                     bool is_lhs_scalar,
                     bool is_rhs_scalar,
                     cuda::stream_ref stream);
/**
 * @brief Deploys single type or double type dispatcher that runs equality operation on each element
 * of @p lhs and @p rhs columns.
 *
 * Comparison operators are EQUAL, NOT_EQUAL, NULL_EQUALS, NULL_NOT_EQUALS.
 * @p out type is boolean.
 *
 * This template is instantiated for each binary operator.
 *
 * @param out mutable view of output column
 * @param lhs view of left operand column
 * @param rhs view of right operand column
 * @param is_lhs_scalar true if @p lhs is a single element column representing a scalar
 * @param is_rhs_scalar true if @p rhs is a single element column representing a scalar
 * @param op comparison binary operator
 * @param stream CUDA stream used for device memory operations
 */
void dispatch_equality_op(mutable_column_view& out,
                          column_view const& lhs,
                          column_view const& rhs,
                          bool is_lhs_scalar,
                          bool is_rhs_scalar,
                          binary_operator op,
                          cuda::stream_ref stream);
}  // namespace compiled
}  // namespace binops
}  // namespace cudf
