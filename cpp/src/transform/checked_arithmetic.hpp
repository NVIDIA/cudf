/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/binaryop.hpp>
#include <cudf/unary.hpp>

namespace cudf::detail::checked_arithmetic {

[[nodiscard]] constexpr bool is_checked(binary_operator op)
{
  return op == binary_operator::ADD_OVERFLOW || op == binary_operator::SUB_OVERFLOW ||
         op == binary_operator::MUL_OVERFLOW || op == binary_operator::DIV_OVERFLOW ||
         op == binary_operator::MOD_OVERFLOW;
}

[[nodiscard]] constexpr bool is_checked(unary_operator op)
{
  return op == unary_operator::NEG_OVERFLOW || op == unary_operator::ABS_OVERFLOW;
}

std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         error_policy policy,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         error_policy policy,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         error_policy policy,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

std::unique_ptr<column> unary_operation(column_view const& input,
                                        unary_operator op,
                                        error_policy policy,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr);

}  // namespace cudf::detail::checked_arithmetic
