/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Computes bitwise AND of two values.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <integer A, integer B>
__device__ auto bitwise_and(A a, B b) -> decltype(a & b)
{
  return a & b;
}

/**
 * @brief Computes bitwise NOT of one value.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <integer T>
__device__ auto bitwise_invert(T a) -> decltype(~a)
{
  return ~a;
}

/**
 * @brief Computes bitwise OR of two values.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <integer A, integer B>
__device__ auto bitwise_or(A a, B b) -> decltype(a | b)
{
  return a | b;
}

/**
 * @brief Computes bitwise XOR of two values.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <integer A, integer B>
__device__ auto bitwise_xor(A a, B b) -> decltype(a ^ b)
{
  return a ^ b;
}

/**
 * @brief Shifts a value left by a bit count.
 *
 * @tparam A Value type.
 * @tparam B Shift count type.
 * @param a Input value.
 * @param b Shift count.
 */
template <integer A, integer B>
__device__ auto bitwise_shift_left(A a, B b) -> decltype(a << b)
{
  return a << b;
}

/**
 * @brief Shifts a value right by a bit count.
 *
 * @tparam A Value type.
 * @tparam B Shift count type.
 * @param a Input value.
 * @param b Shift count.
 */
template <integer A, integer B>
__device__ auto bitwise_shift_right(A a, B b) -> decltype(a >> b)
{
  return a >> b;
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
