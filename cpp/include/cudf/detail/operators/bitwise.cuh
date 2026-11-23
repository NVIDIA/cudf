/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Counts set bits in the unsigned representation of a value.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <cuda::std::integral T>
__device__ int32_t bit_count(T a)
{
  if constexpr (cuda::std::is_same_v<T, bool>) {
    return static_cast<int32_t>(a);
  } else {
    return cuda::std::popcount(static_cast<cuda::std::make_unsigned_t<T>>(a));
  }
}

/**
 * @brief Computes bitwise AND of two values.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <concepts::integral_not_bool A, concepts::integral_not_bool B>
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
template <concepts::integral_not_bool T>
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
template <concepts::integral_not_bool A, concepts::integral_not_bool B>
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
template <concepts::integral_not_bool A, concepts::integral_not_bool B>
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
template <concepts::integral_not_bool A, concepts::integral_not_bool B>
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
template <concepts::integral_not_bool A, concepts::integral_not_bool B>
__device__ auto bitwise_shift_right(A a, B b) -> decltype(a >> b)
{
  return a >> b;
}

/**
 * @brief Shifts the unsigned representation of a value right by a bit count.
 *
 * @tparam A Value type.
 * @tparam B Shift count type.
 * @param a Input value.
 * @param b Shift count.
 */
template <concepts::integral_not_bool A, typename B>
__device__ auto bitwise_shift_right_unsigned(A a, B b)
  -> decltype(static_cast<cuda::std::make_unsigned_t<A>>(a) >> b)
{
  return static_cast<cuda::std::make_unsigned_t<A>>(a) >> b;
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
