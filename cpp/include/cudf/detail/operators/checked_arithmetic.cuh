/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/detail/operators/math.cuh>
#include <cudf/errc.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/numeric>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Adds operands with overflow detection.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 * @return `errc::ARITHMETIC_OVERFLOW` on overflow, else the result.
 */
template <integer A, integer B>
__device__ cuda::std::expected<A, errc> add_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  A r;
  if (cuda::add_overflow(r, a, b)) { return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW}; }
  return r;
}

template <floating_point A, floating_point B>
__device__ cuda::std::expected<A, errc> add_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  return a + b;
}

template <typename A, typename B>
__device__ cuda::std::expected<numeric::decimal<A>, errc> add_overflow(numeric::decimal<A> a,
                                                                       numeric::decimal<B> b)
  requires(cuda::std::same_as<A, B>)
{
  auto scale = cuda::std::min(a.scale(), b.scale());

  if (numeric::addition_overflow<A>(a.rescaled(scale).value(), b.rescaled(scale).value())) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }

  return numeric::decimal<A>{numeric::scaled_integer<A>{
    a.rescaled(scale).value() + b.rescaled(scale).value(), numeric::scale_type{scale}}};
}

/**
 * @brief Subtracts operands with overflow detection.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 * @return `errc::ARITHMETIC_OVERFLOW` on overflow, else the result.
 */
template <integer A, integer B>
__device__ cuda::std::expected<A, errc> sub_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  A r;
  if (cuda::sub_overflow(r, a, b)) { return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW}; }
  return r;
}

template <floating_point A, floating_point B>
__device__ cuda::std::expected<A, errc> sub_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  return a - b;
}

template <typename A, typename B>
__device__ cuda::std::expected<numeric::decimal<A>, errc> sub_overflow(numeric::decimal<A> a,
                                                                       numeric::decimal<B> b)
  requires(cuda::std::same_as<A, B>)
{
  auto scale = cuda::std::min(a.scale(), b.scale());

  if (numeric::subtraction_overflow<A>(a.rescaled(scale).value(), b.rescaled(scale).value())) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }

  return numeric::decimal<A>{numeric::scaled_integer<A>{
    a.rescaled(scale).value() - b.rescaled(scale).value(), numeric::scale_type{scale}}};
}

/**
 * @brief Multiplies operands with overflow detection.
 *
 * @tparam A Left operand type.
 * @tparam B Right operand type.
 * @param a Left operand.
 * @param b Right operand.
 * @return `errc::ARITHMETIC_OVERFLOW` on overflow, else the result.
 */
template <integer A, integer B>
__device__ cuda::std::expected<A, errc> mul_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  A r;
  if (cuda::mul_overflow(r, a, b)) { return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW}; }
  return r;
}

template <floating_point A, floating_point B>
__device__ cuda::std::expected<A, errc> mul_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  return a * b;
}

template <typename A, typename B>
__device__ cuda::std::expected<numeric::decimal<A>, errc> mul_overflow(numeric::decimal<A> a,
                                                                       numeric::decimal<B> b)
  requires(cuda::std::same_as<A, B>)
{
  if (numeric::multiplication_overflow<A>(a.value(), b.value())) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }

  return numeric::decimal<A>{
    numeric::scaled_integer<A>{a.value() * b.value(), numeric::scale_type{a.scale() + b.scale()}}};
}

/**
 * @brief Divides operands with overflow detection.
 *
 * @tparam A Dividend type.
 * @tparam B Divisor type.
 * @param a Dividend.
 * @param b Divisor.
 * @return `errc::DIVISION_BY_ZERO` on zero divisor, `errc::ARITHMETIC_OVERFLOW` on overflow, else
 * `errc::SUCCESS`.
 */
template <integer A, integer B>
__device__ cuda::std::expected<A, errc> div_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  if (b == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }
  A r;
  if (cuda::div_overflow(r, a, b)) { return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW}; }
  return r;
}

template <floating_point A, floating_point B>
__device__ cuda::std::expected<A, errc> div_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  return a / b;
}

template <typename A, typename B>
__device__ cuda::std::expected<numeric::decimal<A>, errc> div_overflow(numeric::decimal<A> a,
                                                                       numeric::decimal<B> b)
  requires(cuda::std::same_as<A, B>)
{
  if (b.value() == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }

  if (numeric::division_overflow<A>(a.value(), b.value())) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }

  return numeric::decimal<A>{
    numeric::scaled_integer<A>{a.value() / b.value(), numeric::scale_type{a.scale() - b.scale()}}};
}

/**
 * @brief Computes modulus with overflow detection.
 *
 * @tparam A Dividend type.
 * @tparam B Divisor type.
 * @param a Dividend.
 * @param b Divisor.
 * @return `errc::DIVISION_BY_ZERO` on zero divisor, else the result.
 */
template <signed_integer A, signed_integer B>
__device__ cuda::std::expected<A, errc> mod_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  if (b == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }

  // avoid signed overflow UB / trap for minimum value divided by -1.
  if (a == cuda::std::numeric_limits<A>::min() && b == A{-1}) { return A{0}; }

  return a % b;
}

template <unsigned_integer A, unsigned_integer B>
__device__ cuda::std::expected<A, errc> mod_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  if (b == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }
  return a % b;
}

template <floating_point A, floating_point B>
__device__ cuda::std::expected<A, errc> mod_overflow(A a, B b)
  requires(cuda::std::same_as<A, B>)
{
  if (b == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }
  return cuda::std::fmod(a, b);
}

template <typename A, typename B>
__device__ cuda::std::expected<numeric::decimal<A>, errc> mod_overflow(numeric::decimal<A> a,
                                                                       numeric::decimal<B> b)
  requires(cuda::std::same_as<A, B>)
{
  if (b.value() == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }

  if (numeric::division_overflow<A>(a.value(), b.value())) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }

  return a % b;
}

/**
 * @brief Computes absolute value with overflow detection.
 *
 * @tparam T Value type.
 * @param a Input value.
 * @return `errc::ARITHMETIC_OVERFLOW` on overflow, else the result.
 */
template <signed_integer T>
__device__ cuda::std::expected<T, errc> abs_overflow(T a)
{
  if (a == cuda::std::numeric_limits<T>::min()) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }
  return (a < 0) ? -a : a;
}

template <unsigned_integer T>
__device__ cuda::std::expected<T, errc> abs_overflow(T a)
{
  return a;
}

template <floating_point T>
__device__ cuda::std::expected<T, errc> abs_overflow(T a)
{
  return cuda::std::fabs(a);
}

template <typename R>
__device__ cuda::std::expected<numeric::decimal<R>, errc> abs_overflow(numeric::decimal<R> a)
{
  if (a.value() == cuda::std::numeric_limits<R>::min()) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }
  auto rep = a.value() < 0 ? -a.value() : a.value();
  return numeric::decimal<R>{numeric::scaled_integer<R>{rep, numeric::scale_type{a.scale()}}};
}

/**
 * @brief Computes unary negation with overflow detection.
 *
 * @tparam T Value type.
 * @param a Input value.
 * @return `errc::ARITHMETIC_OVERFLOW` on overflow, else the result.
 */
template <signed_integer T>
__device__ cuda::std::expected<T, errc> neg_overflow(T a)
{
  if (a == cuda::std::numeric_limits<T>::min()) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }
  return -a;
}

template <unsigned_integer T>
__device__ cuda::std::expected<T, errc> neg_overflow(T a)
{
  if (a != 0) { return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW}; }
  return a;
}

template <floating_point T>
__device__ cuda::std::expected<T, errc> neg_overflow(T a)
{
  return -a;
}

template <typename R>
__device__ cuda::std::expected<numeric::decimal<R>, errc> neg_overflow(numeric::decimal<R> a)
{
  if (a.value() == cuda::std::numeric_limits<R>::min()) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }
  auto rep = -a.value();
  return numeric::decimal<R>{numeric::scaled_integer<R>{rep, numeric::scale_type{a.scale()}}};
}

/**
 * @brief Validates decimal precision against a target precision value.
 *
 * @tparam R Decimal representation type.
 * @tparam P Target precision type.
 * @param a Input decimal value.
 * @param precision Maximum allowed precision.
 * @return `errc::ARITHMETIC_OVERFLOW` when precision is invalid or exceeded, else the result.
 */
template <typename R, cuda::std::integral P>
__device__ cuda::std::expected<numeric::decimal<R>, errc> check_precision(numeric::decimal<R> a,
                                                                          P precision)
{
  if (precision <= 0) { return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW}; }

  auto value = a.value();
  if (value == cuda::std::numeric_limits<R>::min()) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }

  auto abs_value = value < 0 ? -value : value;

  if (abs_value >= numeric::detail::ipow<R, numeric::Radix::BASE_10>(precision)) {
    return cuda::std::unexpected{errc::ARITHMETIC_OVERFLOW};
  }

  return a;
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
