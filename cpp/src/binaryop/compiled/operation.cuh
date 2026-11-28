/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/operators/arithmetic.cuh>
#include <cudf/detail/operators/bitwise.cuh>
#include <cudf/detail/operators/comparison.cuh>
#include <cudf/detail/operators/logic.cuh>
#include <cudf/detail/operators/math.cuh>
#include <cudf/detail/operators/trigonometric.cuh>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace cudf::binops::compiled {

// Callable adapters preserve binary-op result types and its validity-flag interface.
namespace ops {

#define CUDF_BINARY_OPERATOR(Name, Function)                                                  \
  struct Name {                                                                               \
    template <typename A, typename B>                                                         \
    __device__ auto operator()(A a, B b) const -> decltype(cudf::detail::ops::Function(a, b)) \
    {                                                                                         \
      return cudf::detail::ops::Function(a, b);                                               \
    }                                                                                         \
  };

CUDF_BINARY_OPERATOR(Add, add)
CUDF_BINARY_OPERATOR(Sub, sub)
CUDF_BINARY_OPERATOR(TrueDiv, true_div)
CUDF_BINARY_OPERATOR(PMod, pmod)
CUDF_BINARY_OPERATOR(IntPow, int_pow)
CUDF_BINARY_OPERATOR(LogBase, log_base)
CUDF_BINARY_OPERATOR(ATan2, arctan2)
CUDF_BINARY_OPERATOR(ShiftRightUnsigned, bitwise_shift_right_unsigned)
CUDF_BINARY_OPERATOR(LogicalAnd, logical_and)
CUDF_BINARY_OPERATOR(LogicalOr, logical_or)
CUDF_BINARY_OPERATOR(Equal, equal)
CUDF_BINARY_OPERATOR(NotEqual, not_equal)
CUDF_BINARY_OPERATOR(Less, less)
CUDF_BINARY_OPERATOR(Greater, greater)
CUDF_BINARY_OPERATOR(LessEqual, less_equal)
CUDF_BINARY_OPERATOR(GreaterEqual, greater_equal)

#undef CUDF_BINARY_OPERATOR

struct Mul {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without the following line, compilation error
           // _deps/libcudacxx-src/include/cuda/std/detail/libcxx/include/chrono(917): error:
           // identifier "cuda::std::__3::ratio<(long)86400000000l, (long)1l> ::num" is undefined in
           // device code
           (is_duration<TypeLhs>() and cuda::std::is_integral<TypeRhs>()) or
           (cuda::std::is_integral<TypeLhs>() and is_duration<TypeRhs>()) or
           (is_fixed_point<TypeLhs>() and is_numeric<TypeRhs>()) or
           (is_numeric<TypeLhs>() and is_fixed_point<TypeRhs>());
  }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs * rhs)
    requires(is_supported<T1, T2>())
  {
    return cudf::detail::ops::mul(lhs, rhs);
  }
};

struct Div {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without this, compilation error on chrono:917
           (is_duration<TypeLhs>() and
            (cuda::std::is_integral<TypeRhs>() or is_duration<TypeRhs>())) or
           (is_fixed_point<TypeLhs>() and is_numeric<TypeRhs>()) or
           (is_numeric<TypeLhs>() and is_fixed_point<TypeRhs>());
  }
  template <typename T1, typename T2>
  __device__ inline auto operator()(T1 const& lhs, T2 const& rhs) -> decltype(lhs / rhs)
    requires(is_supported<T1, T2>())
  {
    return cudf::detail::ops::div(lhs, rhs);
  }
};

struct Mod {
  template <typename TypeLhs, typename TypeRhs>
  static constexpr inline bool is_supported()
  {
    return has_common_type_v<TypeLhs, TypeRhs> or
           // FIXME: without this, compilation error
           //_deps/libcudacxx-src/include/cuda/std/detail/libcxx/include/chrono(1337):
           // error : expression must have integral or unscoped enum type
           (is_duration<TypeLhs>() and
            (cuda::std::is_integral<TypeRhs>() or is_duration<TypeRhs>()));
  }
  template <typename A, typename B>
  __device__ auto operator()(A const& lhs, B const& rhs) const
    -> decltype(cudf::detail::ops::mod(lhs, rhs))
    requires(is_supported<A, B>())
  {
    return cudf::detail::ops::mod(lhs, rhs);
  }
};

// The shared AST bitwise operators exclude bool; binary operations also accept bool operands.
template <typename T>
__device__ auto promote_bool(T value)
  -> cuda::std::conditional_t<cuda::std::is_same_v<T, bool>, int, T>
{
  if constexpr (cuda::std::is_same_v<T, bool>) {
    return static_cast<int>(value);
  } else {
    return value;
  }
}

#define CUDF_INTEGRAL_BINARY_OPERATOR(Name, Function)                            \
  struct Name {                                                                  \
    template <typename A, typename B>                                            \
    __device__ auto operator()(A a, B b) const                                   \
      -> decltype(cudf::detail::ops::Function(promote_bool(a), promote_bool(b))) \
    {                                                                            \
      return cudf::detail::ops::Function(promote_bool(a), promote_bool(b));      \
    }                                                                            \
  };

CUDF_INTEGRAL_BINARY_OPERATOR(BitwiseAnd, bitwise_and)
CUDF_INTEGRAL_BINARY_OPERATOR(BitwiseOr, bitwise_or)
CUDF_INTEGRAL_BINARY_OPERATOR(BitwiseXor, bitwise_xor)
CUDF_INTEGRAL_BINARY_OPERATOR(ShiftLeft, bitwise_shift_left)
CUDF_INTEGRAL_BINARY_OPERATOR(ShiftRight, bitwise_shift_right)

#undef CUDF_INTEGRAL_BINARY_OPERATOR

struct FloorDiv {
  template <typename A, typename B>
  __device__ auto operator()(A a, B b) const -> decltype(a / b)
    requires requires { cudf::detail::ops::floor_div(promote_bool(a), promote_bool(b)); }
  {
    return cudf::detail::ops::floor_div(promote_bool(a), promote_bool(b));
  }
};

struct PyMod {
  template <typename A, typename B>
  __device__ auto operator()(A a, B b) const -> decltype(cudf::detail::ops::pymod(a, b))
    requires(cuda::std::integral<cuda::std::common_type_t<A, B>> ||
             (cudf::concepts::fixed_point<A> && cuda::std::same_as<A, B>) ||
             cudf::concepts::duration<A>)
  {
    return cudf::detail::ops::pymod(a, b);
  }

  template <typename A, typename B>
  __device__ double operator()(A a, B b) const
    requires(cuda::std::floating_point<cuda::std::common_type_t<A, B>>)
  {
    return cudf::detail::ops::pymod(static_cast<double>(a), static_cast<double>(b));
  }
};

struct Pow {
  template <typename A, typename B>
  __device__ double operator()(A a, B b) const
    requires(cuda::std::is_convertible_v<A, double> && cuda::std::is_convertible_v<B, double>)
  {
    return cudf::detail::ops::pow(static_cast<double>(a), static_cast<double>(b));
  }
};

template <typename T>
__device__ auto optional_value(T value, bool valid)
{
  return valid ? cuda::std::optional<T>{value} : cuda::std::nullopt;
}

template <typename T>
__device__ auto validity_result(T result, bool& valid)
{
  if constexpr (cudf::detail::ops::nullable<T>) {
    valid = result.has_value();
    return result.value_or(typename T::value_type{});
  } else {
    valid = true;
    return result;
  }
}

#define CUDF_NULL_BINARY_OPERATOR(Name, Function)                                              \
  struct Name {                                                                                \
    template <typename A, typename B>                                                          \
    __device__ auto operator()(A a, B b, bool a_valid, bool b_valid, bool& output_valid) const \
      -> decltype(cudf::detail::ops::Function(a, b))                                           \
    {                                                                                          \
      return validity_result(                                                                  \
        cudf::detail::ops::Function(optional_value(a, a_valid), optional_value(b, b_valid)),   \
        output_valid);                                                                         \
    }                                                                                          \
    template <typename A, typename B>                                                          \
    __device__ auto operator()(A a, B b) const -> decltype(cudf::detail::ops::Function(a, b)); \
  };

CUDF_NULL_BINARY_OPERATOR(NullEquals, null_equal)
CUDF_NULL_BINARY_OPERATOR(NullNotEquals, null_not_equal)
CUDF_NULL_BINARY_OPERATOR(NullMax, null_max)
CUDF_NULL_BINARY_OPERATOR(NullMin, null_min)
CUDF_NULL_BINARY_OPERATOR(NullLogicalAnd, null_logical_and)
CUDF_NULL_BINARY_OPERATOR(NullLogicalOr, null_logical_or)

#undef CUDF_NULL_BINARY_OPERATOR

}  // namespace ops
}  // namespace cudf::binops::compiled
