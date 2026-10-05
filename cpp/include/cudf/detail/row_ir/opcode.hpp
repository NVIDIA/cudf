/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/detail/operators/operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace detail::row_ir {

enum class [[nodiscard]] opcode : int32_t {
  GET_INPUT,
  SET_OUTPUT,

  // Identity operators
  IDENTITY,

  // Null handling operators
  IS_NULL,

  COALESCE,
  PREDICATE,

  /// Arithmetic operators
  ADD,
  SUB,
  MUL,
  DIV,
  NEG,
  ABS,
  MOD,
  PYMOD,
  TRUE_DIV,
  FLOOR_DIV,

  /// Overflow-checking Arithmetic functions. raise errors on overflow, division by zero, etc.
  ADD_OVERFLOW,
  SUB_OVERFLOW,
  MUL_OVERFLOW,
  DIV_OVERFLOW,
  NEG_OVERFLOW,
  ABS_OVERFLOW,
  MOD_OVERFLOW,
  CHECK_PRECISION,

  /// Bitwise operators
  BITWISE_AND,
  BITWISE_INVERT,
  BITWISE_OR,
  BITWISE_XOR,
  BITWISE_SHIFT_LEFT,
  BITWISE_SHIFT_RIGHT,

  /// Type conversion/scaling operators
  CAST_TO_BOOL8,
  CAST_TO_INT8,
  CAST_TO_INT16,
  CAST_TO_INT32,
  CAST_TO_INT64,
  CAST_TO_UINT8,
  CAST_TO_UINT16,
  CAST_TO_UINT32,
  CAST_TO_UINT64,
  CAST_TO_FLOAT32,
  CAST_TO_FLOAT64,
  CAST_TO_DECIMAL32,
  CAST_TO_DECIMAL64,
  CAST_TO_DECIMAL128,
  RESCALE,

  /// Comparison & Logic operators
  EQUAL,
  NOT_EQUAL,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
  NULL_EQUAL,
  NULL_LOGICAL_AND,
  NULL_LOGICAL_OR,
  LOGICAL_AND,
  LOGICAL_OR,
  LOGICAL_NOT,
  IF_ELSE,

  /// Mathematical operators
  CBRT,
  CEIL,
  FLOOR,
  RINT,
  SQRT,
  POW,
  EXP,
  LOG,

  /// Trigonometric operators
  ARCCOS,
  ARCCOSH,
  ARCSIN,
  ARCSINH,
  ARCTAN,
  ARCTANH,
  COS,
  COSH,
  SIN,
  SINH,
  TAN,
  TANH,
};

template <opcode op>
struct opcode_evaluator;

#define CUDF_OPCODE_EVALUATOR(Op, func_name)                \
  template <>                                               \
  struct opcode_evaluator<opcode::Op> {                     \
    template <typename... Args>                             \
    __device__ static inline constexpr auto eval(Args... a) \
      -> decltype(cudf::detail::ops::func_name(a...))       \
    { return cudf::detail::ops::func_name(a...); }          \
  };

CUDF_OPCODE_EVALUATOR(IDENTITY, identity)
CUDF_OPCODE_EVALUATOR(IS_NULL, is_null)
CUDF_OPCODE_EVALUATOR(COALESCE, coalesce)
CUDF_OPCODE_EVALUATOR(PREDICATE, predicate)
CUDF_OPCODE_EVALUATOR(ADD, add)
CUDF_OPCODE_EVALUATOR(SUB, sub)
CUDF_OPCODE_EVALUATOR(MUL, mul)
CUDF_OPCODE_EVALUATOR(DIV, div)
CUDF_OPCODE_EVALUATOR(NEG, neg)
CUDF_OPCODE_EVALUATOR(ABS, abs)
CUDF_OPCODE_EVALUATOR(MOD, mod)
CUDF_OPCODE_EVALUATOR(PYMOD, pymod)
CUDF_OPCODE_EVALUATOR(TRUE_DIV, true_div)
CUDF_OPCODE_EVALUATOR(FLOOR_DIV, floor_div)
CUDF_OPCODE_EVALUATOR(ADD_OVERFLOW, add_overflow)
CUDF_OPCODE_EVALUATOR(SUB_OVERFLOW, sub_overflow)
CUDF_OPCODE_EVALUATOR(MUL_OVERFLOW, mul_overflow)
CUDF_OPCODE_EVALUATOR(DIV_OVERFLOW, div_overflow)
CUDF_OPCODE_EVALUATOR(NEG_OVERFLOW, neg_overflow)
CUDF_OPCODE_EVALUATOR(ABS_OVERFLOW, abs_overflow)
CUDF_OPCODE_EVALUATOR(MOD_OVERFLOW, mod_overflow)
CUDF_OPCODE_EVALUATOR(CHECK_PRECISION, check_precision)
CUDF_OPCODE_EVALUATOR(BITWISE_AND, bitwise_and)
CUDF_OPCODE_EVALUATOR(BITWISE_INVERT, bitwise_invert)
CUDF_OPCODE_EVALUATOR(BITWISE_OR, bitwise_or)
CUDF_OPCODE_EVALUATOR(BITWISE_XOR, bitwise_xor)
CUDF_OPCODE_EVALUATOR(BITWISE_SHIFT_LEFT, bitwise_shift_left)
CUDF_OPCODE_EVALUATOR(BITWISE_SHIFT_RIGHT, bitwise_shift_right)
CUDF_OPCODE_EVALUATOR(CAST_TO_BOOL8, cast_to_bool8)
CUDF_OPCODE_EVALUATOR(CAST_TO_INT8, cast_to_int8)
CUDF_OPCODE_EVALUATOR(CAST_TO_INT16, cast_to_int16)
CUDF_OPCODE_EVALUATOR(CAST_TO_INT32, cast_to_int32)
CUDF_OPCODE_EVALUATOR(CAST_TO_INT64, cast_to_int64)
CUDF_OPCODE_EVALUATOR(CAST_TO_UINT8, cast_to_uint8)
CUDF_OPCODE_EVALUATOR(CAST_TO_UINT16, cast_to_uint16)
CUDF_OPCODE_EVALUATOR(CAST_TO_UINT32, cast_to_uint32)
CUDF_OPCODE_EVALUATOR(CAST_TO_UINT64, cast_to_uint64)
CUDF_OPCODE_EVALUATOR(CAST_TO_FLOAT32, cast_to_float32)
CUDF_OPCODE_EVALUATOR(CAST_TO_FLOAT64, cast_to_float64)
CUDF_OPCODE_EVALUATOR(CAST_TO_DECIMAL32, cast_to_decimal32)
CUDF_OPCODE_EVALUATOR(CAST_TO_DECIMAL64, cast_to_decimal64)
CUDF_OPCODE_EVALUATOR(CAST_TO_DECIMAL128, cast_to_decimal128)
CUDF_OPCODE_EVALUATOR(RESCALE, rescale)
CUDF_OPCODE_EVALUATOR(EQUAL, equal)
CUDF_OPCODE_EVALUATOR(NOT_EQUAL, not_equal)
CUDF_OPCODE_EVALUATOR(GREATER, greater)
CUDF_OPCODE_EVALUATOR(GREATER_EQUAL, greater_equal)
CUDF_OPCODE_EVALUATOR(LESS, less)
CUDF_OPCODE_EVALUATOR(LESS_EQUAL, less_equal)
CUDF_OPCODE_EVALUATOR(NULL_EQUAL, null_equal)
CUDF_OPCODE_EVALUATOR(NULL_LOGICAL_AND, null_logical_and)
CUDF_OPCODE_EVALUATOR(NULL_LOGICAL_OR, null_logical_or)
CUDF_OPCODE_EVALUATOR(LOGICAL_AND, logical_and)
CUDF_OPCODE_EVALUATOR(LOGICAL_OR, logical_or)
CUDF_OPCODE_EVALUATOR(LOGICAL_NOT, logical_not)
CUDF_OPCODE_EVALUATOR(IF_ELSE, if_else)
CUDF_OPCODE_EVALUATOR(CBRT, cbrt)
CUDF_OPCODE_EVALUATOR(CEIL, ceil)
CUDF_OPCODE_EVALUATOR(FLOOR, floor)
CUDF_OPCODE_EVALUATOR(RINT, rint)
CUDF_OPCODE_EVALUATOR(SQRT, sqrt)
CUDF_OPCODE_EVALUATOR(POW, pow)
CUDF_OPCODE_EVALUATOR(EXP, exp)
CUDF_OPCODE_EVALUATOR(LOG, log)
CUDF_OPCODE_EVALUATOR(ARCCOS, arccos)
CUDF_OPCODE_EVALUATOR(ARCCOSH, arccosh)
CUDF_OPCODE_EVALUATOR(ARCSIN, arcsin)
CUDF_OPCODE_EVALUATOR(ARCSINH, arcsinh)
CUDF_OPCODE_EVALUATOR(ARCTAN, arctan)
CUDF_OPCODE_EVALUATOR(ARCTANH, arctanh)
CUDF_OPCODE_EVALUATOR(COS, cos)
CUDF_OPCODE_EVALUATOR(COSH, cosh)
CUDF_OPCODE_EVALUATOR(SIN, sin)
CUDF_OPCODE_EVALUATOR(SINH, sinh)
CUDF_OPCODE_EVALUATOR(TAN, tan)
CUDF_OPCODE_EVALUATOR(TANH, tanh)

#undef CUDF_OPCODE_EVALUATOR

template <typename T>
inline constexpr bool is_fallible_result = false;

template <typename T>
inline constexpr bool is_fallible_result<cuda::std::expected<T, cudf::errc>> = true;

template <opcode op, typename... T>
concept evaluable = requires(T... args) { opcode_evaluator<op>::eval(args...); };

// evaluation of non-nullable values
template <opcode op, cudf::error_policy error_policy, typename... T>
__device__ constexpr auto evaluate(T... args)
  requires(!cudf::detail::ops::nullable<T> && ... && evaluable<op, T...>)
{
  static_assert(error_policy != cudf::error_policy::NULLIFY,
                "cudf::error_policy::NULLIFY is not supported for non-nullable types");
  return opcode_evaluator<op>::eval(args...);
}

// null-aware evaluation
template <opcode op, cudf::error_policy error_policy, typename... T>
__device__ constexpr auto evaluate(cuda::std::optional<T>... args)
  requires(evaluable<op, cuda::std::optional<T>...>)
{
  using result_t = decltype(opcode_evaluator<op>::eval(args...));

  if constexpr (is_fallible_result<result_t>) {
    // unwrap the result of the operator and propagate errors
    using optional_value_t = typename result_t::value_type;
    auto result            = opcode_evaluator<op>::eval(args...);
    if constexpr (error_policy == cudf::error_policy::NULLIFY) {
      if (!result.has_value()) {
        return optional_value_t{};
      } else {
        return result.value();
      }
    } else {
      return result;
    }
  } else {
    return opcode_evaluator<op>::eval(args...);
  }
}

// null-propagating evaluation
template <opcode op, cudf::error_policy error_policy, typename... T>
__device__ constexpr auto evaluate(cuda::std::optional<T>... args)
  requires(!evaluable<op, cuda::std::optional<T>...> && evaluable<op, T...>)
{
  using result_t = decltype(opcode_evaluator<op>::eval(args.value()...));

  if constexpr (is_fallible_result<result_t>) {
    using value_t          = typename result_t::value_type;
    using optional_value_t = cuda::std::optional<value_t>;
    using expected_t       = cuda::std::expected<optional_value_t, errc>;

    if ((!args.has_value() || ...)) {
      if constexpr (error_policy == cudf::error_policy::NULLIFY) {
        return optional_value_t{};
      } else {
        return expected_t{optional_value_t{}};
      }
    }

    auto result = opcode_evaluator<op>::eval(args.value()...);

    if constexpr (error_policy == cudf::error_policy::NULLIFY) {
      if (!result.has_value()) {
        return optional_value_t{};
      } else {
        return optional_value_t{result.value()};
      }

    } else {
      if (!result.has_value()) {
        return expected_t{cuda::std::unexpected{result.error()}};
      } else {
        return expected_t{optional_value_t{result.value()}};
      }
    }
  } else {
    using value_t          = result_t;
    using optional_value_t = cuda::std::optional<value_t>;

    if ((!args.has_value() || ...)) { return optional_value_t{}; }

    return optional_value_t{opcode_evaluator<op>::eval(args.value()...)};
  }
}

}  // namespace detail::row_ir
}  // namespace CUDF_EXPORT cudf
