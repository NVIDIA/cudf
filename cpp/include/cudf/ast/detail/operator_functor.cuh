/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/ast_operator.hpp>
#include <cudf/detail/operators/operators.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ast::detail {

template <ast_operator op>
struct operator_invoker;

#define CUDF_AST_OPERATOR_MAP(OP, func_name, num_args)                                            \
  template <>                                                                                     \
  struct operator_invoker<ast_operator::OP> {                                                     \
    static constexpr auto arity = num_args;                                                       \
    template <typename... Args>                                                                   \
    __device__ static inline auto eval(Args... a) -> decltype(cudf::detail::ops::func_name(a...)) \
    { return cudf::detail::ops::func_name(a...); }                                                \
  };

CUDF_AST_OPERATOR_MAP(ADD, add, 2)
CUDF_AST_OPERATOR_MAP(SUB, sub, 2)
CUDF_AST_OPERATOR_MAP(MUL, mul, 2)
CUDF_AST_OPERATOR_MAP(DIV, div, 2)
CUDF_AST_OPERATOR_MAP(TRUE_DIV, true_div, 2)
CUDF_AST_OPERATOR_MAP(FLOOR_DIV, floor_div, 2)
CUDF_AST_OPERATOR_MAP(MOD, mod, 2)
CUDF_AST_OPERATOR_MAP(PYMOD, pymod, 2)
CUDF_AST_OPERATOR_MAP(POW, pow, 2)
CUDF_AST_OPERATOR_MAP(EQUAL, equal, 2)
CUDF_AST_OPERATOR_MAP(NOT_EQUAL, not_equal, 2)
CUDF_AST_OPERATOR_MAP(LESS, less, 2)
CUDF_AST_OPERATOR_MAP(GREATER, greater, 2)
CUDF_AST_OPERATOR_MAP(LESS_EQUAL, less_equal, 2)
CUDF_AST_OPERATOR_MAP(GREATER_EQUAL, greater_equal, 2)
CUDF_AST_OPERATOR_MAP(BITWISE_AND, bitwise_and, 2)
CUDF_AST_OPERATOR_MAP(BITWISE_OR, bitwise_or, 2)
CUDF_AST_OPERATOR_MAP(BITWISE_XOR, bitwise_xor, 2)
CUDF_AST_OPERATOR_MAP(LOGICAL_AND, logical_and, 2)
CUDF_AST_OPERATOR_MAP(LOGICAL_OR, logical_or, 2)
CUDF_AST_OPERATOR_MAP(IDENTITY, identity, 1)
CUDF_AST_OPERATOR_MAP(SIN, sin, 1)
CUDF_AST_OPERATOR_MAP(COS, cos, 1)
CUDF_AST_OPERATOR_MAP(TAN, tan, 1)
CUDF_AST_OPERATOR_MAP(ARCSIN, arcsin, 1)
CUDF_AST_OPERATOR_MAP(ARCCOS, arccos, 1)
CUDF_AST_OPERATOR_MAP(ARCTAN, arctan, 1)
CUDF_AST_OPERATOR_MAP(SINH, sinh, 1)
CUDF_AST_OPERATOR_MAP(COSH, cosh, 1)
CUDF_AST_OPERATOR_MAP(TANH, tanh, 1)
CUDF_AST_OPERATOR_MAP(ARCSINH, arcsinh, 1)
CUDF_AST_OPERATOR_MAP(ARCCOSH, arccosh, 1)
CUDF_AST_OPERATOR_MAP(ARCTANH, arctanh, 1)
CUDF_AST_OPERATOR_MAP(EXP, exp, 1)
CUDF_AST_OPERATOR_MAP(LOG, log, 1)
CUDF_AST_OPERATOR_MAP(SQRT, sqrt, 1)
CUDF_AST_OPERATOR_MAP(CBRT, cbrt, 1)
CUDF_AST_OPERATOR_MAP(CEIL, ceil, 1)
CUDF_AST_OPERATOR_MAP(FLOOR, floor, 1)
CUDF_AST_OPERATOR_MAP(ABS, abs, 1)
CUDF_AST_OPERATOR_MAP(RINT, rint, 1)
CUDF_AST_OPERATOR_MAP(BIT_INVERT, bitwise_invert, 1)
CUDF_AST_OPERATOR_MAP(NOT, logical_not, 1)
CUDF_AST_OPERATOR_MAP(CAST_TO_INT64, cast_to_int64, 1)
CUDF_AST_OPERATOR_MAP(CAST_TO_UINT64, cast_to_uint64, 1)
CUDF_AST_OPERATOR_MAP(CAST_TO_FLOAT64, cast_to_float64, 1)
CUDF_AST_OPERATOR_MAP(IS_NULL, is_null, 1)
CUDF_AST_OPERATOR_MAP(NULL_EQUAL, null_equal, 2)
CUDF_AST_OPERATOR_MAP(NULL_LOGICAL_AND, null_logical_and, 2)
CUDF_AST_OPERATOR_MAP(NULL_LOGICAL_OR, null_logical_or, 2)

#undef CUDF_AST_OPERATOR_MAP

/**
 * @brief Operator functor.
 *
 * This functor is templated on an `ast_operator`, with each template specialization defining a
 * callable `eval` that executes the operation. The functor specialization also has a member
 * `arity` defining the number of operands that are accepted by the call to `eval`. The
 * `eval` is templated on the types of its inputs (e.g. `typename LHS` and `typename RHS` for
 * a binary operator). Trailing return types are defined as `decltype(result)` where `result` is
 * the returned value. The trailing return types allow SFINAE to only consider template
 * instantiations for valid combinations of types. This, in turn, allows the operator functors to be
 * used with traits like `is_valid_binary_op` that rely on `std::is_invocable` and related features.
 *
 * @tparam op AST operator.
 */
template <ast_operator op>
struct operator_functor {
  static constexpr auto arity = operator_invoker<op>::arity;

  template <typename T>
  __device__ inline auto operator()(T a)
    requires(!cudf::detail::ops::nullable<T> && requires { operator_invoker<op>::eval(a); })
  { return operator_invoker<op>::eval(a); }

  template <typename T>
  __device__ inline auto operator()(T a)
    requires(
    cudf::detail::ops::nullable<T>  && (requires { operator_invoker<op>::eval(a); } ||
    requires { operator_invoker<op>::eval(a.value()); })
    )
  {
    // If the operator is not defined for optional<T>, but is defined for T then it is assumed to be
    // null-propagating.
    if constexpr (requires { operator_invoker<op>::eval(a); }) {
      return operator_invoker<op>::eval(a);
    } else {
      using result_t = cuda::std::optional<decltype(operator_invoker<op>::eval(a.value()))>;
      if (a.has_value()) {
        return result_t{operator_invoker<op>::eval(a.value())};
      } else {
        return result_t{};
      }
    }
  }

  template <typename T>
  __device__ inline auto operator()(T a, T b)
    requires(!cudf::detail::ops::nullable<T> && requires { operator_invoker<op>::eval(a, b); })
  { return operator_invoker<op>::eval(a, b); }

  template <typename T>
  __device__ inline auto operator()(T a, T b)
    requires(
      cudf::detail::ops::nullable<T> &&
      (requires { operator_invoker<op>::eval(a, b); } ||
      requires { operator_invoker<op>::eval(a.value(), b.value()); }))
  {
    // If the operator is not defined for optional<T>, but is defined for T then it is assumed to be
    // null-propagating.
    if constexpr (requires { operator_invoker<op>::eval(a, b); }) {
      return operator_invoker<op>::eval(a, b);
    } else {
      using result_t =
        cuda::std::optional<decltype(operator_invoker<op>::eval(a.value(), b.value()))>;
      if (a.has_value() && b.has_value()) {
        return result_t{operator_invoker<op>::eval(a.value(), b.value())};
      } else {
        return result_t{};
      }
    }
  }

  static constexpr int32_t fixed_point_result_scale(int32_t a, int32_t b)
    requires(op == ast_operator::ADD || op == ast_operator::SUB || op == ast_operator::MUL ||
             op == ast_operator::DIV || op == ast_operator::MOD || op == ast_operator::PYMOD)
  {
    // this implementation must be kept in sync with the implementation of
    // `cudf::detail::row_ir::get_op_output_scale`
    // (https://github.com/lamarrr/cudf/blob/a3a81058ba6288cf8d5dec995925690ef7f28d69/cpp/src/jit/row_ir.cpp#L39)
    if constexpr (op == ast_operator::ADD || op == ast_operator::SUB) {
      return cuda::std::min(a, b);
    } else if constexpr (op == ast_operator::MUL) {
      return a + b;
    } else if constexpr (op == ast_operator::DIV) {
      return a - b;
    } else if constexpr (op == ast_operator::MOD) {
      return cuda::std::min(a, b);
    } else if constexpr (op == ast_operator::PYMOD) {
      return cuda::std::min(a, b);
    }
  }
};

}  // namespace ast::detail
}  // namespace CUDF_EXPORT cudf
