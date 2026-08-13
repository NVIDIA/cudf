/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/detail/operator_functor.cuh>
#include <cudf/ast/detail/possibly_null.cuh>
#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace ast::detail {

// Traits for valid operator / type combinations
template <typename Op, typename LHS, typename RHS>
constexpr bool is_valid_binary_op = requires(Op op, LHS lhs, RHS rhs) { op(lhs, rhs); };

template <typename Op, typename T>
constexpr bool is_valid_unary_op = requires(Op op, T value) { op(value); };

/**
 * @brief Operator dispatcher
 *
 * @tparam F Type of forwarded functor.
 * @tparam Ts Parameter pack of forwarded arguments.
 * @param f Forwarded functor to be called.
 * @param args Forwarded arguments to `operator()` of `f`.
 */
template <typename F, typename... Ts>
CUDF_HOST_DEVICE inline constexpr decltype(auto) ast_operator_dispatcher(ast_operator op,
                                                                         F&& f,
                                                                         Ts&&... args)
{
  switch (op) {
    case ast_operator::ADD:
      return f.template operator()<ast_operator::ADD>(cuda::std::forward<Ts>(args)...);
    case ast_operator::SUB:
      return f.template operator()<ast_operator::SUB>(cuda::std::forward<Ts>(args)...);
    case ast_operator::MUL:
      return f.template operator()<ast_operator::MUL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::DIV:
      return f.template operator()<ast_operator::DIV>(cuda::std::forward<Ts>(args)...);
    case ast_operator::TRUE_DIV:
      return f.template operator()<ast_operator::TRUE_DIV>(cuda::std::forward<Ts>(args)...);
    case ast_operator::FLOOR_DIV:
      return f.template operator()<ast_operator::FLOOR_DIV>(cuda::std::forward<Ts>(args)...);
    case ast_operator::MOD:
      return f.template operator()<ast_operator::MOD>(cuda::std::forward<Ts>(args)...);
    case ast_operator::PYMOD:
      return f.template operator()<ast_operator::PYMOD>(cuda::std::forward<Ts>(args)...);
    case ast_operator::POW:
      return f.template operator()<ast_operator::POW>(cuda::std::forward<Ts>(args)...);
    case ast_operator::EQUAL:
      return f.template operator()<ast_operator::EQUAL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::NULL_EQUAL:
      return f.template operator()<ast_operator::NULL_EQUAL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::NOT_EQUAL:
      return f.template operator()<ast_operator::NOT_EQUAL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::LESS:
      return f.template operator()<ast_operator::LESS>(cuda::std::forward<Ts>(args)...);
    case ast_operator::GREATER:
      return f.template operator()<ast_operator::GREATER>(cuda::std::forward<Ts>(args)...);
    case ast_operator::LESS_EQUAL:
      return f.template operator()<ast_operator::LESS_EQUAL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::GREATER_EQUAL:
      return f.template operator()<ast_operator::GREATER_EQUAL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::BITWISE_AND:
      return f.template operator()<ast_operator::BITWISE_AND>(cuda::std::forward<Ts>(args)...);
    case ast_operator::BITWISE_OR:
      return f.template operator()<ast_operator::BITWISE_OR>(cuda::std::forward<Ts>(args)...);
    case ast_operator::BITWISE_XOR:
      return f.template operator()<ast_operator::BITWISE_XOR>(cuda::std::forward<Ts>(args)...);
    case ast_operator::LOGICAL_AND:
      return f.template operator()<ast_operator::LOGICAL_AND>(cuda::std::forward<Ts>(args)...);
    case ast_operator::NULL_LOGICAL_AND:
      return f.template operator()<ast_operator::NULL_LOGICAL_AND>(cuda::std::forward<Ts>(args)...);
    case ast_operator::LOGICAL_OR:
      return f.template operator()<ast_operator::LOGICAL_OR>(cuda::std::forward<Ts>(args)...);
    case ast_operator::NULL_LOGICAL_OR:
      return f.template operator()<ast_operator::NULL_LOGICAL_OR>(cuda::std::forward<Ts>(args)...);
    case ast_operator::IDENTITY:
      return f.template operator()<ast_operator::IDENTITY>(cuda::std::forward<Ts>(args)...);
    case ast_operator::IS_NULL:
      return f.template operator()<ast_operator::IS_NULL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::SIN:
      return f.template operator()<ast_operator::SIN>(cuda::std::forward<Ts>(args)...);
    case ast_operator::COS:
      return f.template operator()<ast_operator::COS>(cuda::std::forward<Ts>(args)...);
    case ast_operator::TAN:
      return f.template operator()<ast_operator::TAN>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCSIN:
      return f.template operator()<ast_operator::ARCSIN>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCCOS:
      return f.template operator()<ast_operator::ARCCOS>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCTAN:
      return f.template operator()<ast_operator::ARCTAN>(cuda::std::forward<Ts>(args)...);
    case ast_operator::SINH:
      return f.template operator()<ast_operator::SINH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::COSH:
      return f.template operator()<ast_operator::COSH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::TANH:
      return f.template operator()<ast_operator::TANH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCSINH:
      return f.template operator()<ast_operator::ARCSINH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCCOSH:
      return f.template operator()<ast_operator::ARCCOSH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCTANH:
      return f.template operator()<ast_operator::ARCTANH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::EXP:
      return f.template operator()<ast_operator::EXP>(cuda::std::forward<Ts>(args)...);
    case ast_operator::LOG:
      return f.template operator()<ast_operator::LOG>(cuda::std::forward<Ts>(args)...);
    case ast_operator::SQRT:
      return f.template operator()<ast_operator::SQRT>(cuda::std::forward<Ts>(args)...);
    case ast_operator::CBRT:
      return f.template operator()<ast_operator::CBRT>(cuda::std::forward<Ts>(args)...);
    case ast_operator::CEIL:
      return f.template operator()<ast_operator::CEIL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::FLOOR:
      return f.template operator()<ast_operator::FLOOR>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ABS:
      return f.template operator()<ast_operator::ABS>(cuda::std::forward<Ts>(args)...);
    case ast_operator::RINT:
      return f.template operator()<ast_operator::RINT>(cuda::std::forward<Ts>(args)...);
    case ast_operator::BIT_INVERT:
      return f.template operator()<ast_operator::BIT_INVERT>(cuda::std::forward<Ts>(args)...);
    case ast_operator::NOT:
      return f.template operator()<ast_operator::NOT>(cuda::std::forward<Ts>(args)...);
    case ast_operator::CAST_TO_INT64:
      return f.template operator()<ast_operator::CAST_TO_INT64>(cuda::std::forward<Ts>(args)...);
    case ast_operator::CAST_TO_UINT64:
      return f.template operator()<ast_operator::CAST_TO_UINT64>(cuda::std::forward<Ts>(args)...);
    case ast_operator::CAST_TO_FLOAT64:
      return f.template operator()<ast_operator::CAST_TO_FLOAT64>(cuda::std::forward<Ts>(args)...);
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Invalid operator.");
#else
      CUDF_UNREACHABLE("Invalid operator.");
#endif
    }
  }
}

}  // namespace ast::detail
}  // namespace CUDF_EXPORT cudf
