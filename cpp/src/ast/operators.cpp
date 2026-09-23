/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/ast/detail/operators.cuh>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/type_traits>

#include <optional>
#include <vector>

namespace cudf {
namespace ast {
namespace detail {
namespace {

struct arity_functor {
  template <ast_operator op>
  void operator()(cudf::size_type& result)
  {
    result = operator_functor<op>::arity;
  }
};

/**
 * @brief Functor to determine the return type of an operator from its input types.
 */
struct return_type_functor {
  std::span<cudf::data_type const> operand_types;

  /**
   * @brief Callable for binary operators to determine return type.
   *
   * @tparam OperatorFunctor Operator functor to perform.
   * @tparam LHS Left input type.
   * @tparam RHS Right input type.
   * @param result Pointer whose value is assigned to the result data type.
   */
  template <typename OperatorFunctor, typename LHS, typename RHS>
  void operator()(cudf::data_type& result)
    requires(is_valid_binary_op<OperatorFunctor, LHS, RHS>)
  {
    using Out         = cuda::std::invoke_result_t<OperatorFunctor, LHS, RHS>;
    auto const out_id = cudf::type_to_id<Out>();
    if constexpr (cudf::is_fixed_point<Out>()) {
      auto const lhs_type = operand_types[0];
      auto const rhs_type = operand_types[1];
      result              = cudf::data_type(
        out_id, OperatorFunctor::fixed_point_result_scale(lhs_type.scale(), rhs_type.scale()));
    } else {
      result = cudf::data_type{out_id};
    }
  }

  template <typename OperatorFunctor, typename LHS, typename RHS>
  void operator()(cudf::data_type& result)
    requires(!is_valid_binary_op<OperatorFunctor, LHS, RHS>)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid binary operation. Return type cannot be determined.");
#else
    CUDF_UNREACHABLE("Invalid binary operation. Return type cannot be determined.");
#endif
    result = cudf::data_type{cudf::type_id::EMPTY};
  }

  /**
   * @brief Callable for unary operators to determine return type.
   *
   * @tparam OperatorFunctor Operator functor to perform.
   * @tparam T Input type.
   * @param result Pointer whose value is assigned to the result data type.
   */
  template <typename OperatorFunctor, typename T>
  void operator()(cudf::data_type& result)
    requires(is_valid_unary_op<OperatorFunctor, T>)
  {
    using Out         = cuda::std::invoke_result_t<OperatorFunctor, T>;
    auto const out_id = cudf::type_to_id<Out>();
    if constexpr (cudf::is_fixed_point<Out>()) {
      result = cudf::data_type(out_id, operand_types[0].scale());
    } else {
      result = cudf::data_type{out_id};
    }
  }

  template <typename OperatorFunctor, typename T>
  void operator()(cudf::data_type& result)
    requires(!is_valid_unary_op<OperatorFunctor, T>)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid unary operation. Return type cannot be determined.");
#else
    CUDF_UNREACHABLE("Invalid unary operation. Return type cannot be determined.");
#endif
    result = cudf::data_type{cudf::type_id::EMPTY};
  }
};

/**
 * @brief Functor used to single-type-dispatch binary operators.
 *
 * This functor's `operator()` is templated to validate calls to its operators based on the input
 * type, as determined by the `is_valid_binary_op` trait. This function assumes that both inputs are
 * the same type, and dispatches based on the type of the left input.
 *
 * @tparam OperatorFunctor Binary operator functor.
 */
template <typename OperatorFunctor>
struct single_dispatch_binary_operator_types {
  template <typename LHS, typename F, typename... Ts>
  inline void operator()(F&& f, Ts&&... args)
    requires(is_valid_binary_op<OperatorFunctor, LHS, LHS>)
  {
    f.template operator()<OperatorFunctor, LHS, LHS>(std::forward<Ts>(args)...);
  }

  template <typename LHS, typename F, typename... Ts>
  inline void operator()(F&& f, Ts&&... args)
    requires(!is_valid_binary_op<OperatorFunctor, LHS, LHS>)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid binary operation.");
#else
    CUDF_UNREACHABLE("Invalid binary operation.");
#endif
  }
};

/**
 * @brief Functor performing a type dispatch for a binary operator.
 *
 * This functor performs single dispatch, which assumes lhs_type == rhs_type. This may not be true
 * for all binary operators but holds for all currently implemented operators.
 */
struct type_dispatch_binary_op {
  /**
   * @brief Performs type dispatch for a binary operator.
   *
   * @tparam op AST operator.
   * @tparam F Type of forwarded functor.
   * @tparam Ts Parameter pack of forwarded arguments.
   * @param lhs_type Type of left input data.
   * @param rhs_type Type of right input data.
   * @param f Forwarded functor to be called.
   * @param args Forwarded arguments to `operator()` of `f`.
   */
  template <ast_operator op, typename F, typename... Ts>
  inline void operator()(cudf::data_type lhs_type, cudf::data_type rhs_type, F&& f, Ts&&... args)
  {
    // Single dispatch (assume lhs_type == rhs_type)
    type_dispatcher(
      lhs_type,
      // Always dispatch to the non-null operator for the purpose of type determination.
      detail::single_dispatch_binary_operator_types<operator_functor<op>>{},
      std::forward<F>(f),
      std::forward<Ts>(args)...);
  }
};

/**
 * @brief Dispatches a runtime binary operator to a templated type dispatcher.
 *
 * @tparam F Type of forwarded functor.
 * @tparam Ts Parameter pack of forwarded arguments.
 * @param op The binary operator to dispatch
 * @param input_types Types for the input data
 * @param f Forwarded functor to be called.
 * @param args Forwarded arguments to `operator()` of `f`.
 */
template <typename F, typename... Ts>
inline constexpr void binary_operator_dispatcher(ast_operator op,
                                                 std::span<cudf::data_type const> input_types,
                                                 F&& f,
                                                 Ts&&... args)
{
  ast_operator_dispatcher(op,
                          detail::type_dispatch_binary_op{},
                          input_types[0],
                          input_types[1],
                          std::forward<F>(f),
                          std::forward<Ts>(args)...);
}

/**
 * @brief Functor used to type-dispatch unary operators.
 *
 * This functor's `operator()` is templated to validate calls to its operators based on the input
 * type, as determined by the `is_valid_unary_op` trait.
 *
 * @tparam OperatorFunctor Unary operator functor.
 */
template <typename OperatorFunctor>
struct dispatch_unary_operator_types {
  template <typename InputT, typename F, typename... Ts>
  inline void operator()(F&& f, Ts&&... args)
    requires(is_valid_unary_op<OperatorFunctor, InputT>)
  {
    f.template operator()<OperatorFunctor, InputT>(std::forward<Ts>(args)...);
  }

  template <typename InputT, typename F, typename... Ts>
  inline void operator()(F&& f, Ts&&... args)
    requires(!is_valid_unary_op<OperatorFunctor, InputT>)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid unary operation.");
#else
    CUDF_UNREACHABLE("Invalid unary operation.");
#endif
  }
};

/**
 * @brief Functor performing a type dispatch for a unary operator.
 */
struct type_dispatch_unary_op {
  template <ast_operator op, typename F, typename... Ts>
  inline void operator()(cudf::data_type input_type, F&& f, Ts&&... args)
  {
    type_dispatcher(
      input_type,
      // Always dispatch to the non-null operator for the purpose of type determination.
      detail::dispatch_unary_operator_types<operator_functor<op>>{},
      std::forward<F>(f),
      std::forward<Ts>(args)...);
  }
};

/**
 * @brief Dispatches a runtime unary operator to a templated type dispatcher.
 *
 * @tparam F Type of forwarded functor.
 * @tparam Ts Parameter pack of forwarded arguments.
 * @param op The unary operator to dispatch
 * @param input_types Types for the input data
 * @param f Forwarded functor to be called.
 * @param args Forwarded arguments to `operator()` of `f`.
 */
template <typename F, typename... Ts>
inline constexpr void unary_operator_dispatcher(ast_operator op,
                                                std::span<cudf::data_type const> input_types,
                                                F&& f,
                                                Ts&&... args)
{
  ast_operator_dispatcher(op,
                          detail::type_dispatch_unary_op{},
                          input_types[0],
                          std::forward<F>(f),
                          std::forward<Ts>(args)...);
}

}  // namespace

cudf::data_type ast_operator_return_type(ast_operator op,
                                         std::vector<cudf::data_type> const& operand_types)
{
  cudf::data_type result{cudf::type_id::EMPTY};
  switch (operand_types.size()) {
    case 1: {
      unary_operator_dispatcher(
        op, operand_types, detail::return_type_functor{operand_types}, result);
      break;
    }
    case 2: {
      binary_operator_dispatcher(
        op, operand_types, detail::return_type_functor{operand_types}, result);
      break;
    }
    default: CUDF_FAIL("Unsupported operator return type."); break;
  }
  return result;
}

bool is_comparison_operator(ast_operator op)
{
  return op == ast_operator::EQUAL || op == ast_operator::NULL_EQUAL ||
         op == ast_operator::NOT_EQUAL || op == ast_operator::LESS || op == ast_operator::GREATER ||
         op == ast_operator::LESS_EQUAL || op == ast_operator::GREATER_EQUAL;
}

cudf::size_type ast_operator_arity(ast_operator op)
{
  cudf::size_type result{};
  ast_operator_dispatcher(op, arity_functor{}, result);
  return result;
}

std::string_view ast_operator_string(ast_operator op)
{
  switch (op) {
    case ast_operator::ADD: return "ADD";
    case ast_operator::SUB: return "SUB";
    case ast_operator::MUL: return "MUL";
    case ast_operator::DIV: return "DIV";
    case ast_operator::TRUE_DIV: return "TRUE_DIV";
    case ast_operator::FLOOR_DIV: return "FLOOR_DIV";
    case ast_operator::MOD: return "MOD";
    case ast_operator::PYMOD: return "PYMOD";
    case ast_operator::POW: return "POW";
    case ast_operator::EQUAL: return "EQUAL";
    case ast_operator::NULL_EQUAL: return "NULL_EQUAL";
    case ast_operator::NOT_EQUAL: return "NOT_EQUAL";
    case ast_operator::LESS: return "LESS";
    case ast_operator::GREATER: return "GREATER";
    case ast_operator::LESS_EQUAL: return "LESS_EQUAL";
    case ast_operator::GREATER_EQUAL: return "GREATER_EQUAL";
    case ast_operator::BITWISE_AND: return "BITWISE_AND";
    case ast_operator::BITWISE_OR: return "BITWISE_OR";
    case ast_operator::BITWISE_XOR: return "BITWISE_XOR";
    case ast_operator::LOGICAL_AND: return "LOGICAL_AND";
    case ast_operator::NULL_LOGICAL_AND: return "NULL_LOGICAL_AND";
    case ast_operator::LOGICAL_OR: return "LOGICAL_OR";
    case ast_operator::NULL_LOGICAL_OR: return "NULL_LOGICAL_OR";
    case ast_operator::IDENTITY: return "IDENTITY";
    case ast_operator::IS_NULL: return "IS_NULL";
    case ast_operator::SIN: return "SIN";
    case ast_operator::COS: return "COS";
    case ast_operator::TAN: return "TAN";
    case ast_operator::ARCSIN: return "ARCSIN";
    case ast_operator::ARCCOS: return "ARCCOS";
    case ast_operator::ARCTAN: return "ARCTAN";
    case ast_operator::SINH: return "SINH";
    case ast_operator::COSH: return "COSH";
    case ast_operator::TANH: return "TANH";
    case ast_operator::ARCSINH: return "ARCSINH";
    case ast_operator::ARCCOSH: return "ARCCOSH";
    case ast_operator::ARCTANH: return "ARCTANH";
    case ast_operator::EXP: return "EXP";
    case ast_operator::LOG: return "LOG";
    case ast_operator::SQRT: return "SQRT";
    case ast_operator::CBRT: return "CBRT";
    case ast_operator::CEIL: return "CEIL";
    case ast_operator::FLOOR: return "FLOOR";
    case ast_operator::ABS: return "ABS";
    case ast_operator::RINT: return "RINT";
    case ast_operator::BIT_INVERT: return "BIT_INVERT";
    case ast_operator::NOT: return "NOT";
    case ast_operator::CAST_TO_INT64: return "CAST_TO_INT64";
    case ast_operator::CAST_TO_UINT64: return "CAST_TO_UINT64";
    case ast_operator::CAST_TO_FLOAT64: return "CAST_TO_FLOAT64";
    default: CUDF_FAIL("Unrecognized operator type.");
  }
}

}  // namespace detail

}  // namespace ast

}  // namespace cudf
