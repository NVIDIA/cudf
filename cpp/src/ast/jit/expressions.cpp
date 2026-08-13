/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ast/jit/expressions.hpp"

#include "jit/row_ir.hpp"

#include <cudf/ast/expressions.hpp>

namespace cudf {
namespace ast {
namespace jit {
namespace detail {

cudf::size_type operation::accept(cudf::ast::detail::expression_parser& visitor) const
{
  CUDF_FAIL(
    "JIT operation is an internal expression and should not be visited by expression_parser",
    std::invalid_argument);
}

std::reference_wrapper<expression const> operation::accept(
  cudf::ast::detail::expression_transformer& visitor) const
{
  CUDF_FAIL(
    "JIT operation is an internal expression and should not be visited by "
    "expression_transformer",
    std::invalid_argument);
}

bool operation::may_evaluate_null(table_view const& left,
                                  table_view const& right,
                                  rmm::cuda_stream_view stream) const
{
  CUDF_FAIL("JIT operation is an internal expression and should not be evaluated directly",
            std::invalid_argument);
}

std::unique_ptr<cudf::detail::row_ir::node> operation::accept(
  cudf::detail::row_ir::ast_converter& converter) const
{
  return converter.add_ir_node(*this);
}

}  // namespace detail
}  // namespace jit

using opcode = cudf::detail::row_ir::opcode;

namespace {

[[nodiscard]] opcode as_opcode(cudf::ast::jit::op operator_id)
{
  using enum cudf::ast::jit::op;
  switch (operator_id) {
    case IDENTITY: return opcode::IDENTITY;
    case IS_NULL: return opcode::IS_NULL;
    case COALESCE: return opcode::COALESCE;
    case PREDICATE: return opcode::PREDICATE;
    case ADD: return opcode::ADD;
    case SUB: return opcode::SUB;
    case MUL: return opcode::MUL;
    case DIV: return opcode::DIV;
    case NEG: return opcode::NEG;
    case ABS: return opcode::ABS;
    case MOD: return opcode::MOD;
    case PYMOD: return opcode::PYMOD;
    case TRUE_DIV: return opcode::TRUE_DIV;
    case FLOOR_DIV: return opcode::FLOOR_DIV;
    case ADD_OVERFLOW: return opcode::ADD_OVERFLOW;
    case SUB_OVERFLOW: return opcode::SUB_OVERFLOW;
    case MUL_OVERFLOW: return opcode::MUL_OVERFLOW;
    case DIV_OVERFLOW: return opcode::DIV_OVERFLOW;
    case NEG_OVERFLOW: return opcode::NEG_OVERFLOW;
    case ABS_OVERFLOW: return opcode::ABS_OVERFLOW;
    case MOD_OVERFLOW: return opcode::MOD_OVERFLOW;
    case CHECK_PRECISION: return opcode::CHECK_PRECISION;
    case BITWISE_AND: return opcode::BITWISE_AND;
    case BITWISE_INVERT: return opcode::BITWISE_INVERT;
    case BITWISE_OR: return opcode::BITWISE_OR;
    case BITWISE_XOR: return opcode::BITWISE_XOR;
    case BITWISE_SHIFT_LEFT: return opcode::BITWISE_SHIFT_LEFT;
    case BITWISE_SHIFT_RIGHT: return opcode::BITWISE_SHIFT_RIGHT;
    case CAST_TO_BOOL8: return opcode::CAST_TO_BOOL8;
    case CAST_TO_INT8: return opcode::CAST_TO_INT8;
    case CAST_TO_INT16: return opcode::CAST_TO_INT16;
    case CAST_TO_INT32: return opcode::CAST_TO_INT32;
    case CAST_TO_INT64: return opcode::CAST_TO_INT64;
    case CAST_TO_UINT8: return opcode::CAST_TO_UINT8;
    case CAST_TO_UINT16: return opcode::CAST_TO_UINT16;
    case CAST_TO_UINT32: return opcode::CAST_TO_UINT32;
    case CAST_TO_UINT64: return opcode::CAST_TO_UINT64;
    case CAST_TO_FLOAT32: return opcode::CAST_TO_FLOAT32;
    case CAST_TO_FLOAT64: return opcode::CAST_TO_FLOAT64;
    case CAST_TO_DECIMAL32: return opcode::CAST_TO_DECIMAL32;
    case CAST_TO_DECIMAL64: return opcode::CAST_TO_DECIMAL64;
    case CAST_TO_DECIMAL128: return opcode::CAST_TO_DECIMAL128;
    case RESCALE: return opcode::RESCALE;
    case EQUAL: return opcode::EQUAL;
    case NOT_EQUAL: return opcode::NOT_EQUAL;
    case GREATER: return opcode::GREATER;
    case GREATER_EQUAL: return opcode::GREATER_EQUAL;
    case LESS: return opcode::LESS;
    case LESS_EQUAL: return opcode::LESS_EQUAL;
    case NULL_EQUAL: return opcode::NULL_EQUAL;
    case NULL_LOGICAL_AND: return opcode::NULL_LOGICAL_AND;
    case NULL_LOGICAL_OR: return opcode::NULL_LOGICAL_OR;
    case LOGICAL_AND: return opcode::LOGICAL_AND;
    case LOGICAL_OR: return opcode::LOGICAL_OR;
    case LOGICAL_NOT: return opcode::LOGICAL_NOT;
    case IF_ELSE: return opcode::IF_ELSE;
    case CBRT: return opcode::CBRT;
    case CEIL: return opcode::CEIL;
    case FLOOR: return opcode::FLOOR;
    case RINT: return opcode::RINT;
    case SQRT: return opcode::SQRT;
    case POW: return opcode::POW;
    case EXP: return opcode::EXP;
    case LOG: return opcode::LOG;
    case ARCCOS: return opcode::ARCCOS;
    case ARCCOSH: return opcode::ARCCOSH;
    case ARCSIN: return opcode::ARCSIN;
    case ARCSINH: return opcode::ARCSINH;
    case ARCTAN: return opcode::ARCTAN;
    case ARCTANH: return opcode::ARCTANH;
    case COS: return opcode::COS;
    case COSH: return opcode::COSH;
    case SIN: return opcode::SIN;
    case SINH: return opcode::SINH;
    case TAN: return opcode::TAN;
    case TANH: return opcode::TANH;
    default: CUDF_FAIL("Invalid JIT op", std::invalid_argument);
  }
}

}  // namespace

expression const& jit::operation(ast::tree& tree,
                                 op operator_id,
                                 std::vector<std::reference_wrapper<expression const>> const& args,
                                 cudf::error_policy error_policy,
                                 std::optional<int32_t> target_scale)
{
  return tree.push(
    detail::operation(as_opcode(operator_id), std::move(args), error_policy, target_scale));
}

expression const& jit::operation(
  ast::tree& tree,
  op operator_id,
  std::initializer_list<std::reference_wrapper<expression const>> args,
  cudf::error_policy error_policy,
  std::optional<int32_t> target_scale)
{
  return operation(tree,
                   operator_id,
                   std::vector<std::reference_wrapper<expression const>>(args),
                   error_policy,
                   target_scale);
}

}  // namespace ast
}  // namespace cudf
