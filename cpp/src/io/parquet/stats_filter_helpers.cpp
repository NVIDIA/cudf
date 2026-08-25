/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "stats_filter_helpers.hpp"

#include "expression_transform_helpers.hpp"

#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf::io::parquet::detail {

stats_columns_collector::stats_columns_collector(ast::expression const& expr,
                                                 cudf::size_type num_columns)
  : _num_columns(num_columns)
{
  _columns_mask.resize(num_columns, false);
  expr.accept(*this);
}

std::reference_wrapper<ast::expression const> stats_columns_collector::visit(
  ast::literal const& expr)
{
  return expr;
}

std::reference_wrapper<ast::expression const> stats_columns_collector::visit(
  ast::column_reference const& expr)
{
  CUDF_EXPECTS(expr.get_table_source() == ast::table_reference::LEFT,
               "Statistics AST supports only left table");
  CUDF_EXPECTS(expr.get_column_index() < _num_columns,
               "Column index cannot be more than number of columns in the table");
  return expr;
}

std::reference_wrapper<ast::expression const> stats_columns_collector::visit(
  ast::column_name_reference const& expr)
{
  CUDF_FAIL("Column name reference is not supported in statistics AST");
}

std::reference_wrapper<ast::expression const> stats_columns_collector::visit(
  ast::operation const& expr)
{
  using cudf::ast::ast_operator;

  auto const input_op       = expr.get_operator();
  auto const operator_arity = cudf::ast::detail::ast_operator_arity(input_op);

  if (operator_arity == 1) {
    auto const [kind, col_ref] = extract_unary_operand(expr);

    if (kind == operand_kind::COLUMN_REF) {
      col_ref->accept(*this);
      if (input_op == ast_operator::IS_NULL) {
        _columns_mask[col_ref->get_column_index()] = true;
        _has_is_null_operator                      = true;
      }
    } else {
      std::ignore = visit_operands(expr.get_operands());
    }
    return expr;
  }

  // Binary operation
  auto const [op, lhs_kind, rhs_kind, col_ref, _] = extract_binary_operands(expr);

  if (lhs_kind == operand_kind::COLUMN_REF and rhs_kind == operand_kind::LITERAL) {
    col_ref->accept(*this);
    if (op == ast_operator::EQUAL or op == ast_operator::NOT_EQUAL or op == ast_operator::LESS or
        op == ast_operator::LESS_EQUAL or op == ast_operator::GREATER or
        op == ast_operator::GREATER_EQUAL) {
      _columns_mask[col_ref->get_column_index()] = true;
    }
  } else {
    // Visit the operands and ignore any output as we only want to build the column mask
    std::ignore = visit_operands(expr.get_operands());
  }
  return expr;
}

std::pair<thrust::host_vector<bool>, bool> stats_columns_collector::get_stats_columns_mask() &&
{
  return {std::move(_columns_mask), _has_is_null_operator};
}

stats_expression_converter::stats_expression_converter(ast::expression const& expr,
                                                       size_type num_columns,
                                                       bool has_is_null_operator,
                                                       cuda::stream_ref stream)
  : _always_true_scalar{std::make_unique<cudf::numeric_scalar<bool>>(true, true, stream)},
    _always_true{std::make_unique<ast::literal>(*_always_true_scalar)}
{
  _stats_cols_per_column = has_is_null_operator ? 3 : 2;
  _num_columns           = num_columns;
  expr.accept(*this);
}

std::reference_wrapper<ast::expression const> stats_expression_converter::visit(
  ast::operation const& expr)
{
  using cudf::ast::ast_operator;

  auto const input_op       = expr.get_operator();
  auto const operator_arity = cudf::ast::detail::ast_operator_arity(input_op);

  // Unary operation
  if (operator_arity == 1) {
    auto const [kind, col_ref] = extract_unary_operand(expr);

    if (kind == operand_kind::COLUMN_REF) {
      col_ref->accept(*this);

      auto const col_index = col_ref->get_column_index();

      // Evaluate IS_NULL unary operator
      if (input_op == ast_operator::IS_NULL) {
        CUDF_EXPECTS(std::cmp_equal(_stats_cols_per_column, 3),
                     "IS_NULL operator cannot be evaluated without nullability information column");
        auto const& vnull =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column + 2});
        _stats_expr.push(ast::operation{ast_operator::IDENTITY, vnull});
        return _stats_expr.back();
      }  // For all other unary operators, push and return the `_always_true` expression
      else {
        _stats_expr.push(ast::operation{ast_operator::IDENTITY, *_always_true});
        return *_always_true;
      }
    } else {
      // Special handling for the NOT operator since is necessary as stats transforms use different
      // columns (vmin, vmax, is_null) for different operators such that NOT(col < val) is not
      // equivalent to NOT(vmin < val) and instead is equivalent to vmax >= val.
      if (input_op == ast_operator::NOT) {
        auto const* child_operation =
          dynamic_cast<ast::operation const*>(&expr.get_operands().front().get());
        if (child_operation != nullptr) {
          auto const child_op = child_operation->get_operator();

          // If the child operator is IS_NULL, we can safely negate it without any modifications
          if (child_op == ast_operator::IS_NULL) {
            auto new_operands = visit_operands(expr.get_operands());
            if (&new_operands.front().get() == _always_true.get()) {
              _stats_expr.push(ast::operation{ast_operator::IDENTITY, _stats_expr.back()});
              return *_always_true;
            } else {
              _stats_expr.push(ast::operation{ast_operator::NOT, new_operands.front()});
              return _stats_expr.back();
            }
          }  // Binary operation wrapped
          else if (cudf::ast::detail::ast_operator_arity(child_op) == 2) {
            auto const binary_operands = extract_binary_operands(*child_operation);
            auto const lhs_kind        = binary_operands.lhs_type;
            auto const rhs_kind        = binary_operands.rhs_type;

            // For NOT(col op lit) negate the operator if negatable and visit the negated operation
            // directly
            if (lhs_kind == operand_kind::COLUMN_REF and rhs_kind == operand_kind::LITERAL) {
              auto const negated_op = transform_operator<operator_transform::NEGATE>(child_op);
              if (negated_op.has_value()) {
                auto const& child_operands = child_operation->get_operands();
                return visit(
                  ast::operation{*negated_op, child_operands.front(), child_operands.back()});
              }
            }
          }
        }
      }
      // For all other unsafe NOT forms such as NOT(expr AND expr) as well as all other unary
      // operators such as ABS(expr), visit operands and push _always_true
      std::ignore = visit_operands(expr.get_operands());
      _stats_expr.push(ast::operation{ast_operator::IDENTITY, *_always_true});
      return *_always_true;
    }
  }

  // Binary operation
  auto const [op, lhs_kind, rhs_kind, col_ref, literal_ptr] = extract_binary_operands(expr);

  // Push expressions for `col op lit` or `lit op col` forms
  if (lhs_kind == operand_kind::COLUMN_REF and rhs_kind == operand_kind::LITERAL) {
    col_ref->accept(*this);

    auto const col_index = col_ref->get_column_index();
    // Push literal into the ast::tree
    auto const& literal = _stats_expr.push(*literal_ptr);

    switch (op) {
      /* transform to stats conditions
      col == val --> vmin <= val && vmax >= val
      col != val --> !(vmin == val && vmax == val)
      col >  val --> vmax > val
      col <  val --> vmin < val
      col >= val --> vmax >= val
      col <= val --> vmin <= val
      */
      case ast_operator::EQUAL: {
        auto const& vmin =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column});
        auto const& vmax =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column + 1});
        _stats_expr.push(ast::operation{
          ast::ast_operator::LOGICAL_AND,
          _stats_expr.push(ast::operation{ast_operator::GREATER_EQUAL, vmax, literal}),
          _stats_expr.push(ast::operation{ast_operator::LESS_EQUAL, vmin, literal})});
        break;
      }
      case ast_operator::NOT_EQUAL: {
        auto const& vmin =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column});
        auto const& vmax =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column + 1});
        _stats_expr.push(
          ast::operation{ast_operator::LOGICAL_OR,
                         _stats_expr.push(ast::operation{ast_operator::NOT_EQUAL, vmin, vmax}),
                         _stats_expr.push(ast::operation{ast_operator::NOT_EQUAL, vmax, literal})});
        break;
      }
      case ast_operator::LESS: [[fallthrough]];
      case ast_operator::LESS_EQUAL: {
        auto const& vmin =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column});
        _stats_expr.push(ast::operation{op, vmin, literal});
        break;
      }
      case ast_operator::GREATER: [[fallthrough]];
      case ast_operator::GREATER_EQUAL: {
        auto const& vmax =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column + 1});
        _stats_expr.push(ast::operation{op, vmax, literal});
        break;
      }
      default: {
        _stats_expr.push(ast::operation{ast_operator::IDENTITY, *_always_true});
        return *_always_true;
      }
    };
  }  // Visit operands and push expression for `expr op expr` form
  else if (lhs_kind == operand_kind::EXPRESSION and rhs_kind == operand_kind::EXPRESSION) {
    auto new_operands = visit_operands(expr.get_operands());
    _stats_expr.push(ast::operation{op, new_operands.front(), new_operands.back()});
  }  // Push _always_true for `col op col`, `expr op col`, `expr op lit` forms
  else {
    _stats_expr.push(ast::operation{ast_operator::IDENTITY, *_always_true});
    return *_always_true;
  }
  return _stats_expr.back();
}

std::reference_wrapper<ast::expression const> stats_expression_converter::get_stats_expr() const
{
  return _stats_expr.back();
}

}  // namespace cudf::io::parquet::detail
