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

namespace {

/**
 * @brief Maps a logical connective to its null-aware equivalent, returning any other operator as is
 *
 * A null in a statistics column means the writer did not record the statistic, so it reads as
 * "unknown, keep this chunk". The null-aware connectives keep a decisive verdict decisive
 * (`false AND unknown` is false); the plain ones return null if either side is null, letting one
 * absent statistic switch off pruning for the whole expression.
 */
[[nodiscard]] ast::ast_operator null_aware_operator(ast::ast_operator op)
{
  switch (op) {
    case ast::ast_operator::LOGICAL_AND: return ast::ast_operator::NULL_LOGICAL_AND;
    case ast::ast_operator::LOGICAL_OR: return ast::ast_operator::NULL_LOGICAL_OR;
    default: return op;
  }
}

/**
 * @brief Returns whether a comparison operator can prune row groups via statistics
 *
 * Some Parquet writers exclude `NaN`s from stats, so a floating-point chunk holding a NaN is
 * indistinguishable from one that does not. `col != val` is the only comparison leaf a NaN
 * satisfies, so it is the only one we cannot prune.
 *
 * @param op The comparison operator
 * @param dtype The data type of the column being compared
 * @return true if the comparison can be used to prune row groups
 */
[[nodiscard]] bool is_prunable_comparison(ast::ast_operator op, cudf::data_type dtype)
{
  using cudf::ast::ast_operator;
  switch (op) {
    case ast_operator::EQUAL: [[fallthrough]];
    case ast_operator::LESS: [[fallthrough]];
    case ast_operator::LESS_EQUAL: [[fallthrough]];
    case ast_operator::GREATER: [[fallthrough]];
    case ast_operator::GREATER_EQUAL: return true;
    case ast_operator::NOT_EQUAL: return not cudf::is_floating_point(dtype);
    default: return false;
  }
}

}  // namespace

stats_columns_collector::stats_columns_collector(std::span<cudf::data_type const> output_dtypes)
  : _output_dtypes(output_dtypes)
{
  _columns_mask.resize(_output_dtypes.size(), false);
}

stats_columns_collector::stats_columns_collector(ast::expression const& expr,
                                                 std::span<cudf::data_type const> output_dtypes)
  : stats_columns_collector(output_dtypes)
{
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
  CUDF_EXPECTS(static_cast<size_t>(expr.get_column_index()) < _output_dtypes.size(),
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
      if (input_op == ast_operator::IS_NULL) { _columns_mask[col_ref->get_column_index()] = true; }
    } else {
      std::ignore = visit_operands(expr.get_operands());
    }
    return expr;
  }

  // Binary operation
  auto const [op, lhs_kind, rhs_kind, col_ref, _] = extract_binary_operands(expr);

  if (lhs_kind == operand_kind::COLUMN_REF and rhs_kind == operand_kind::LITERAL) {
    col_ref->accept(*this);
    auto const col_index = col_ref->get_column_index();
    if (is_prunable_comparison(op, _output_dtypes[col_index])) { _columns_mask[col_index] = true; }
  } else {
    // Visit the operands and ignore any output as we only want to build the column mask
    std::ignore = visit_operands(expr.get_operands());
  }
  return expr;
}

thrust::host_vector<bool> stats_columns_collector::get_stats_columns_mask() &&
{
  return std::move(_columns_mask);
}

stats_expression_converter::stats_expression_converter(
  ast::expression const& expr,
  std::span<cudf::data_type const> output_dtypes,
  cuda::stream_ref stream)
  : stats_columns_collector{output_dtypes},
    _always_true_scalar{std::make_unique<cudf::numeric_scalar<bool>>(true, true, stream)},
    _always_true{std::make_unique<ast::literal>(*_always_true_scalar)}
{
  _stats_cols_per_column = 3;
  expr.accept(*this);
}

void stats_expression_converter::push_non_null_guard(size_type col_index,
                                                     ast::expression const& stats_expr)
{
  using cudf::ast::ast_operator;

  auto const& all_null =
    _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column + 2});
  // Answering "not entirely null" takes all three of the column's states, so a plain NOT will not
  // do: its null state says the chunk holds both nulls and values, or that the writer recorded no
  // null count, and both of those answer this question true. NOT alone answers it null and hands an
  // unknown to a comparison that is in fact decisive.
  auto const& not_all_null = _stats_expr.push(
    ast::operation{ast_operator::NULL_LOGICAL_OR,
                   _stats_expr.push(ast::operation{ast_operator::IS_NULL, all_null}),
                   _stats_expr.push(ast::operation{ast_operator::NOT, all_null})});
  // Null-aware so that the false this side pushes for an all-null chunk prunes it even though the
  // min and max it lacks leave `stats_expr` unknown.
  _stats_expr.push(ast::operation{ast_operator::NULL_LOGICAL_AND, not_all_null, stats_expr});
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
      // `parquet_filter_normalizer::push_down_negation` deliberately does not complement ordering
      // comparisons (NaN makes `NOT(a < b)` differ from `a >= b`), so `NOT(col op lit)` forms
      // reach here. Stats transforms use different columns (vmin, vmax, is_null) for different
      // operators such that NOT(col < val) is not equivalent to NOT(vmin < val) and instead is
      // equivalent to vmax >= val.
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
            // For NOT(col op lit) or NOT(lit op col), negate the operator if negatable and visit
            // the negated operation directly.
            auto const binary_operands = extract_binary_operands(*child_operation);
            auto const lhs_kind        = binary_operands.lhs_type;
            auto const rhs_kind        = binary_operands.rhs_type;

            // `col_ref` is only non-null for the `col op lit` form, so both checks below must
            // stay inside this branch
            if (lhs_kind == operand_kind::COLUMN_REF and rhs_kind == operand_kind::LITERAL) {
              binary_operands.col_ref->accept(*this);

              // A comparison cannot be negated when the column may hold a `NaN` (floating points).
              if (not cudf::is_floating_point(
                    _output_dtypes[binary_operands.col_ref->get_column_index()])) {
                auto const negated_op =
                  transform_operator<operator_transform::NEGATE>(child_operation->get_operator());
                if (negated_op.has_value()) {
                  auto const& child_operands = child_operation->get_operands();
                  return visit(
                    ast::operation{*negated_op, child_operands.front(), child_operands.back()});
                }
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

    // Some Parquet writers exclude `NaN`s from stats, so we can't reliably prune row groups for
    // columns that may contain them.
    if (not is_prunable_comparison(op, _output_dtypes[col_index])) {
      _stats_expr.push(ast::operation{ast_operator::IDENTITY, *_always_true});
      return *_always_true;
    }

    // Push literal into the ast::tree
    auto const& literal = _stats_expr.push(*literal_ptr);

    switch (op) {
      /* transform to stats conditions
      col == val --> vmin <= val && vmax >= val
      col != val --> vmin != vmax || vmax != val
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
        // The two halves are separately optional in the statistics, so they are combined null-aware
        // to keep whichever one is present decisive.
        auto const& in_range = _stats_expr.push(ast::operation{
          ast::ast_operator::NULL_LOGICAL_AND,
          _stats_expr.push(ast::operation{ast_operator::GREATER_EQUAL, vmax, literal}),
          _stats_expr.push(ast::operation{ast_operator::LESS_EQUAL, vmin, literal})});
        // An all-null chunk has no min or max, so this range test is unknown there and would keep
        // the chunk. The guard makes it prune instead.
        push_non_null_guard(col_index, in_range);
        break;
      }
      case ast_operator::NOT_EQUAL: {
        auto const& vmin =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column});
        auto const& vmax =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column + 1});
        // Null-aware for the same reason as the range test above: either half can be the one the
        // statistics carry.
        auto const& outside_range = _stats_expr.push(
          ast::operation{ast_operator::NULL_LOGICAL_OR,
                         _stats_expr.push(ast::operation{ast_operator::NOT_EQUAL, vmin, vmax}),
                         _stats_expr.push(ast::operation{ast_operator::NOT_EQUAL, vmax, literal})});
        // A null does not satisfy `!=` either, and an all-null chunk has no min or max to make this
        // test decisive, so the guard prunes it.
        push_non_null_guard(col_index, outside_range);
        break;
      }
      case ast_operator::LESS: [[fallthrough]];
      case ast_operator::LESS_EQUAL: {
        auto const& vmin =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column});
        // An all-null chunk has no min, leaving this test unknown, so the guard prunes it.
        push_non_null_guard(col_index, _stats_expr.push(ast::operation{op, vmin, literal}));
        break;
      }
      case ast_operator::GREATER: [[fallthrough]];
      case ast_operator::GREATER_EQUAL: {
        auto const& vmax =
          _stats_expr.push(ast::column_reference{col_index * _stats_cols_per_column + 1});
        // An all-null chunk has no max, leaving this test unknown, so the guard prunes it.
        push_non_null_guard(col_index, _stats_expr.push(ast::operation{op, vmax, literal}));
        break;
      }
      default: CUDF_UNREACHABLE("Non-prunable operator should not reach stats conversion");
    };
  }  // Visit operands and push expression for `expr op expr` form
  else if (lhs_kind == operand_kind::EXPRESSION and rhs_kind == operand_kind::EXPRESSION) {
    auto new_operands = visit_operands(expr.get_operands());
    _stats_expr.push(
      ast::operation{null_aware_operator(op), new_operands.front(), new_operands.back()});
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
