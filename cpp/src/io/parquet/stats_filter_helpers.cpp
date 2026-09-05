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
  ast::expression const& expr, std::span<cudf::data_type const> output_dtypes)
  : pruning_expression_builder{output_dtypes}
{
  _stats_expr = build(expr);
}

ast::expression const& stats_expression_converter::push_non_null_guard(
  size_type col_index, ast::expression const& stats_expr)
{
  using cudf::ast::ast_operator;

  auto const& all_null = _tree.push(ast::column_reference{col_index * stats_cols_per_column + 2});
  // Answering "not entirely null" takes all three of the column's states, so a plain NOT will not
  // do: its null state says the chunk holds both nulls and values, or that the writer recorded no
  // null count, and both of those answer this question true. NOT alone answers it null and hands an
  // unknown to a comparison that is in fact decisive.
  auto const& not_all_null =
    _tree.push(ast::operation{ast_operator::NULL_LOGICAL_OR,
                              _tree.push(ast::operation{ast_operator::IS_NULL, all_null}),
                              _tree.push(ast::operation{ast_operator::NOT, all_null})});
  // Null-aware so that the false this side pushes for an all-null chunk prunes it even though the
  // min and max it lacks leave `stats_expr` unknown.
  return _tree.push(ast::operation{ast_operator::NULL_LOGICAL_AND, not_all_null, stats_expr});
}

maybe_pruning_expr stats_expression_converter::build_comparison(
  ast::ast_operator op, ast::column_reference const& col_ref, ast::literal const& literal_ref)
{
  using cudf::ast::ast_operator;

  auto const col_index = col_ref.get_column_index();

  // Some Parquet writers exclude `NaN`s from stats, so we can't reliably prune row groups for
  // columns that may contain them.
  if (not is_prunable_comparison(op, _output_dtypes[col_index])) { return std::nullopt; }

  auto const& literal = _tree.push(literal_ref);

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
      auto const& vmin = _tree.push(ast::column_reference{col_index * stats_cols_per_column});
      auto const& vmax = _tree.push(ast::column_reference{col_index * stats_cols_per_column + 1});
      // The two halves are separately optional in the statistics, so they are combined null-aware
      // to keep whichever one is present decisive.
      auto const& in_range = _tree.push(
        ast::operation{ast_operator::NULL_LOGICAL_AND,
                       _tree.push(ast::operation{ast_operator::GREATER_EQUAL, vmax, literal}),
                       _tree.push(ast::operation{ast_operator::LESS_EQUAL, vmin, literal})});
      // An all-null chunk has no min or max, so this range test is unknown there and would keep
      // the chunk. The guard makes it prune instead.
      return push_non_null_guard(col_index, in_range);
    }
    case ast_operator::NOT_EQUAL: {
      auto const& vmin = _tree.push(ast::column_reference{col_index * stats_cols_per_column});
      auto const& vmax = _tree.push(ast::column_reference{col_index * stats_cols_per_column + 1});
      // Null-aware for the same reason as the range test above: either half can be the one the
      // statistics carry.
      auto const& outside_range = _tree.push(
        ast::operation{ast_operator::NULL_LOGICAL_OR,
                       _tree.push(ast::operation{ast_operator::NOT_EQUAL, vmin, vmax}),
                       _tree.push(ast::operation{ast_operator::NOT_EQUAL, vmax, literal})});
      // A null does not satisfy `!=` either, and an all-null chunk has no min or max to make this
      // test decisive, so the guard prunes it.
      return push_non_null_guard(col_index, outside_range);
    }
    case ast_operator::LESS: [[fallthrough]];
    case ast_operator::LESS_EQUAL: {
      auto const& vmin = _tree.push(ast::column_reference{col_index * stats_cols_per_column});
      // An all-null chunk has no min, leaving this test unknown, so the guard prunes it.
      return push_non_null_guard(col_index, _tree.push(ast::operation{op, vmin, literal}));
    }
    case ast_operator::GREATER: [[fallthrough]];
    case ast_operator::GREATER_EQUAL: {
      auto const& vmax = _tree.push(ast::column_reference{col_index * stats_cols_per_column + 1});
      // An all-null chunk has no max, leaving this test unknown, so the guard prunes it.
      return push_non_null_guard(col_index, _tree.push(ast::operation{op, vmax, literal}));
    }
    default: CUDF_UNREACHABLE("Non-prunable operator should not reach stats conversion");
  }
}

maybe_pruning_expr stats_expression_converter::build_unary(ast::ast_operator op,
                                                           ast::column_reference const& col_ref)
{
  using cudf::ast::ast_operator;

  if (op != ast_operator::IS_NULL) { return std::nullopt; }
  auto const& all_null =
    _tree.push(ast::column_reference{col_ref.get_column_index() * stats_cols_per_column + 2});
  return _tree.push(ast::operation{ast_operator::IDENTITY, all_null});
}

maybe_pruning_expr stats_expression_converter::build_negated_unary(
  ast::ast_operator op, ast::column_reference const& col_ref)
{
  using cudf::ast::ast_operator;

  if (op != ast_operator::IS_NULL) { return std::nullopt; }
  auto const& all_null =
    _tree.push(ast::column_reference{col_ref.get_column_index() * stats_cols_per_column + 2});
  return _tree.push(ast::operation{ast_operator::NOT, all_null});
}

maybe_pruning_expr stats_expression_converter::build_negated_comparison(
  ast::ast_operator op, ast::column_reference const& col_ref, ast::literal const& literal)
{
  // A comparison cannot be complemented when the column may hold a `NaN`: IEEE-754 makes every
  // ordered comparison with a NaN false, so `NOT(col < val)` is true where `col >= val` is not.
  if (cudf::is_floating_point(_output_dtypes[col_ref.get_column_index()])) { return std::nullopt; }

  auto const negated_op = transform_operator<operator_transform::NEGATE>(op);
  if (not negated_op.has_value()) { return std::nullopt; }
  return build_comparison(*negated_op, col_ref, literal);
}

maybe_pruning_expr stats_expression_converter::get_stats_expr() const { return _stats_expr; }

}  // namespace cudf::io::parquet::detail
