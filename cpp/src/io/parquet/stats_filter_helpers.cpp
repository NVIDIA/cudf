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

#include <optional>

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

stats_columns_collector::stats_columns_collector(ast::expression const& expr,
                                                 std::span<cudf::data_type const> output_dtypes)
  : parquet_expression_simplifier{output_dtypes}
{
  _columns_mask.resize(_output_dtypes.size(), false);
  std::ignore = simplify_expr(expr);
}

simplified_expression_opt stats_columns_collector::simplify_comparison(
  ast::ast_operator op, ast::column_reference const& col_ref, ast::literal const&)
{
  auto const col_index = col_ref.get_column_index();
  if (not is_prunable_comparison(op, _output_dtypes[col_index])) { return std::nullopt; }
  _columns_mask[col_index] = true;
  return placeholder_expr();
}

simplified_expression_opt stats_columns_collector::simplify_unary_op(
  ast::ast_operator op, ast::column_reference const& col_ref)
{
  if (op != ast::ast_operator::IS_NULL) { return std::nullopt; }
  _columns_mask[col_ref.get_column_index()] = true;
  return placeholder_expr();
}

simplified_expression_opt stats_columns_collector::simplify_negated_unary_op(
  ast::ast_operator op, ast::column_reference const& col_ref)
{
  if (op != ast::ast_operator::IS_NULL) { return std::nullopt; }
  _columns_mask[col_ref.get_column_index()] = true;

  return placeholder_expr();
}

simplified_expression_opt stats_columns_collector::simplify_negated_comparison(
  ast::ast_operator op, ast::column_reference const& col_ref, ast::literal const& literal)
{
  // A comparison cannot be complemented when the column may hold a `NaN`: IEEE-754 makes every
  // ordered comparison with a NaN false, so `NOT(col < val)` is true where `col >= val` is not.
  if (cudf::is_floating_point(_output_dtypes[col_ref.get_column_index()])) { return std::nullopt; }

  auto const negated_op = transform_operator<operator_transform::NEGATE>(op);
  if (not negated_op.has_value()) { return std::nullopt; }
  return simplify_comparison(*negated_op, col_ref, literal);
}

thrust::host_vector<bool> stats_columns_collector::get_stats_columns_mask() &&
{
  return std::move(_columns_mask);
}

stats_expression_converter::stats_expression_converter(
  ast::expression const& expr, std::span<cudf::data_type const> output_dtypes)
  : parquet_expression_simplifier{output_dtypes}
{
  _stats_expr = simplify_expr(expr);
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

simplified_expression_opt stats_expression_converter::simplify_comparison(
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

simplified_expression_opt stats_expression_converter::simplify_unary_op(
  ast::ast_operator op, ast::column_reference const& col_ref)
{
  using cudf::ast::ast_operator;

  if (op != ast_operator::IS_NULL) { return std::nullopt; }
  auto const& all_null =
    _tree.push(ast::column_reference{col_ref.get_column_index() * stats_cols_per_column + 2});
  return _tree.push(ast::operation{ast_operator::IDENTITY, all_null});
}

simplified_expression_opt stats_expression_converter::simplify_negated_unary_op(
  ast::ast_operator op, ast::column_reference const& col_ref)
{
  using cudf::ast::ast_operator;

  if (op != ast_operator::IS_NULL) { return std::nullopt; }
  auto const& all_null =
    _tree.push(ast::column_reference{col_ref.get_column_index() * stats_cols_per_column + 2});
  return _tree.push(ast::operation{ast_operator::NOT, all_null});
}

simplified_expression_opt stats_expression_converter::simplify_negated_comparison(
  ast::ast_operator op, ast::column_reference const& col_ref, ast::literal const& literal)
{
  // A comparison cannot be complemented when the column may hold a `NaN`: IEEE-754 makes every
  // ordered comparison with a NaN false, so `NOT(col < val)` is true where `col >= val` is not.
  if (cudf::is_floating_point(_output_dtypes[col_ref.get_column_index()])) { return std::nullopt; }

  auto const negated_op = transform_operator<operator_transform::NEGATE>(op);
  if (not negated_op.has_value()) { return std::nullopt; }
  return simplify_comparison(*negated_op, col_ref, literal);
}

simplified_expression_opt stats_expression_converter::get_stats_expr() const { return _stats_expr; }

}  // namespace cudf::io::parquet::detail
