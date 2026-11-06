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
 * @brief Returns whether a comparison operator can prune row groups or pages via statistics
 *
 * @param op The comparison operator
 * @return True if the comparison can be used to prune row groups or pages
 */
[[nodiscard]] bool is_prunable_comparison(ast::ast_operator op)
{
  using cudf::ast::ast_operator;
  switch (op) {
    case ast_operator::EQUAL: [[fallthrough]];
    case ast_operator::NOT_EQUAL: [[fallthrough]];
    case ast_operator::LESS: [[fallthrough]];
    case ast_operator::LESS_EQUAL: [[fallthrough]];
    case ast_operator::GREATER: [[fallthrough]];
    case ast_operator::GREATER_EQUAL: return true;
    default: return false;
  }
}

}  // namespace

stats_columns_collector::stats_columns_collector(ast::expression const& expr,
                                                 std::span<cudf::data_type const> output_dtypes)
  : parquet_expression_simplifier{output_dtypes}
{
  // Return an empty mask if statistics cannot prune anything with this filter
  auto const simplified_expr = simplify_expr(expr);
  if (not simplified_expr.has_value()) { return; }

  _columns_mask.resize(_output_dtypes.size(), false);

  // Walk the simplified tree and collect columns references
  collect_surviving_predicates(simplified_expr.value().get());
}

void stats_columns_collector::collect_surviving_predicates(ast::expression const& simplified_expr)
{
  if (auto const* col_ref = dynamic_cast<ast::column_reference const*>(&simplified_expr);
      col_ref != nullptr) {
    _columns_mask[col_ref->get_column_index()] = true;
    return;
  }
  for (auto const& operand : dynamic_cast<ast::operation const&>(simplified_expr).get_operands()) {
    collect_surviving_predicates(operand.get());
  }
}

simplified_expression_opt stats_columns_collector::simplify_comparison(
  ast::ast_operator op, ast::column_reference const& col_ref, ast::literal const&)
{
  if (not is_prunable_comparison(op)) { return std::nullopt; }
  return col_ref;
}

simplified_expression_opt stats_columns_collector::simplify_unary_op(
  ast::ast_operator op, ast::column_reference const& col_ref)
{
  if (op != ast::ast_operator::IS_NULL) { return std::nullopt; }
  return col_ref;
}

simplified_expression_opt stats_columns_collector::simplify_negated_unary_op(
  ast::ast_operator op, ast::column_reference const& col_ref)
{
  if (op != ast::ast_operator::IS_NULL) { return std::nullopt; }
  return col_ref;
}

simplified_expression_opt stats_columns_collector::simplify_negated_comparison(
  ast::ast_operator op, ast::column_reference const& col_ref, ast::literal const& literal)
{
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

ast::expression const& stats_expression_converter::push_nan_guard(size_type col_index,
                                                                  ast::expression const& literal,
                                                                  ast::expression const& stats_expr)
{
  using cudf::ast::ast_operator;

  if (not cudf::is_floating_point(_output_dtypes[col_index])) { return stats_expr; }

  // A `NaN` literal satisfies the predicate for every non-null value, which `lit != lit` detects
  auto const& is_nan_lit = _tree.push(ast::operation{ast_operator::NOT_EQUAL, literal, literal});
  auto const& has_nan    = _tree.push(ast::column_reference{col_index * stats_cols_per_column + 3});
  auto const& nan_matches =
    _tree.push(ast::operation{ast_operator::NULL_LOGICAL_OR, is_nan_lit, has_nan});

  // Null-aware so that a chunk known to be NaN-free is decided by `stats_expr` alone, while one
  // whose NaN count is unknown is kept unless `stats_expr` is true anyway.
  return _tree.push(ast::operation{ast_operator::NULL_LOGICAL_OR, nan_matches, stats_expr});
}

simplified_expression_opt stats_expression_converter::simplify_comparison(
  ast::ast_operator op, ast::column_reference const& col_ref, ast::literal const& literal_ref)
{
  if (not is_prunable_comparison(op)) { return std::nullopt; }
  return push_comparison(
    op, col_ref.get_column_index(), literal_ref, op == ast::ast_operator::NOT_EQUAL);
}

ast::expression const& stats_expression_converter::push_comparison(ast::ast_operator op,
                                                                   size_type col_index,
                                                                   ast::literal const& literal_ref,
                                                                   bool nan_satisfies)
{
  using cudf::ast::ast_operator;

  auto const& literal    = _tree.push(literal_ref);
  auto const& stats_expr = [&]() -> ast::expression const& {
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
        // The two halves are separately optional in the statistics, so they are combined
        // null-aware to keep whichever one is present decisive.
        return _tree.push(
          ast::operation{ast_operator::NULL_LOGICAL_AND,
                         _tree.push(ast::operation{ast_operator::GREATER_EQUAL, vmax, literal}),
                         _tree.push(ast::operation{ast_operator::LESS_EQUAL, vmin, literal})});
      }
      case ast_operator::NOT_EQUAL: {
        auto const& vmin = _tree.push(ast::column_reference{col_index * stats_cols_per_column});
        auto const& vmax = _tree.push(ast::column_reference{col_index * stats_cols_per_column + 1});
        // Null-aware for the same reason as the range test above: either half can be the one the
        // statistics carry.
        return _tree.push(
          ast::operation{ast_operator::NULL_LOGICAL_OR,
                         _tree.push(ast::operation{ast_operator::NOT_EQUAL, vmin, vmax}),
                         _tree.push(ast::operation{ast_operator::NOT_EQUAL, vmax, literal})});
      }
      case ast_operator::LESS: [[fallthrough]];
      case ast_operator::LESS_EQUAL: {
        auto const& vmin = _tree.push(ast::column_reference{col_index * stats_cols_per_column});
        return _tree.push(ast::operation{op, vmin, literal});
      }
      case ast_operator::GREATER: [[fallthrough]];
      case ast_operator::GREATER_EQUAL: {
        auto const& vmax = _tree.push(ast::column_reference{col_index * stats_cols_per_column + 1});
        return _tree.push(ast::operation{op, vmax, literal});
      }
      default: CUDF_UNREACHABLE("Non-prunable operator should not reach stats conversion");
    }
  }();

  // A `NaN` satisfies `!=` and negated ordered comparisons but is excluded from the min and max, so
  // a chunk that may hold one is kept.
  auto const& nan_guarded =
    nan_satisfies ? push_nan_guard(col_index, literal, stats_expr) : stats_expr;
  // A null satisfies no comparison, and an all-null chunk has no min or max to make the test
  // decisive, so the non-null guard prunes it.
  return push_non_null_guard(col_index, nan_guarded);
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
  auto const negated_op = transform_operator<operator_transform::NEGATE>(op);
  if (not negated_op.has_value() or not is_prunable_comparison(*negated_op)) {
    return std::nullopt;
  }
  // IEEE-754 makes every ordered comparison with a `NaN` false, so a `NaN` value or literal
  // satisfies `NOT(col < val)` but not `col >= val`. Equality complements exactly, so only
  // `NOT(col == val)`, i.e. `col != val`, is satisfied by a `NaN` among those.
  return push_comparison(
    *negated_op, col_ref.get_column_index(), literal, *negated_op != ast::ast_operator::EQUAL);
}

simplified_expression_opt stats_expression_converter::get_stats_expr() const { return _stats_expr; }

}  // namespace cudf::io::parquet::detail
