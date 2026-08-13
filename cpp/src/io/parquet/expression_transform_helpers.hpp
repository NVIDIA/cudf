/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "column_path_helpers.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cudf::io::parquet::detail {

/**
 * @brief Classification of an AST expression operand
 */
enum class operand_kind : uint8_t { COLUMN_REF = 0, LITERAL = 1, EXPRESSION = 2 };

/**
 * @brief Extracted unary operand from an AST operation
 */
struct unary_operand {
  operand_kind operand_type;
  ast::column_reference const* col_ref;  ///< Non-null only when the operand is COLUMN_REF
};

/**
 * @brief Extracted binary operator and operands from an AST operation
 *
 * For `lit op col` expressions, the input non-commutative operator is inverted and the
 * operands are normalized to `col op lit` form.
 */
struct binary_operands {
  ast::ast_operator op;  ///< Input or inverted operator to normalize the `lit op col` expressions
  operand_kind lhs_type;
  operand_kind rhs_type;
  ast::column_reference const*
    col_ref;  ///< Reliable only when the expression is of the form `col op lit` or `lit op col`
  ast::literal const*
    literal;  ///< Reliable only when the expression is of the form `col op lit` or `lit op col`
};

/**
 * @brief Extracts the unary operand from a unary operation
 */
[[nodiscard]] unary_operand extract_unary_operand(ast::operation const& expr);

/**
 * @brief Decomposes a binary operation into classified parts.
 *
 * When the expression is of the form `lit op col`, the operator is inverted and the result
 * is normalized so that col_ref and literal are set as if the form were `col op lit`.
 */
[[nodiscard]] binary_operands extract_binary_operands(ast::operation const& expr);

/**
 * @brief Specifies how to transform a comparison operator
 */
enum class operator_transform : uint8_t {
  INVERT,  ///< Swap operand sides: `a < b` becomes `b > a`
  NEGATE   ///< Logical negation: `NOT(a < b)` becomes `a >= b`
};

/**
 * @brief Applies the specified transformation to an operator
 *
 * INVERT swaps operand order (e.g. LESS => GREATER) for normalizing `lit op col` to `col op lit`.
 * NEGATE returns the logical complement (e.g. LESS => GREATER_EQUAL) for handling NOT(col op lit).
 *
 * @tparam mode Transformation mode
 *
 * @param op Operator to transform
 * @return Transformed operator or std::nullopt. For INVERT mode, commutative and
 * untransformable operators are returned as is (no std::nullopt)
 */
template <operator_transform mode>
[[nodiscard]] inline std::optional<ast::ast_operator> transform_operator(ast::ast_operator op)
{
  if constexpr (mode == operator_transform::INVERT) {
    switch (op) {
      case ast::ast_operator::LESS: return ast::ast_operator::GREATER;
      case ast::ast_operator::GREATER: return ast::ast_operator::LESS;
      case ast::ast_operator::LESS_EQUAL: return ast::ast_operator::GREATER_EQUAL;
      case ast::ast_operator::GREATER_EQUAL: return ast::ast_operator::LESS_EQUAL;
      default: return std::make_optional(op);
    }
  } else {
    // mode == NEGATE
    switch (op) {
      case ast::ast_operator::LESS: return ast::ast_operator::GREATER_EQUAL;
      case ast::ast_operator::GREATER: return ast::ast_operator::LESS_EQUAL;
      case ast::ast_operator::LESS_EQUAL: return ast::ast_operator::GREATER;
      case ast::ast_operator::GREATER_EQUAL: return ast::ast_operator::LESS;
      case ast::ast_operator::EQUAL: return ast::ast_operator::NOT_EQUAL;
      case ast::ast_operator::NOT_EQUAL: return ast::ast_operator::EQUAL;
      default: return std::nullopt;
    }
  }
}

/**
 * @brief Handle unary operation transform for membership-based row group filters. i.e., bloom
 * filter and dictionary page filter.
 *
 * @tparam VisitorType Type of the AST visitor that implements accept()
 * @tparam VisitOperandsFn Callable matching `(host_span<reference_wrapper<expr>>) ->
 * vector<reference_wrapper<expr>>`
 *
 * @param expr Unary operation to transform
 * @param expr_tree The AST tree to push transformed expressions into
 * @param always_true Reference to the always_true sentinel literal
 * @param visitor The visitor used to accept column references
 * @param visit_operands_fn Callable to visit operands and return the transformed operands
 * @return Transformed expression or _always_true if the operation cannot be evaluated
 */
template <typename VisitorType, typename VisitOperandsFn>
[[nodiscard]] inline std::reference_wrapper<ast::expression const> apply_unary_membership_transform(
  ast::operation const& expr,
  ast::tree& expr_tree,
  std::reference_wrapper<ast::expression const> const always_true,
  VisitorType& visitor,
  VisitOperandsFn&& visit_operands_fn)
{
  auto const [kind, col_ref] = extract_unary_operand(expr);

  // For `op col` form, push the `_always_true` expression
  if (kind == operand_kind::COLUMN_REF) {
    col_ref->accept(visitor);
    expr_tree.push(ast::operation{ast::ast_operator::IDENTITY, always_true});
    return always_true;
  }
  // For `op expr` form, visit operands and push expression
  else {
    auto new_operands = visit_operands_fn(expr.get_operands());
    if (&new_operands.front().get() == &always_true.get()) {
      // Pass through the _always_true child operand as is
      expr_tree.push(ast::operation{ast::ast_operator::IDENTITY, expr_tree.back()});
      return always_true;
    } else {
      auto const input_op = expr.get_operator();
      expr_tree.push(ast::operation{input_op, new_operands.front()});
      return expr_tree.back();
    }
  }
}

/**
 * @brief Collects column names from the expression ignoring the `skip_names`
 */
class names_from_expression : public ast::detail::expression_transformer {
 public:
  names_from_expression() = default;

  names_from_expression(std::optional<std::reference_wrapper<ast::expression const>> expr,
                        std::vector<std::string> const& skip_names,
                        cudf::io::parquet_reader_options const& options,
                        std::vector<SchemaElement> const& schema_tree);

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Returns the column names in AST.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::vector<std::string> to_vector() &&;

 private:
  void visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  std::unordered_map<cudf::size_type, std::string> _column_indices_to_names;
  std::unordered_set<std::string> _column_names;
  column_path_set _skip_names;
};

/**
 * @brief Converts named columns to index reference columns
 */
class named_to_reference_converter : public ast::detail::expression_transformer {
 public:
  named_to_reference_converter() = default;

  named_to_reference_converter(std::optional<std::reference_wrapper<ast::expression const>> expr,
                               table_metadata const& metadata,
                               bool case_sensitive_names);

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Returns the converted AST expression
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::optional<std::reference_wrapper<ast::expression const>> get_converted_expr()
    const
  {
    return _converted_expr;
  }

 protected:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  column_path_map<size_type> _column_name_to_index;
  std::optional<std::reference_wrapper<ast::expression const>> _converted_expr;
  // Using std::list or std::deque to avoid reference invalidation
  std::list<ast::column_reference> _col_ref;
  std::list<ast::operation> _operators;
};

/**
 * @brief Collects lists of equality predicate literals in the AST expression, one list per input
 * table column. This is used in row group filtering based on bloom filters.
 */
class equality_literals_collector : public ast::detail::expression_transformer {
 public:
  equality_literals_collector() = default;

  equality_literals_collector(ast::expression const& expr,
                              cudf::host_span<cudf::data_type const> output_dtypes,
                              cudf::host_span<cudf::size_type const> output_column_schemas = {},
                              cudf::host_span<SchemaElement const> schema_tree             = {});

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Vectors of equality literals in the AST expression, one per input table column
   *
   * @return Vectors of equality literals, one per input table column
   */
  [[nodiscard]] std::vector<std::vector<ast::literal*>> get_literals() &&;

 protected:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  cudf::host_span<cudf::data_type const> _output_dtypes;
  std::vector<std::vector<ast::literal*>> _literals;

 private:
  cudf::host_span<cudf::size_type const> _output_column_schemas;
  cudf::host_span<SchemaElement const> _schema_tree;
};

/**
 * @brief Offsets every column referencein an expression by the specified value
 *
 */
class offset_column_references : public named_to_reference_converter {
 public:
  offset_column_references(std::optional<std::reference_wrapper<ast::expression const>> expr,
                           size_type offset);

  // Use `visit` overloads from named_to_reference_converter
  using named_to_reference_converter::visit;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

 private:
  size_type _offset{0};
};

/**
 * @brief Maps indices of (all or selected) columns to their names
 *
 * @param options Parquet reader options
 * @param schema_tree Parquet schema tree
 *
 * @return Map of column indices to their names
 */
[[nodiscard]] std::unordered_map<cudf::size_type, std::string> map_column_indices_to_names(
  cudf::io::parquet_reader_options const& options,
  std::span<SchemaElement const> schema_tree,
  bool case_sensitive_names);

/**
 * @brief Get the column names in expression object
 *
 * @param expr The optional expression object to get the column names from
 * @param skip_names The names of column names to skip in returned column names
 * @param options Reader options
 * @param schema_tree The schema tree describing the file structure
 * @return The column names present in expression object except the skip_names
 */
[[nodiscard]] std::vector<std::string> get_column_names_in_expression(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  std::vector<std::string> const& skip_names,
  cudf::io::parquet_reader_options const& options,
  std::vector<SchemaElement> const& schema_tree);

/**
 * @brief Filter table using the provided (StatsAST or BloomfilterAST) expression and
 * collect filtered row group indices
 *
 * @param ast_table Table of stats or bloom filter membership columns
 * @param ast_expr StatsAST or BloomfilterAST expression to filter with
 * @param input_row_group_indices Lists of input row groups to read, one per source
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Collected filtered row group indices, one vector per source, if any. A std::nullopt if
 * all row groups are required or if the computed predicate is all nulls
 */
[[nodiscard]] std::optional<std::vector<std::vector<size_type>>> collect_filtered_row_group_indices(
  cudf::table_view ast_table,
  std::reference_wrapper<ast::expression const> ast_expr,
  host_span<std::vector<size_type> const> input_row_group_indices,
  rmm::cuda_stream_view stream);

}  // namespace cudf::io::parquet::detail
