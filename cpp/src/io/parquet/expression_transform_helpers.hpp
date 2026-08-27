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

#include <cuda/stream>

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
  requires(mode == operator_transform::INVERT or mode == operator_transform::NEGATE)
[[nodiscard]] std::optional<ast::ast_operator> transform_operator(ast::ast_operator op);

/**
 * @brief Returns the De Morgan operator for the given operator
 *
 * @param op Operator to transform
 * @return De Morgan operator or std::nullopt
 */
[[nodiscard]] std::optional<ast::ast_operator> de_morgan_operator(ast::ast_operator op);

/**
 * @brief Returns whether an expression is boolean-valued
 *
 * Comparison, logical and null-checking operations always yield `BOOL8`, and a literal knows its
 * own type. A column reference cannot be typed here, as the filter is normalized before the output
 * data types are known, so it is reported as not boolean.
 *
 * @param expr Expression to classify
 * @return Whether the expression yields `BOOL8`
 */
[[nodiscard]] bool is_boolean_valued(ast::expression const& expr);

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
  std::unordered_map<cudf::size_type, std::string> _column_indices_to_names;
  std::unordered_set<std::string> _column_names;
  column_path_set _skip_names;
};

/**
 * @brief Converts named columns to index reference columns and rewrites the expression into
 * negation normal form, pushing logical negations down to the leaves.
 */
class parquet_filter_normalizer : public ast::detail::expression_transformer {
 public:
  parquet_filter_normalizer() = default;

  parquet_filter_normalizer(std::optional<std::reference_wrapper<ast::expression const>> expr,
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
   * @return Converted expression, if an input expression was provided
   */
  [[nodiscard]] std::optional<std::reference_wrapper<ast::expression const>> get_converted_expr()
    const;

 protected:
  /**
   * @brief Rewrites `NOT(operand)` into an equivalent expression with the negation pushed into
   * `operand`'s own operands
   *
   * Only rewrites that are exact in every case cudf's AST evaluates are applied, as the converted
   * expression also filters the decoded rows:
   *
   * - `NOT(NOT(x))` => `x`, but only when `x` is provably boolean. `NOT` yields `bool` for any
   *   operand, so for a non-boolean `x` the double negation means `x != 0` rather than `x`
   * - De Morgan forms: `NOT(a AND b)` => `NOT(a) OR NOT(b)` for both the null-propagating
   *   (`LOGICAL_*`) and the Kleene (`NULL_LOGICAL_*`) operators
   * - `NOT(a == b)` => `a != b` and vice versa
   * - `NOT(IS_NULL(x))` and `NOT(NULL_EQUAL(a, b))` => left alone as they have no complement
   * - Ordering comparisons (`<`, `>`, `<=`, `>=`) are *not* complemented as IEEE-754 makes every
   *   comparison with a `NaN` false, so `NOT(a < b)` is true while `a >= b` is not
   *
   * @param operand The operand of the `NOT` operation to rewrite
   * @return The rewritten expression, or std::nullopt if no exact rewrite exists
   */
  [[nodiscard]] std::optional<std::reference_wrapper<ast::expression const>> push_down_negation(
    ast::expression const& operand);

  /**
   * @brief Returns the converted negation of `operand`, pushing the negation down if possible and
   * otherwise wrapping the converted operand in a `NOT`
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> negate(
    ast::expression const& operand);

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
class offset_column_references : public ast::detail::expression_transformer {
 public:
  offset_column_references(std::optional<std::reference_wrapper<ast::expression const>> expr,
                           size_type offset);

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

  /**
   * @brief Returns the converted AST expression
   *
   * @return Converted expression, if an input expression was provided
   */
  [[nodiscard]] std::optional<std::reference_wrapper<ast::expression const>> get_converted_expr()
    const;

 private:
  ast::tree _tree;
  std::optional<std::reference_wrapper<ast::expression const>> _converted_expr;
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
  cuda::stream_ref stream);

}  // namespace cudf::io::parquet::detail
