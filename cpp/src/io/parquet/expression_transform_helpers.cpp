/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "expression_transform_helpers.hpp"

#include "column_path_helpers.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <string>
#include <utility>

namespace cudf::io::parquet::detail {

namespace {

/**
 * @brief Classifies an AST operand as column reference, literal, or expression
 *
 * @param operand AST operand to classify
 * @return Classified operand kind
 */
[[nodiscard]] operand_kind classify_operand(ast::expression const& operand)
{
  if (dynamic_cast<ast::column_reference const*>(&operand) != nullptr) {
    return operand_kind::COLUMN_REF;
  } else if (dynamic_cast<ast::literal const*>(&operand) != nullptr) {
    return operand_kind::LITERAL;
  } else if (dynamic_cast<ast::column_name_reference const*>(&operand) != nullptr) {
    CUDF_UNREACHABLE("Column name references are not supported in Parquet expression transformers");
  } else {
    return operand_kind::EXPRESSION;
  }
}

}  // namespace

template <operator_transform mode>
std::optional<ast::ast_operator> transform_operator(ast::ast_operator op)
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

template std::optional<ast::ast_operator> transform_operator<operator_transform::INVERT>(
  ast::ast_operator op);
template std::optional<ast::ast_operator> transform_operator<operator_transform::NEGATE>(
  ast::ast_operator op);

std::optional<ast::ast_operator> de_morgan_operator(ast::ast_operator op)
{
  switch (op) {
    case ast::ast_operator::LOGICAL_AND: return ast::ast_operator::LOGICAL_OR;
    case ast::ast_operator::LOGICAL_OR: return ast::ast_operator::LOGICAL_AND;
    case ast::ast_operator::NULL_LOGICAL_AND: return ast::ast_operator::NULL_LOGICAL_OR;
    case ast::ast_operator::NULL_LOGICAL_OR: return ast::ast_operator::NULL_LOGICAL_AND;
    default: return std::nullopt;
  }
}

bool is_boolean_valued(ast::expression const& expr)
{
  using cudf::ast::ast_operator;

  // A literal knows its own type
  if (auto const* literal = dynamic_cast<ast::literal const*>(&expr); literal != nullptr) {
    return literal->get_data_type().id() == cudf::type_id::BOOL8;
  }

  // A column reference cannot be typed here - the normalizer runs before the output data types are
  // known - so report it as not provably boolean
  auto const* operation = dynamic_cast<ast::operation const*>(&expr);
  if (operation == nullptr) { return false; }

  switch (operation->get_operator()) {
    case ast_operator::EQUAL: [[fallthrough]];
    case ast_operator::NOT_EQUAL: [[fallthrough]];
    case ast_operator::NULL_EQUAL: [[fallthrough]];
    case ast_operator::LESS: [[fallthrough]];
    case ast_operator::GREATER: [[fallthrough]];
    case ast_operator::LESS_EQUAL: [[fallthrough]];
    case ast_operator::GREATER_EQUAL: [[fallthrough]];
    case ast_operator::LOGICAL_AND: [[fallthrough]];
    case ast_operator::LOGICAL_OR: [[fallthrough]];
    case ast_operator::NULL_LOGICAL_AND: [[fallthrough]];
    case ast_operator::NULL_LOGICAL_OR: [[fallthrough]];
    case ast_operator::IS_NULL: [[fallthrough]];
    case ast_operator::NOT: return true;
    default: return false;
  }
}

unary_operand extract_unary_operand(ast::operation const& expr)
{
  auto const& operands = expr.get_operands();
  CUDF_EXPECTS(operands.size() == 1, "Expected single operand for a unary operation");
  auto const& operand = operands[0].get();

  return {.operand_type = classify_operand(operand),
          .col_ref      = dynamic_cast<ast::column_reference const*>(&operand)};
}

binary_operands extract_binary_operands(ast::operation const& expr)
{
  auto const operands = expr.get_operands();
  auto const op       = expr.get_operator();
  CUDF_EXPECTS(operands.size() == 2, "Expected two operands for a binary operation");

  auto const& lhs = operands[0].get();
  auto const& rhs = operands[1].get();

  auto const lhs_kind = classify_operand(lhs);
  auto const rhs_kind = classify_operand(rhs);

  // Normalize `lit op col` to `col op lit` by inverting the operator
  if (lhs_kind == operand_kind::LITERAL and rhs_kind == operand_kind::COLUMN_REF) {
    return {.op       = transform_operator<operator_transform::INVERT>(op).value(),
            .lhs_type = rhs_kind,
            .rhs_type = lhs_kind,
            .col_ref  = dynamic_cast<ast::column_reference const*>(&rhs),
            .literal  = dynamic_cast<ast::literal const*>(&lhs)};
  } else {
    return {.op       = op,
            .lhs_type = lhs_kind,
            .rhs_type = rhs_kind,
            .col_ref  = (lhs_kind == operand_kind::COLUMN_REF)
                          ? dynamic_cast<ast::column_reference const*>(&lhs)
                          : dynamic_cast<ast::column_reference const*>(&rhs),
            .literal  = (rhs_kind == operand_kind::LITERAL)
                          ? dynamic_cast<ast::literal const*>(&rhs)
                          : dynamic_cast<ast::literal const*>(&lhs)};
  }
}

parquet_filter_normalizer::parquet_filter_normalizer(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  table_metadata const& metadata,
  bool case_sensitive_names)
  : _column_name_to_index(make_column_path_map<size_type>(case_sensitive_names))
{
  if (!expr.has_value()) { return; }
  // create map for column name.
  std::transform(metadata.schema_info.cbegin(),
                 metadata.schema_info.cend(),
                 cuda::counting_iterator<std::size_t>{0},
                 std::inserter(_column_name_to_index, _column_name_to_index.end()),
                 [](auto const& sch, auto index) { return std::make_pair(sch.name, index); });

  expr.value().get().accept(*this);
}

std::reference_wrapper<ast::expression const> parquet_filter_normalizer::visit(
  ast::literal const& expr)
{
  _converted_expr = std::reference_wrapper<ast::expression const>(expr);
  return expr;
}

std::reference_wrapper<ast::expression const> parquet_filter_normalizer::visit(
  ast::column_reference const& expr)
{
  _converted_expr = std::reference_wrapper<ast::expression const>(expr);
  return expr;
}

std::reference_wrapper<ast::expression const> parquet_filter_normalizer::visit(
  ast::column_name_reference const& expr)
{
  // check if column name is in metadata
  auto col_index_it = _column_name_to_index.find(expr.get_column_name());
  CUDF_EXPECTS(col_index_it != _column_name_to_index.end(),
               "Column name in the filter expression not found in the "
               "metadata of selected columns. Note that only top-level "
               "columns except structs and lists are supported in "
               "Parquet filter expression",
               std::invalid_argument);
  auto col_index = col_index_it->second;
  _col_ref.emplace_back(col_index);
  _converted_expr = std::reference_wrapper<ast::expression const>(_col_ref.back());
  return std::reference_wrapper<ast::expression const>(_col_ref.back());
}

std::optional<std::reference_wrapper<ast::expression const>>
parquet_filter_normalizer::push_down_negation(ast::expression const& operand)
{
  using cudf::ast::ast_operator;

  auto const* child_operation = dynamic_cast<ast::operation const*>(&operand);
  if (child_operation == nullptr) { return std::nullopt; }

  auto const op       = child_operation->get_operator();
  auto const operands = child_operation->get_operands();

  // `NOT(NOT(x))` is `x`, including when `x` is null, but only when `x` is boolean.
  if (op == ast_operator::NOT and is_boolean_valued(operands.front().get())) {
    return operands.front().get().accept(*this);
  }

  // De Morgan's laws compute exact equivalences for both the null-propagating (`LOGICAL_*`) and the
  // Kleene (`NULL_LOGICAL_*`) operators, including when an operand is null
  if (auto const de_morgan_op = de_morgan_operator(op); de_morgan_op.has_value()) {
    // Negate both operands before emplacing the parent, as negating them appends to `_operators`
    auto const lhs = negate(operands.front().get());
    auto const rhs = negate(operands.back().get());
    _operators.emplace_back(de_morgan_op.value(), lhs, rhs);
    return std::reference_wrapper<ast::expression const>(_operators.back());
  }

  // Complement equality operators (exact for floats too) as `NaN == x` is false and `NaN != x` is
  // true. Ordering comparisons (every comparison with `NaN` is false) are excluded.
  if (op == ast_operator::EQUAL or op == ast_operator::NOT_EQUAL) {
    auto const negation = transform_operator<operator_transform::NEGATE>(op).value();
    auto new_operands   = visit_operands(operands);
    _operators.emplace_back(negation, new_operands.front(), new_operands.back());
    return std::reference_wrapper<ast::expression const>(_operators.back());
  }

  return std::nullopt;
}

std::reference_wrapper<ast::expression const> parquet_filter_normalizer::negate(
  ast::expression const& operand)
{
  // Push negation down to operands
  if (auto const negated = push_down_negation(operand); negated.has_value()) {
    return negated.value();
  }
  // No exact rewrite available, so convert the operand and wrap it back in a `NOT`
  auto const new_operand = operand.accept(*this);
  _operators.emplace_back(ast::ast_operator::NOT, new_operand);
  return std::reference_wrapper<ast::expression const>(_operators.back());
}

std::reference_wrapper<ast::expression const> parquet_filter_normalizer::visit(
  ast::operation const& expr)
{
  auto const operands       = expr.get_operands();
  auto op                   = expr.get_operator();
  auto const operator_arity = cudf::ast::detail::ast_operator_arity(op);

  // Push down negation to leaves so that downstream transformers don't have to handle `NOT` over
  // rewritten operands
  if (op == ast::ast_operator::NOT) {
    if (auto const negated = push_down_negation(operands.front().get()); negated.has_value()) {
      _converted_expr = negated.value();
      return negated.value();
    }
  }

  auto new_operands = visit_operands(operands);
  if (operator_arity == 2) {
    _operators.emplace_back(op, new_operands.front(), new_operands.back());
  } else if (operator_arity == 1) {
    _operators.emplace_back(op, new_operands.front());
  }
  _converted_expr = std::reference_wrapper<ast::expression const>(_operators.back());
  return std::reference_wrapper<ast::expression const>(_operators.back());
}

names_from_expression::names_from_expression(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  std::vector<std::string> const& skip_names,
  cudf::io::parquet_reader_options const& options,
  std::vector<SchemaElement> const& schema_tree)
  : _skip_names(make_column_path_set(options.is_enabled_case_sensitive_names()))
{
  _skip_names.insert(skip_names.cbegin(), skip_names.cend());

  if (!expr.has_value()) { return; }

  // Map original column names to indices
  auto constexpr case_sensitive_names = true;
  _column_indices_to_names =
    map_column_indices_to_names(options, schema_tree, case_sensitive_names);

  expr.value().get().accept(*this);
}

std::reference_wrapper<ast::expression const> names_from_expression::visit(ast::literal const& expr)
{
  return expr;
}

std::reference_wrapper<ast::expression const> names_from_expression::visit(
  ast::column_reference const& expr)
{
  // Map the column index to its name
  auto const col_name_iter = _column_indices_to_names.find(expr.get_column_index());
  CUDF_EXPECTS(
    col_name_iter != _column_indices_to_names.end(),
    "Column index in the filter expression not found in the column indices to names map. Note that "
    "only top-level columns except structs and lists are supported in "
    "Parquet filter expression",
    std::invalid_argument);
  auto const col_name = col_name_iter->second;
  // If the normalized column name is not in the skip_names, add it
  if (_skip_names.count(col_name) == 0) { _column_names.insert(col_name); }
  return expr;
}

std::reference_wrapper<ast::expression const> names_from_expression::visit(
  ast::column_name_reference const& expr)
{
  // collect column names
  auto const col_name = expr.get_column_name();
  // If the normalized column name is not in the skip_names, add it
  if (_skip_names.count(col_name) == 0) { _column_names.insert(col_name); }
  return expr;
}

std::reference_wrapper<ast::expression const> names_from_expression::visit(
  ast::operation const& expr)
{
  std::ignore = visit_operands(expr.get_operands());
  return expr;
}

std::vector<std::string> names_from_expression::to_vector() &&
{
  return {std::make_move_iterator(_column_names.begin()),
          std::make_move_iterator(_column_names.end())};
}

offset_column_references::offset_column_references(
  std::optional<std::reference_wrapper<ast::expression const>> expr, size_type offset)
  : _offset{offset}
{
  if (not expr.has_value()) { return; }
  if (offset == 0) {
    _converted_expr = expr;
    return;
  }
  _converted_expr = expr.value().get().accept(*this);
}

std::optional<std::reference_wrapper<ast::expression const>>
parquet_filter_normalizer::get_converted_expr() const
{
  return _converted_expr;
}

std::optional<std::reference_wrapper<ast::expression const>>
offset_column_references::get_converted_expr() const
{
  return _converted_expr;
}

std::reference_wrapper<ast::expression const> offset_column_references::visit(
  ast::literal const& expr)
{
  return expr;
}

std::reference_wrapper<ast::expression const> offset_column_references::visit(
  ast::column_reference const& expr)
{
  return _tree.push(
    ast::column_reference{expr.get_column_index() + _offset, expr.get_table_source()});
}

std::reference_wrapper<ast::expression const> offset_column_references::visit(
  ast::operation const& expr)
{
  auto const new_operands = visit_operands(expr.get_operands());
  auto const arity        = cudf::ast::detail::ast_operator_arity(expr.get_operator());
  if (arity == 1) { return _tree.push(ast::operation{expr.get_operator(), new_operands.front()}); }
  return _tree.push(ast::operation{expr.get_operator(), new_operands.front(), new_operands.back()});
}

std::reference_wrapper<ast::expression const> offset_column_references::visit(
  ast::column_name_reference const&)
{
  CUDF_FAIL("Column name references are not supported in column reference offsetter");
}

[[nodiscard]] std::unordered_map<cudf::size_type, std::string> map_column_indices_to_names(
  cudf::io::parquet_reader_options const& options,
  std::span<SchemaElement const> schema_tree,
  bool case_sensitive_names)
{
  std::unordered_map<cudf::size_type, std::string> column_indices_to_names;

  auto const& selected_column_names     = options.get_column_names();
  auto const& selected_column_indices   = options.get_column_indices();
  auto const& selected_column_field_ids = options.get_column_field_ids();

  // Map counting indices to the selected column by names
  if (selected_column_names.has_value()) {
    std::transform(selected_column_names->begin(),
                   selected_column_names->end(),
                   cuda::counting_iterator<cudf::size_type>{0},
                   std::inserter(column_indices_to_names, column_indices_to_names.end()),
                   [case_sensitive_names](auto const& col_name, auto const col_index) {
                     return std::make_pair(col_index,
                                           normalize_column_path(col_name, case_sensitive_names));
                   });
  }
  // Map selected top-level column indices to their names from the schema tree
  else if (selected_column_indices.has_value()) {
    auto const& root = schema_tree.front();

    std::transform(
      selected_column_indices->begin(),
      selected_column_indices->end(),
      cuda::counting_iterator<cudf::size_type>{0},
      std::inserter(column_indices_to_names, column_indices_to_names.end()),
      [&](auto selected_col_idx, auto const mapped_col_idx) {
        CUDF_EXPECTS(
          selected_col_idx >= 0 and std::cmp_less(selected_col_idx, root.children_idx.size()),
          "Encountered an invalid col index in the top-level column selection",
          std::invalid_argument);
        auto const schema_idx = root.children_idx[selected_col_idx];
        return std::make_pair(
          mapped_col_idx,
          normalize_column_path(schema_tree[schema_idx].name, case_sensitive_names));
      });
  }
  // Map selected field ids to column paths from the schema tree
  else if (selected_column_field_ids.has_value()) {
    std::transform(
      selected_column_field_ids->begin(),
      selected_column_field_ids->end(),
      cuda::counting_iterator<cudf::size_type>{0},
      std::inserter(column_indices_to_names, column_indices_to_names.end()),
      [&](auto const& field_id, auto const mapped_col_idx) {
        auto const schema_iter =
          std::find_if(schema_tree.begin() + 1, schema_tree.end(), [field_id](auto const& schema) {
            return schema.field_id.has_value() and schema.field_id.value() == field_id;
          });
        CUDF_EXPECTS(schema_iter != schema_tree.end(),
                     "Encountered a non-existent Parquet field ID in selected columns",
                     std::invalid_argument);
        auto const schema_idx = static_cast<int>(std::distance(schema_tree.begin(), schema_iter));
        return std::make_pair(mapped_col_idx,
                              normalize_column_path(column_path_from_index(schema_tree, schema_idx),
                                                    case_sensitive_names));
      });
  }
  // Map all top-level column indices to their names from the schema tree
  else {
    auto const& root = schema_tree.front();

    std::for_each(
      cuda::counting_iterator<int32_t>{0},
      cuda::counting_iterator{static_cast<int32_t>(root.children_idx.size())},
      [&](auto col_idx) {
        auto const schema_idx = root.children_idx[col_idx];
        column_indices_to_names.insert(
          {col_idx, normalize_column_path(schema_tree[schema_idx].name, case_sensitive_names)});
      });
  }

  return column_indices_to_names;
}

[[nodiscard]] std::vector<std::string> get_column_names_in_expression(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  std::vector<std::string> const& skip_names,
  cudf::io::parquet_reader_options const& options,
  std::vector<SchemaElement> const& schema_tree)
{
  return names_from_expression(expr, skip_names, options, schema_tree).to_vector();
}

std::optional<std::vector<std::vector<size_type>>> collect_filtered_row_group_indices(
  cudf::table_view table,
  std::reference_wrapper<ast::expression const> ast_expr,
  host_span<std::vector<size_type> const> input_row_group_indices,
  cuda::stream_ref stream)
{
  // Filter the input table using AST expression
  auto predicate_col = cudf::detail::compute_column(
    table, ast_expr.get(), stream, cudf::get_current_device_resource_ref());
  auto predicate = predicate_col->view();
  CUDF_EXPECTS(predicate.type().id() == cudf::type_id::BOOL8,
               "Filter expression must return a boolean column");

  auto host_bitmask = [&] {
    std::size_t const num_bitmasks = num_bitmask_words(predicate.size());
    if (predicate.nullable()) {
      return cudf::detail::make_pinned_vector(
        cudf::device_span<bitmask_type const>{predicate.null_mask(), num_bitmasks}, stream);
    } else {
      auto bitmask = cudf::detail::make_pinned_vector<bitmask_type>(num_bitmasks, stream);
      std::fill(bitmask.begin(), bitmask.end(), ~bitmask_type{0});
      return bitmask;
    }
  }();

  auto validity_it = cudf::detail::make_counting_transform_iterator(
    0, [bitmask = host_bitmask.data()](auto bit_index) { return bit_is_set(bitmask, bit_index); });

  // Return only filtered row groups based on predicate
  auto is_row_group_required = cudf::detail::make_pinned_vector(
    cudf::device_span<uint8_t const>{predicate.data<uint8_t>(),
                                     static_cast<size_t>(predicate.size())},
    stream);

  // Return if all are required, or all are nulls.
  if (predicate.null_count() == predicate.size() or std::all_of(is_row_group_required.cbegin(),
                                                                is_row_group_required.cend(),
                                                                [](auto i) { return bool(i); })) {
    return std::nullopt;
  }

  // Collect indices of the filtered row groups
  size_type is_required_idx = 0;
  std::vector<std::vector<size_type>> filtered_row_group_indices;
  for (auto const& input_row_group_index : input_row_group_indices) {
    std::vector<size_type> filtered_row_groups;
    for (auto const rg_idx : input_row_group_index) {
      if ((!validity_it[is_required_idx]) || is_row_group_required[is_required_idx]) {
        filtered_row_groups.push_back(rg_idx);
      }
      ++is_required_idx;
    }
    filtered_row_group_indices.push_back(std::move(filtered_row_groups));
  }

  return {filtered_row_group_indices};
}

}  // namespace cudf::io::parquet::detail
