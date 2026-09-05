/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudf {
namespace jni {
namespace ast {

enum class compilation_mode { DEFAULT, JIT };

/** A class to capture all resources associated with a compiled AST expression. */
class compiled_expr {
  compilation_mode const mode;

  // Keep literal owners before the tree so its non-owning nodes are destroyed first.
  /** GPU scalar instances that correspond to literal nodes */
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  /** One-row columns backing literals in a JIT expression tree */
  std::vector<std::unique_ptr<cudf::column>> scalar_columns;

  /** All expression nodes within the expression tree */
  cudf::ast::tree expressions;

 public:
  explicit compiled_expr(compilation_mode mode) : mode{mode} {}

  template <typename ScalarType>
  cudf::ast::literal const& add_literal(ScalarType& scalar,
                                        std::unique_ptr<cudf::scalar> scalar_ptr)
  {
    if (is_jit()) {
      auto scalar_column = cudf::make_column_from_scalar(scalar, 1);
      scalars.push_back(std::move(scalar_ptr));
      scalar_columns.push_back(std::move(scalar_column));
      return expressions.emplace<cudf::ast::literal>(
        cudf::scalar_column_view{scalar_columns.back()->view()});
    }

    scalars.push_back(std::move(scalar_ptr));
    return expressions.emplace<cudf::ast::literal>(scalar);
  }

  cudf::ast::column_reference const& add_column_ref(cudf::size_type column_index,
                                                    cudf::ast::table_reference table_ref)
  {
    return expressions.emplace<cudf::ast::column_reference>(column_index, table_ref);
  }

  cudf::ast::column_name_reference const& add_column_name_ref(std::string column_name)
  {
    return expressions.emplace<cudf::ast::column_name_reference>(std::move(column_name));
  }

  cudf::ast::operation const& add_operation(cudf::ast::ast_operator op,
                                            cudf::ast::expression const& child)
  {
    return expressions.emplace<cudf::ast::operation>(op, child);
  }

  cudf::ast::operation const& add_operation(cudf::ast::ast_operator op,
                                            cudf::ast::expression const& left,
                                            cudf::ast::expression const& right)
  {
    return expressions.emplace<cudf::ast::operation>(op, left, right);
  }

  template <typename F>
  cudf::ast::expression const& add_jit_expression(F&& factory)
  {
    if (!is_jit()) {
      throw std::invalid_argument("JIT operations require an expression compiled for JIT");
    }
    return factory(expressions);
  }

  [[nodiscard]] bool has_literals() const { return !scalars.empty(); }

  [[nodiscard]] bool has_jit_literals() const { return !scalar_columns.empty(); }

  [[nodiscard]] bool is_jit() const { return mode == compilation_mode::JIT; }

  void release_jit_staging_scalars()
  {
    if (is_jit()) { scalars.clear(); }
  }

  /** Return the expression node at the top of a default-compatible tree */
  cudf::ast::expression const& get_top_expression() const
  {
    if (is_jit()) {
      throw std::logic_error("JIT-compiled expressions cannot be used by a default AST consumer");
    }
    return expressions.back();
  }

  /** Return the expression node at the top of the JIT tree */
  cudf::ast::expression const& get_jit_top_expression() const
  {
    if (!is_jit()) { throw std::logic_error("Expression was not compiled for JIT"); }
    return expressions.back();
  }
};

}  // namespace ast
}  // namespace jni
}  // namespace cudf
