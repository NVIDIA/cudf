/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cudf {
namespace jni {
namespace ast {

/** A class to capture all resources associated with a compiled AST expression. */
class compiled_expr {
  /** GPU scalar instances that correspond to literal nodes */
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  /** All expression nodes within the expression tree */
  cudf::ast::tree expressions;

 public:
  template <typename ScalarType>
  cudf::ast::literal const& add_literal(ScalarType& scalar,
                                        std::unique_ptr<cudf::scalar> scalar_ptr)
  {
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
    return factory(expressions);
  }

  [[nodiscard]] bool has_literals() const { return !scalars.empty(); }

  /** Return the expression node at the top of the tree */
  cudf::ast::expression const& get_top_expression() const { return expressions.back(); }
};

}  // namespace ast
}  // namespace jni
}  // namespace cudf
