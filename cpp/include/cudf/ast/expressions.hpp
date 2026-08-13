/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/ast_operator.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <vector>

/**
 * @file
 * @brief Class definitions for building and evaluating abstract syntax tree expressions.
 */

namespace CUDF_EXPORT cudf {

namespace detail {
namespace row_ir {

/**
 * @brief The base class for all IR nodes
 *
 * This class defines the interface for IR nodes, which can be instantiated and used to generate
 * code. Each IR node represents a specific operation or value in the program. They represent a
 * single-static-assignment (SSA) variable in the program IR. It is separate from the AST as it
 * contains more detailed program information and analysis that would be needed to instantiate the
 * program and generate correct and robust code.
 */
struct node;

/**
 * @brief A converter that converts AST expressions to IR nodes and CUDA UDFs.
 */
struct ast_converter;

}  // namespace row_ir
}  // namespace detail

namespace ast {
/**
 * @addtogroup expressions
 * @{
 */

// Forward declaration.
namespace detail {
class expression_parser;
class expression_transformer;
}  // namespace detail

/**
 * @brief A generic expression that can be evaluated to return a value.
 *
 * This class is a part of a "visitor" pattern with the `expression_parser` class.
 * Expressions inheriting from this class can accept parsers as visitors.
 */
struct [[nodiscard]] expression {
  /**
   * @brief Accepts a visitor class.
   *
   * @param visitor The `expression_parser` parsing this expression tree
   * @return Index of device data reference for this instance
   */
  virtual cudf::size_type accept(detail::expression_parser& visitor) const = 0;

  /**
   * @brief Accepts a visitor class.
   *
   * @param visitor The `expression_transformer` transforming this expression tree
   * @return Reference wrapper of transformed expression
   */
  virtual std::reference_wrapper<expression const> accept(
    detail::expression_transformer& visitor) const = 0;

  /**
   * @brief Accepts an `row_ir::ast_converter` class.
   *
   * @param visitor The `row_ir::ast_converter` converting this expression tree
   * @return The IR node representing this expression
   */
  [[nodiscard]] virtual std::unique_ptr<cudf::detail::row_ir::node> accept(
    cudf::detail::row_ir::ast_converter& visitor) const = 0;

  /**
   * @brief Returns true if the expression may evaluate to null.
   *
   * @param left The left operand of the expression (The same is used as right operand)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return `true` if the expression may evaluate to null, otherwise false
   */
  [[nodiscard]] bool may_evaluate_null(table_view const& left, rmm::cuda_stream_view stream) const
  {
    return may_evaluate_null(left, left, stream);
  }

  /**
   * @brief Returns true if the expression may evaluate to null.
   *
   * @param left The left operand of the expression
   * @param right The right operand of the expression
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return `true` if the expression may evaluate to null, otherwise false
   */
  [[nodiscard]] virtual bool may_evaluate_null(table_view const& left,
                                               table_view const& right,
                                               rmm::cuda_stream_view stream) const = 0;

  virtual ~expression() {}
};

/**
 * @brief Enum of table references.
 *
 * This determines which table to use in cases with two tables (e.g. joins).
 */
enum class table_reference {
  LEFT,   ///< Column index in the left table
  RIGHT,  ///< Column index in the right table
  OUTPUT  ///< Column index in the output table
};

/**
 * @brief A type-erased scalar_device_view where the value is a fixed width type or a string
 */
class generic_scalar_device_view : public cudf::detail::scalar_device_view_base {
 public:
  /**
   * @brief Returns the stored value.
   *
   * @tparam T The desired type
   * @returns The stored value
   */
  template <typename T>
  __device__ T const value() const noexcept
  {
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      return string_view(static_cast<char const*>(_data), _size);
    }
    if constexpr (cudf::is_fixed_point<T>()) {
      using rep_type   = typename T::rep;
      auto const rep   = *static_cast<rep_type const*>(_data);
      auto const scale = numeric::scale_type{type().scale()};
      return T{numeric::scaled_integer<rep_type>{rep, scale}};
    }
    return *static_cast<T const*>(_data);
  }

  /** @brief Construct a new generic scalar device view object from a numeric scalar
   *
   * @param s The numeric scalar to construct from
   */
  template <typename T>
  generic_scalar_device_view(numeric_scalar<T>& s)
    : generic_scalar_device_view(s.type(), s.data(), s.validity_data())
  {
  }

  /** @brief Construct a new generic scalar device view object from a timestamp scalar
   *
   * @param s The timestamp scalar to construct from
   */
  template <typename T>
  generic_scalar_device_view(timestamp_scalar<T>& s)
    : generic_scalar_device_view(s.type(), s.data(), s.validity_data())
  {
  }

  /** @brief Construct a new generic scalar device view object from a duration scalar
   *
   * @param s The duration scalar to construct from
   */
  template <typename T>
  generic_scalar_device_view(duration_scalar<T>& s)
    : generic_scalar_device_view(s.type(), s.data(), s.validity_data())
  {
  }

  /** @brief Construct a new generic scalar device view object from a string scalar
   *
   * @param s The string scalar to construct from
   */
  generic_scalar_device_view(string_scalar& s)
    : generic_scalar_device_view(s.type(), s.data(), s.validity_data(), s.size())
  {
  }

  /** @brief Construct a new generic scalar device view object from a fixed-point scalar
   *
   * @param s The fixed-point scalar to construct from
   */
  template <typename T>
  generic_scalar_device_view(cudf::fixed_point_scalar<T>& s)
    : generic_scalar_device_view{s.type(), s.data(), s.validity_data()}
  {
  }

 protected:
  void const* _data{};      ///< Pointer to device memory containing the value
  size_type const _size{};  ///< Size of the string in bytes for string scalar

  /**
   * @brief Construct a new fixed width scalar device view object
   *
   * @param type The data type of the value
   * @param data The pointer to the data in device memory
   * @param is_valid The pointer to the bool in device memory that indicates the
   * validity of the stored value
   */
  generic_scalar_device_view(data_type type, void const* data, bool* is_valid)
    : cudf::detail::scalar_device_view_base(type, is_valid), _data(data)
  {
  }

  /** @brief Construct a new string scalar device view object
   *
   * @param type The data type of the value
   * @param data The pointer to the data in device memory
   * @param is_valid The pointer to the bool in device memory that indicates the
   * validity of the stored value
   * @param size The size of the string in bytes
   */
  generic_scalar_device_view(data_type type, void const* data, bool* is_valid, size_type size)
    : cudf::detail::scalar_device_view_base(type, is_valid), _data(data), _size(size)
  {
  }
};

/**
 * @brief A literal value used in an abstract syntax tree.
 */
class literal : public expression {
 public:
  /**
   * @brief Construct a new literal object.
   *
   * @tparam T Numeric scalar template type
   * @param value A numeric scalar value
   */
  template <typename T>
  literal(cudf::numeric_scalar<T>& value) : scalar(value), value(value)
  {
  }

  /**
   * @brief Construct a new literal object.
   *
   * @tparam T Timestamp scalar template type
   * @param value A timestamp scalar value
   */
  template <typename T>
  literal(cudf::timestamp_scalar<T>& value) : scalar(value), value(value)
  {
  }

  /**
   * @brief Construct a new literal object.
   *
   * @tparam T Duration scalar template type
   * @param value A duration scalar value
   */
  template <typename T>
  literal(cudf::duration_scalar<T>& value) : scalar(value), value(value)
  {
  }

  /**
   * @brief Construct a new literal object.
   *
   * @param value A string scalar value
   */
  literal(cudf::string_scalar& value) : scalar(value), value(value) {}

  /**
   * @brief Construct a new literal object.
   *
   * @param value A fixed-point scalar value
   */
  template <typename T>
  literal(cudf::fixed_point_scalar<T>& value) : scalar(value), value(value)
  {
  }

  /**
   * @brief Get the data type.
   *
   * @return The data type of the literal
   */
  [[nodiscard]] cudf::data_type get_data_type() const { return get_value().type(); }

  /**
   * @brief Get the value object.
   *
   * @return The device scalar object
   */
  [[nodiscard]] generic_scalar_device_view get_value() const { return value; }

  /**
   * @brief Get the scalar.
   *
   * @return The scalar object
   */
  [[nodiscard]] cudf::scalar const& get_scalar() const { return scalar; }

  /**
   * @copydoc expression::accept
   */
  cudf::size_type accept(detail::expression_parser& visitor) const override;

  /**
   * @copydoc expression::accept
   */
  std::reference_wrapper<expression const> accept(
    detail::expression_transformer& visitor) const override;

  /**
   * @copydoc expression::accept
   */
  [[nodiscard]] std::unique_ptr<cudf::detail::row_ir::node> accept(
    cudf::detail::row_ir::ast_converter& visitor) const override;

  [[nodiscard]] bool may_evaluate_null(table_view const& left,
                                       table_view const& right,
                                       rmm::cuda_stream_view stream) const override
  {
    return !is_valid(stream);
  }

  /**
   * @brief Check if the underlying scalar is valid.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return true if the underlying scalar is valid
   */
  [[nodiscard]] bool is_valid(rmm::cuda_stream_view stream) const
  {
    return scalar.is_valid(stream);
  }

 private:
  cudf::scalar const& scalar;
  generic_scalar_device_view const value;
};

/**
 * @brief A expression referring to data from a column in a table.
 */
class column_reference : public expression {
 public:
  /**
   * @brief Construct a new column reference object
   *
   * @param column_index Index of this column in the table (provided when the expression is
   * evaluated).
   * @param table_source Which table to use in cases with two tables (e.g. joins)
   */
  column_reference(cudf::size_type column_index,
                   table_reference table_source = table_reference::LEFT)
    : column_index(column_index), table_source(table_source)
  {
  }

  /**
   * @brief Get the column index.
   *
   * @return The column index of the column reference
   */
  [[nodiscard]] cudf::size_type get_column_index() const { return column_index; }

  /**
   * @brief Get the table source.
   *
   * @return table_reference The reference to the table containing this column
   */
  [[nodiscard]] table_reference get_table_source() const { return table_source; }

  /**
   * @brief Get the data type.
   *
   * @param table Table used to determine types
   * @return The data type of the column
   */
  [[nodiscard]] cudf::data_type get_data_type(table_view const& table) const;

  /**
   * @brief Get the data type.
   *
   * @param left_table Left table used to determine types
   * @param right_table Right table used to determine types
   * @return The data type of the column
   */
  [[nodiscard]] cudf::data_type get_data_type(table_view const& left_table,
                                              table_view const& right_table) const;

  /**
   * @copydoc expression::accept
   */
  cudf::size_type accept(detail::expression_parser& visitor) const override;

  /**
   * @copydoc expression::accept
   */
  std::reference_wrapper<expression const> accept(
    detail::expression_transformer& visitor) const override;

  [[nodiscard]] bool may_evaluate_null(table_view const& left,
                                       table_view const& right,
                                       rmm::cuda_stream_view stream) const override
  {
    return (table_source == table_reference::LEFT ? left : right).column(column_index).has_nulls();
  }

  /**
   * @copydoc expression::accept
   */
  [[nodiscard]] std::unique_ptr<cudf::detail::row_ir::node> accept(
    cudf::detail::row_ir::ast_converter& visitor) const override;

 private:
  cudf::size_type column_index;
  table_reference table_source;
};

/**
 * @brief An operation expression holds an operator and zero or more operands.
 */
class operation : public expression {
 public:
  /**
   * @brief Construct a new unary operation object.
   *
   * @param op Operator
   * @param input Input expression (operand)
   */
  operation(ast_operator op, expression const& input);

  /**
   * @brief Construct a new binary operation object.
   *
   * @param op Operator
   * @param left Left input expression (left operand)
   * @param right Right input expression (right operand)
   */
  operation(ast_operator op, expression const& left, expression const& right);

  // operation only stores references to expressions, so it does not accept r-value
  // references: the calling code must own the expressions.
  operation(ast_operator op, expression&& input)                         = delete;
  operation(ast_operator op, expression&& left, expression&& right)      = delete;
  operation(ast_operator op, expression&& left, expression const& right) = delete;
  operation(ast_operator op, expression const& left, expression&& right) = delete;

  /**
   * @brief Get the operator.
   *
   * @return The operator
   */
  [[nodiscard]] ast_operator get_operator() const { return op; }

  /**
   * @brief Get the operands.
   *
   * @return Vector of operands
   */
  [[nodiscard]] std::vector<std::reference_wrapper<expression const>> const& get_operands() const
  {
    return operands;
  }

  /**
   * @copydoc expression::accept
   */
  cudf::size_type accept(detail::expression_parser& visitor) const override;

  /**
   * @copydoc expression::accept
   */
  std::reference_wrapper<expression const> accept(
    detail::expression_transformer& visitor) const override;

  [[nodiscard]] bool may_evaluate_null(table_view const& left,
                                       table_view const& right,
                                       rmm::cuda_stream_view stream) const override;

  /**
   * @copydoc expression::accept
   */
  [[nodiscard]] std::unique_ptr<cudf::detail::row_ir::node> accept(
    cudf::detail::row_ir::ast_converter& visitor) const override;

 private:
  ast_operator op;
  std::vector<std::reference_wrapper<expression const>> operands;
};

namespace detail {

/// @brief An expression that represents a predicate.
///
/// This is an internal expression used in filter operations. It is not intended to be used by
/// external code and is not a part of the public API.
class predicate : public expression {
 public:
  /**
   * @brief Construct a new filter predicate object
   * @param source The source expression from which the predicate value is taken
   */
  predicate(expression const& source) : source_{source} {}

  /**
   * @copydoc expression::accept
   */
  cudf::size_type accept(detail::expression_parser& visitor) const override;

  /**
   * @copydoc expression::accept
   */
  std::reference_wrapper<expression const> accept(
    detail::expression_transformer& visitor) const override;

  [[nodiscard]] bool may_evaluate_null(table_view const& left,
                                       table_view const& right,
                                       rmm::cuda_stream_view stream) const override;

  /**
   * @copydoc expression::accept
   */
  [[nodiscard]] std::unique_ptr<cudf::detail::row_ir::node> accept(
    cudf::detail::row_ir::ast_converter& visitor) const override;

  /**
   * @brief Get the operand expression.
   * @return The operand expression
   */
  [[nodiscard]] expression const& get_operand() const { return source_; }

 private:
  std::reference_wrapper<expression const> source_;
};

}  // namespace detail

/**
 * @brief A expression referring to data from a column in a table.
 */
class column_name_reference : public expression {
 public:
  /**
   * @brief Construct a new column name reference object
   *
   * @param column_name Name of this column in the table metadata (provided when the expression is
   * evaluated).
   */
  column_name_reference(std::string column_name) : column_name(std::move(column_name)) {}

  /**
   * @brief Get the column name.
   *
   * @return The name of this column reference
   */
  [[nodiscard]] std::string get_column_name() const { return column_name; }

  /**
   * @copydoc expression::accept
   */
  cudf::size_type accept(detail::expression_parser& visitor) const override;

  /**
   * @copydoc expression::accept
   */
  std::reference_wrapper<expression const> accept(
    detail::expression_transformer& visitor) const override;

  [[nodiscard]] bool may_evaluate_null(table_view const& left,
                                       table_view const& right,
                                       rmm::cuda_stream_view stream) const override
  {
    return true;
  }

  /**
   * @copydoc expression::accept
   */
  [[nodiscard]] std::unique_ptr<cudf::detail::row_ir::node> accept(
    cudf::detail::row_ir::ast_converter& visitor) const override;

 private:
  std::string column_name;
};

/**
 * @brief An AST expression tree. It owns and contains multiple dependent expressions. All the
 * expressions are destroyed when the tree is destroyed.
 */
class tree {
 public:
  /**
   * @brief construct an empty ast tree
   */
  tree() = default;

  /**
   * @brief Moves the ast tree
   */
  tree(tree&&) = default;

  /**
   * @brief move-assigns the AST tree
   * @returns a reference to the move-assigned tree
   */
  tree& operator=(tree&&) = default;

  ~tree() = default;

  // the tree is not copyable
  tree(tree const&)            = delete;
  tree& operator=(tree const&) = delete;

  /**
   * @brief Add an expression to the AST tree
   * @param args Arguments to use to construct the ast expression
   * @returns a reference to the added expression
   */
  template <typename Expr, typename... Args>
  std::enable_if_t<std::is_base_of_v<expression, Expr>, Expr const&> emplace(Args&&... args)
  {
    auto expr            = std::make_unique<Expr>(std::forward<Args>(args)...);
    Expr const& expr_ref = *expr;
    expressions.emplace_back(std::move(expr));
    return expr_ref;
  }

  /**
   * @brief Add an expression to the AST tree
   * @param expr AST expression to be added
   * @returns a reference to the added expression
   */
  template <typename Expr>
  decltype(auto) push(Expr expr)
  {
    return emplace<Expr>(std::move(expr));
  }

  /**
   * @brief get the first expression in the tree
   * @returns the first inserted expression into the tree
   */
  [[nodiscard]] expression const& front() const { return *expressions.front(); }

  /**
   * @brief get the last expression in the tree
   * @returns the last inserted expression into the tree
   */
  [[nodiscard]] expression const& back() const { return *expressions.back(); }

  /**
   * @brief get the number of expressions added to the tree
   * @returns the number of expressions added to the tree
   */
  [[nodiscard]] size_t size() const { return expressions.size(); }

  /**
   * @brief get the expression at an index in the tree. Index is checked.
   * @param index index of expression in the ast tree
   * @returns the expression at the specified index
   */
  expression const& at(size_t index) { return *expressions.at(index); }

  /**
   * @brief get the expression at an index in the tree. Index is unchecked.
   * @param index index of expression in the ast tree
   * @returns the expression at the specified index
   */
  expression const& operator[](size_t index) const { return *expressions[index]; }

 private:
  // TODO: use better ownership semantics, the unique_ptr here is redundant. Consider using a bump
  // allocator with type-erased deleters.
  std::vector<std::unique_ptr<expression>> expressions;
};

namespace jit {

/**
 * @brief JIT operation kinds for `cudf::ast::jit::operation`.
 */
enum class op : uint8_t {
  // Identity operators
  IDENTITY,

  // Null handling operators
  IS_NULL,

  COALESCE,
  PREDICATE,

  /// Arithmetic operators
  ADD,
  SUB,
  MUL,
  DIV,
  NEG,
  ABS,
  MOD,
  PYMOD,
  TRUE_DIV,
  FLOOR_DIV,

  /// Overflow-checking Arithmetic functions. raise errors on overflow, division by zero, etc.
  ADD_OVERFLOW,
  SUB_OVERFLOW,
  MUL_OVERFLOW,
  DIV_OVERFLOW,
  NEG_OVERFLOW,
  ABS_OVERFLOW,
  MOD_OVERFLOW,
  CHECK_PRECISION,

  /// Bitwise operators
  BITWISE_AND,
  BITWISE_INVERT,
  BITWISE_OR,
  BITWISE_XOR,
  BITWISE_SHIFT_LEFT,
  BITWISE_SHIFT_RIGHT,

  /// Type conversion/scaling operators
  CAST_TO_BOOL8,
  CAST_TO_INT8,
  CAST_TO_INT16,
  CAST_TO_INT32,
  CAST_TO_INT64,
  CAST_TO_UINT8,
  CAST_TO_UINT16,
  CAST_TO_UINT32,
  CAST_TO_UINT64,
  CAST_TO_FLOAT32,
  CAST_TO_FLOAT64,
  CAST_TO_DECIMAL32,
  CAST_TO_DECIMAL64,
  CAST_TO_DECIMAL128,
  RESCALE,

  /// Comparison & Logic operators
  EQUAL,
  NOT_EQUAL,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
  NULL_EQUAL,
  NULL_LOGICAL_AND,
  NULL_LOGICAL_OR,
  LOGICAL_AND,
  LOGICAL_OR,
  LOGICAL_NOT,
  IF_ELSE,

  /// Mathematical operators
  CBRT,
  CEIL,
  FLOOR,
  RINT,
  SQRT,
  POW,
  EXP,
  LOG,

  /// Trigonometric operators
  ARCCOS,
  ARCCOSH,
  ARCSIN,
  ARCSINH,
  ARCTAN,
  ARCTANH,
  COS,
  COSH,
  SIN,
  SINH,
  TAN,
  TANH,
};

/**
 * @brief Creates a JIT expression operation from an opcode and argument list.
 *
 * @param tree The expression tree to which this expression will be added.
 * @param operator_id The JIT operation kind.
 * @param args Operation arguments.
 * @param error_policy Specifies how fallible operations handle errors.
 * @param target_scale Required only for `op::RESCALE`.
 * @return A reference to the created operation expression.
 */
expression const& operation(ast::tree& tree,
                            op operator_id,
                            std::vector<std::reference_wrapper<expression const>> const& args,
                            cudf::error_policy error_policy     = cudf::error_policy::PROPAGATE,
                            std::optional<int32_t> target_scale = std::nullopt);

/**
 * @brief Creates a JIT expression operation from an opcode and argument list.
 *
 * @param tree The expression tree to which this expression will be added.
 * @param operator_id The JIT operation kind.
 * @param args Operation arguments.
 * @param error_policy Specifies how fallible operations handle errors.
 * @param target_scale Required only for `op::RESCALE`.
 * @return A reference to the created operation expression.
 */
expression const& operation(ast::tree& tree,
                            op operator_id,
                            std::initializer_list<std::reference_wrapper<expression const>> args,
                            cudf::error_policy error_policy     = cudf::error_policy::PROPAGATE,
                            std::optional<int32_t> target_scale = std::nullopt);

}  // namespace jit

/** @} */  // end of group
}  // namespace ast
}  // namespace CUDF_EXPORT cudf
