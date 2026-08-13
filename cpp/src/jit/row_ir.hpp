/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "ast/jit/expressions.hpp"

#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/io/types.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief cudf IR for operations acting on elements of a row
 */
namespace row_ir {

/**
 * @brief The target for which the IR is generated.
 */
enum class target {
  CUDA = 0  /// < CUDA C++
};

/**
 * @brief The information about the variable used in the IR.
 */
struct var_info {
  std::string id = {};                         ///< The variable identifier
  data_type type = data_type{type_id::EMPTY};  ///< The data type of the variable
};

/**
 * @brief The information about the variable used in the IR without type information. The type
 * information is auto-deduced from an IR node.
 */
struct untyped_var_info {
  std::string id = {};  ///< The variable identifier
};

/**
 * @brief The information about the target for which the IR is generated.
 */
struct target_info {
  target id = target::CUDA;  ///< The target identifier
};

struct scalar_input {
  std::unique_ptr<column> scalar_column =
    nullptr;  ///< The scalar value represented as a column with a single element
};

struct column_input {
  column_view column                  = {};  ///< The column input
  std::optional<int32_t> table_source = std::nullopt;
  std::optional<int32_t> column_index = std::nullopt;
};

using input = std::variant<scalar_input, column_input>;

/**
 * @brief The arguments needed to invoke a `cudf::transform`
 */
struct [[nodiscard]] transform_args {
  std::vector<std::unique_ptr<column>> scalar_columns      = {};
  std::vector<std::optional<int32_t>> input_table_sources  = {};
  std::vector<std::optional<int32_t>> input_column_indices = {};
  std::string udf                                          = {};
  udf_source_type source_type                              = cudf::udf_source_type::CUDA;
  null_aware is_null_aware                                 = null_aware::NO;
  std::optional<void*> user_data                           = std::nullopt;
  std::vector<transform_input> inputs                      = {};
  std::vector<transform_output> outputs                    = {};
  std::vector<std::unique_ptr<column>> string_offsets      = {};
  std::optional<size_type> row_size                        = std::nullopt;
};

/**
 * @brief The context within which the IR is instantiated.
 * This context is used to generate temporary variable identifiers and any state setup needed for
 * the IR instantiation.
 */
struct [[nodiscard]] instance_context {
 private:
  int32_t num_tmp_vars_   = 0;                 ///< The number of temporary variables generated
  std::string tmp_prefix_ = "tmp_";            ///< The prefix for temporary variable identifiers
  bool has_nulls_         = false;             ///< If expressions involve null values
  std::vector<input> inputs_;                  ///< The inputs for the IR
  std::vector<var_info> input_vars_;           ///< The input variables for the IR
  std::vector<untyped_var_info> output_vars_;  ///< The output variables for the IR
  rmm::cuda_stream_view
    stream_;  ///< The CUDA stream for any device operations during IR generation
  rmm::device_async_resource_ref
    mr_;  ///< The device memory resource for any device memory allocation during IR generation

 public:
  friend struct ast_converter;
  friend struct node;

  instance_context(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
    : stream_(stream), mr_(mr)
  {
  }

  instance_context(instance_context const&) = delete;

  instance_context& operator=(instance_context const&) = delete;

  instance_context(instance_context&&) = default;  ///< Move constructor

  instance_context& operator=(instance_context&&) = default;  ///< Move assignment operator

  ~instance_context() = default;  ///< Destructor

  [[nodiscard]] int32_t add_output();

  [[nodiscard]] int32_t add_input(input in);

  [[nodiscard]] int32_t add_input(scalar const& scalar)
  {
    return add_input(
      scalar_input{.scalar_column = make_column_from_scalar(scalar, 1, stream_, mr_)});
  }

  [[nodiscard]] int32_t add_input(column_view const& column)
  {
    return add_input(column_input{.column = column});
  }

  /**
   * @brief Generate a globally unique temporary variable identifier
   * @return A unique temporary variable identifier
   */
  [[nodiscard]] std::string make_tmp_id();

  /**
   * @brief Returns true if expressions involve null values
   */
  [[nodiscard]] bool has_nulls() const;

  /**
   * @brief Sets whether expressions involve null values
   * @param has_nulls True if expressions involve null values
   */
  void set_has_nulls(bool has_nulls);

  /**
   * @brief Get the input values for the IR
   * @return A span of input values for the IR
   */
  [[nodiscard]] std::span<input const> get_inputs() const;

  /**
   * @brief Get the input variables for the IR
   * @return A span of input variable information
   */
  [[nodiscard]] std::span<var_info const> get_input_vars() const;

  /**
   * @brief Get the output variables for the IR
   * @return A span of output variable information
   */
  [[nodiscard]] std::span<untyped_var_info const> get_output_vars() const;

  /**
   * @brief Get the CUDA stream for device operations during IR generation
   * @return The CUDA stream for device operations during IR generation
   */
  [[nodiscard]] rmm::cuda_stream_view get_stream() const { return stream_; }

  /**
   * @brief Get the device memory resource for device memory allocation during IR generation
   * @return The device memory resource for device memory allocation during IR generation
   */
  [[nodiscard]] rmm::device_async_resource_ref get_mr() const { return mr_; }
};

struct [[nodiscard]] code_sink {
 private:
  std::string code_;

 public:
  void emit(std::string_view code) { code_ += code; }

  [[nodiscard]] std::string const& get_code() const { return code_; }
};

struct [[nodiscard]] input_reference {
  int32_t index = 0;  ///< The index of the input variable
};

struct [[nodiscard]] output_reference {
  int32_t index = 0;  ///< The index of the output variable
};

struct [[nodiscard]] node {
 private:
  std::variant<std::monostate, input_reference, output_reference> reference_ =
    std::monostate{};  ///< The index of the input/output variable
  opcode op_                           = opcode::GET_INPUT;  ///< The operation code
  std::optional<int32_t> target_scale_ = std::nullopt;       ///< The target scale for decimal
  input_reference scale_reference_     = {};  ///< The index of the target scale as an IR input
  error_policy error_policy_ =
    cudf::error_policy::PROPAGATE;                ///< The error policy for the operation
  std::vector<std::unique_ptr<node>> args_ = {};  ///< The arguments of the operation

  data_type type_ = {};  ///< The resolved type information of the IR node

  std::string id_ = {};  ///< The identifier of the IR node

  /**
   * @brief Create a set of argument IR nodes
   */
  template <typename... T>
    requires(std::is_same_v<node, T> && ...)
  static std::vector<std::unique_ptr<node>> arguments(T... args)
  {
    std::vector<std::unique_ptr<node>> result;
    (result.emplace_back(std::make_unique<node>(std::move(args))), ...);
    return result;
  }

  /**
   * @brief Create a set of argument IR nodes
   */
  template <typename... T>
    requires(std::is_same_v<std::unique_ptr<node>, T> && ...)
  static std::vector<std::unique_ptr<node>> arguments(T... args)
  {
    std::vector<std::unique_ptr<node>> result;
    (result.emplace_back(std::move(args)), ...);
    return result;
  }

 public:
  /**
   * @brief Construct a new operation IR node
   * @param op The operation code
   * @param target_scale The target scale for decimal rescaling if applicable
   * @param error_policy The error policy for the operation
   * @param args The arguments of the operation
   */
  node(opcode op,
       std::optional<int32_t> target_scale,
       error_policy error_policy,
       std::vector<std::unique_ptr<node>> args);

  /**
   * @brief Construct a new operation IR node
   * @param op The operation code
   * @param target_scale The target scale for decimal rescaling if applicable
   * @param error_policy The error policy for the operation
   * @param args The arguments of the operation
   */
  template <typename... T>
    requires(std::is_same_v<node, T> && ...)
  node(opcode op, std::optional<int32_t> target_scale, error_policy error_policy, T... args)
    : node(op, target_scale, error_policy, arguments(std::move(args)...))
  {
  }

  /**
   * @brief Construct a new operation IR node
   * @param op The operation code
   * @param target_scale The target scale for decimal rescaling if applicable
   * @param error_policy The error policy for the operation
   * @param args The arguments of the operation
   */
  template <typename... T>
    requires(std::is_same_v<node, T> && ...)
  node(opcode op,
       std::optional<int32_t> target_scale,
       error_policy error_policy,
       std::unique_ptr<T>... args)
    : node(op, target_scale, error_policy, arguments(std::move(args)...))
  {
  }

  /**
   * @brief Construct a new input reference IR node
   * @param input The index of the input variable
   */
  node(input_reference input);

  /**
   * @brief Construct a new output reference IR node
   * @param output The index of the output variable
   * @param arg The argument node that produces the value to be set to the output variable
   */
  node(output_reference reference, std::unique_ptr<node> arg);

  /**
   * @brief Construct a new output reference IR node
   * @param output The index of the output variable
   * @param arg The argument node that produces the value to be set to the output variable
   */
  node(output_reference reference, node arg);

  node(node const& other)            = delete;
  node(node&& other)                 = default;  ///< Move constructor
  node& operator=(node const& other) = delete;
  node& operator=(node&& other)      = default;  ///< Move assignment operator
  ~node()                            = default;  ///< Destructor

  /**
   * @brief Get the identifier of the IR node
   * @return The identifier of the IR node
   */
  [[nodiscard]] std::string_view get_id() const;

  /**
   * @brief Get the type info of the IR node
   * @return The type information of the IR node
   */
  [[nodiscard]] data_type get_type() const;

  /**
   * @brief Get the target scale for decimal rescaling if applicable
   * @return The target scale for decimal rescaling if applicable, std::nullopt otherwise
   */
  [[nodiscard]] std::optional<int32_t> get_target_scale() const;

  /**
   * @brief Get the operation code of the operation
   * @return The operation code of the operation
   */
  [[nodiscard]] opcode get_opcode() const;

  /** @brief Get the arguments of the operation
   * @return A span of unique pointers to the arguments of the operation
   */
  [[nodiscard]] std::span<std::unique_ptr<node> const> get_args() const;

  /**
   * @brief Returns `false` if this node forwards nulls from its inputs to its output.
   * e.g., `ADD` operator is not null-aware because if any of its inputs is null, the output is
   * null. but `NULL_EQUAL` operator is null-aware because it can produce a non-null output even if
   * its inputs are null.
   */
  [[nodiscard]] bool is_null_aware() const;

  /**
   * @brief Returns `true` if this node always produces a valid output even if its inputs are
   * nullable, e.g., `IS_NULL` operator produces a valid boolean output regardless of the
   * nullability of its input.
   */
  [[nodiscard]] bool is_always_valid() const;

  /**
   * @brief Instantiate the IR node with the given context and instance information, setting up any
   * necessary state and preprocessing needed for code generation.
   * @param ctx The context within which the IR is instantiated
   * @param info The instance information
   */
  void instantiate(instance_context& ctx);

  /**
   * @brief Generate the code for the IR node based on the instance context and target information.
   * @param ctx The context within which the IR is instantiated
   * @param info The target information
   * @param instance The instance information
   * @param sink The code sink to which the generated code is emitted
   */
  void emit_code(instance_context& ctx, target_info const& info, code_sink& sink) const;
};

/**
 * @brief AST Converter is a class for converting AST expressions to codegen targets, ie. CUDA.
 */
struct [[nodiscard]] ast_converter {
 private:
  std::vector<std::unique_ptr<row_ir::node>> output_irs_;  ///< The output IR nodes
  rmm::cuda_stream_view
    stream_;  ///< CUDA stream used for device memory operations and kernel launches.
  rmm::device_async_resource_ref
    mr_;  ///< Device memory resource used to allocate the returned table's device memory
  instance_context instance_;  ///< The instance context used during the IR generation
  table_view left_table_;      ///< The left input table for the expression
  table_view right_table_;     ///< The right input table for the expression

 public:
  /**
   * @brief Construct a new AST Converter object
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table's device memory
   */
  ast_converter(rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr,
                table_view left_table,
                table_view right_table)
    : stream_(std::move(stream)),
      mr_(std::move(mr)),
      instance_(stream_, mr_),
      left_table_(std::move(left_table)),
      right_table_(std::move(right_table))
  {
  }

  ast_converter(ast_converter const&)            = delete;
  ast_converter& operator=(ast_converter const&) = delete;

  ast_converter(ast_converter&&) = default;  ///< Move constructor

  ast_converter& operator=(ast_converter&&) = default;  ///< Move assignment operator

  ~ast_converter() = default;  ///< Destructor

 public:
  [[nodiscard]] std::unique_ptr<row_ir::node> add_ir_node(ast::literal const& expr);

  [[nodiscard]] std::unique_ptr<row_ir::node> add_ir_node(ast::column_reference const& expr);

  [[nodiscard]] std::unique_ptr<row_ir::node> add_ir_node(ast::operation const& expr);

  [[nodiscard]] std::unique_ptr<row_ir::node> add_ir_node(ast::detail::predicate const& expr);

  [[nodiscard]] std::unique_ptr<row_ir::node> add_ir_node(ast::jit::detail::operation const& expr);

  [[nodiscard]] std::tuple<std::string, null_aware, output_nullability> generate_code(
    target target, ast::expression const& expr, std::string_view function_name);

  /**
   * @brief Convert an AST `compute_column` expression to a `cudf::transform`
   * @param target The target for which the IR is generated
   * @param expr The AST expression to convert
   * @param left_table The left input table for the expression
   * @param right_table The right input table for the expression
   * @param table The input table for the expression
   * @param function_name The name of the generated function
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table's device memory
   * @return The result of the conversion, containing the transform arguments and scalar columns
   */
  static transform_args compute_column(target target,
                                       ast::expression const& expr,
                                       table_view const& left_table,
                                       table_view const& right_table,
                                       std::string_view function_name,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

  /**
   * @brief Convert an AST `filter` expression to a `cudf::filter`
   * @param target The target for which the IR is generated
   * @param expr The AST expression to convert
   * @param left_table The left input table for the expression
   * @param right_table The right input table for the expression
   * @param table The input table for the expression
   * @param function_name The name of the generated function
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned table's device memory
   * @return The result of the conversion, containing the filter arguments and scalar columns
   */
  static transform_args filter(target target,
                               ast::expression const& expr,
                               table_view const& left_table,
                               table_view const& right_table,
                               std::string_view function_name,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);
};

}  // namespace row_ir
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
