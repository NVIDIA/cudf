/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/table/table_view.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace cudf::ast::jit::detail {

struct operation : public ast::expression {
  /**
   * @brief Construct a new operation object with a target scale (for rescale and precision check
   * operations).
   * @param op The opcode for this operation
   * @param args The arguments for this operation
   * @param error_policy The error policy for this operation (only applicable for fallible
   * operations)
   * @param target_scale The target scale for this operation (only applicable for rescale and
   * precision check operations)
   */
  operation(cudf::detail::row_ir::opcode op,
            std::vector<std::reference_wrapper<expression const>> args,
            cudf::error_policy error_policy     = cudf::error_policy::PROPAGATE,
            std::optional<int32_t> target_scale = std::nullopt)
    : op_{op}, args_{std::move(args)}, error_policy_{error_policy}, target_scale_{target_scale}
  {
  }

  operation(operation const&)            = default;  //< Copy constructor
  operation(operation&&)                 = default;  //< Move constructor
  operation& operator=(operation const&) = default;  //< Copy assignment
  operation& operator=(operation&&)      = default;  //< Move assignment
  ~operation() override                  = default;  //< Destructor

  /**
   * @brief Get the opcode.
   *
   * @return The opcode
   */
  [[nodiscard]] cudf::detail::row_ir::opcode get_opcode() const { return op_; }

  /**
   * @brief Get the operands.
   *
   * @return Span of operands
   */
  [[nodiscard]] std::span<std::reference_wrapper<expression const> const> get_arguments() const
  {
    return args_;
  }

  /**
   * @brief Get the target scale for rescale and precision check operations.
   *
   * @return The target scale if applicable, std::nullopt otherwise
   */
  [[nodiscard]] std::optional<int32_t> get_target_scale() const { return target_scale_; }

  /**
   * @brief Whether this operation should nullify the output on error (e.g. overflow, divide by
   * zero)
   *
   * @return The error policy for this operation
   */
  [[nodiscard]] cudf::error_policy get_error_policy() const { return error_policy_; }

  /**
   * @copydoc expression::accept
   */
  cudf::size_type accept(cudf::ast::detail::expression_parser& visitor) const override;

  /**
   * @copydoc expression::accept
   */
  std::reference_wrapper<expression const> accept(
    cudf::ast::detail::expression_transformer& visitor) const override;

  [[nodiscard]] bool may_evaluate_null(table_view const& left,
                                       table_view const& right,
                                       rmm::cuda_stream_view stream) const override;

  /**
   * @copydoc expression::accept
   */
  [[nodiscard]] std::unique_ptr<cudf::detail::row_ir::node> accept(
    cudf::detail::row_ir::ast_converter& visitor) const override;

 private:
  cudf::detail::row_ir::opcode op_;
  std::vector<std::reference_wrapper<expression const>> args_;
  cudf::error_policy error_policy_     = cudf::error_policy::PROPAGATE;
  std::optional<int32_t> target_scale_ = std::nullopt;
};

}  // namespace cudf::ast::jit::detail
