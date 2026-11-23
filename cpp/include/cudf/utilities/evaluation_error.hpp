/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#include <exception>
#include <optional>
#include <string>
#include <utility>

namespace CUDF_EXPORT cudf {

namespace ast::jit {
enum class op : uint8_t;
}

/**
 * @brief An evaluation failure and the row and operator that produced it.
 * @ingroup utility_error
 *
 * Row and operator context is disengaged when unavailable.
 */
struct evaluation_error : std::exception {
  /**
   * @brief Constructs an error without row or operator context.
   * @param error Evaluation status
   * @param message Diagnostic message
   */
  evaluation_error(errc error, std::string message)
    : evaluation_error(error, std::nullopt, std::nullopt, std::nullopt, std::move(message))
  {
  }

  /**
   * @brief Constructs an error with execution context.
   * @param error Evaluation status
   * @param row_index Logical input row, or candidate-pair index for a join; nullopt if unavailable
   * @param op_name Operator name or UDF symbol, nullopt if unavailable
   * @param op Built-in JIT operator, nullopt for UDF calls or unavailable context
   * @param message Diagnostic message
   */
  evaluation_error(errc error,
                   std::optional<size_type> row_index,
                   std::optional<std::string> op_name,
                   std::optional<ast::jit::op> op,
                   std::string message)
    : error_(error),
      row_index_(row_index),
      op_name_(std::move(op_name)),
      op_(op),
      message_(std::move(message))
  {
  }

  /**
   * @brief Returns the diagnostic message.
   *
   * @return Message with the available context
   */
  [[nodiscard]] char const* what() const noexcept override { return message_.c_str(); }

  /**
   * @brief Returns the evaluation status.
   *
   * @return Error code, including user-defined codes
   */
  [[nodiscard]] errc error_code() const noexcept { return error_; }

  /**
   * @brief Returns the logical input row.
   *
   * @return Row index, or std::nullopt if unavailable
   */
  [[nodiscard]] std::optional<size_type> row_index() const noexcept { return row_index_; }

  /**
   * @brief Returns the operator name or UDF symbol.
   *
   * @return Name, or std::nullopt if unavailable
   */
  [[nodiscard]] std::optional<std::string> const& op_name() const noexcept { return op_name_; }

  /**
   * @brief Returns the built-in JIT operator.
   *
   * @return Operator, or std::nullopt for UDF calls or unavailable context
   */
  [[nodiscard]] std::optional<ast::jit::op> op() const noexcept { return op_; }

 private:
  errc error_;
  std::optional<size_type> row_index_;
  std::optional<std::string> op_name_;
  std::optional<ast::jit::op> op_;
  std::string message_;
};

}  // namespace CUDF_EXPORT cudf
