/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "error.hpp"

#include <cudf/utilities/evaluation_error.hpp>

#include <format>
#include <optional>
#include <string>
#include <utility>

namespace cudf::detail::row_ir {
[[nodiscard]] std::string_view to_string(opcode op);
[[nodiscard]] std::optional<ast::jit::op> get_ast_operator(opcode op);
}  // namespace cudf::detail::row_ir

namespace cudf::detail {

void handle_evaluation_status(packed_evaluation_status packed,
                              std::span<error_operator const> operators,
                              std::string_view operation)
{
  auto const status = evaluation_status::unpack(packed);

  if (status.error_code == errc::SUCCESS) { return; }

  auto const index = status.op_index;
  auto const error = status.error_code;
  std::optional<size_type> row;
  std::optional<std::string> name;
  std::optional<ast::jit::op> op;
  std::string location;

  if (status.row_index != -1) {
    row      = status.row_index;
    location = std::format(" at row {}", *row);
  }

  if (index < operators.size()) {
    auto const& entry = operators[index];
    if (auto const* opcode = std::get_if<row_ir::opcode>(&entry)) {
      name = row_ir::to_string(*opcode);
      op   = row_ir::get_ast_operator(*opcode);
    } else {
      name = std::get<std::string>(entry);
    }
    location += std::format(" at operator {}{}", op.has_value() ? "ast::jit::op::" : "", *name);
  }

  throw evaluation_error(error,
                         row,
                         std::move(name),
                         op,
                         std::format("{} failed{} with error `{}` ({})",
                                     operation,
                                     location,
                                     to_string(error),
                                     static_cast<error_int>(error)));
}

}  // namespace cudf::detail
