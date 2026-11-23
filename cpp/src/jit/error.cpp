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
                              std::string_view operation,
                              std::string_view row_label)
{
  auto const status = evaluation_status::unpack(packed);

  if (status.error_code == errc::SUCCESS) { return; }

  auto const index = status.op_index;
  auto const error = status.error_code;
  std::optional<size_type> row;
  std::optional<std::string> name;
  std::optional<ast::jit::op> op;
  std::optional<int32_t> expression_id;
  std::string location;

  if (status.row_index != -1) {
    row      = status.row_index;
    location = std::format(" at {} {}", row_label, *row);
  }

  if (index < operators.size()) {
    auto const& entry = operators[index];
    expression_id     = entry.expression_id;
    if (auto const* opcode = std::get_if<row_ir::opcode>(&entry.op)) {
      name = row_ir::to_string(*opcode);
      op   = row_ir::get_ast_operator(*opcode);
    } else {
      name = std::get<std::string>(entry.op);
    }
    location += std::format(" at operator `{}{}`", op.has_value() ? "ast::jit::op::" : "", *name);
  }

  throw evaluation_error(error,
                         row,
                         std::move(name),
                         op,
                         std::format("{} failed{} with error `{}` ({})",
                                     operation,
                                     location,
                                     to_string(error),
                                     static_cast<error_int>(error)),
                         expression_id);
}

}  // namespace cudf::detail
