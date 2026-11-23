/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/evaluation_status.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace cudf::detail::row_ir {
enum class opcode : int32_t;
}  // namespace cudf::detail::row_ir

namespace cudf::detail {

struct error_operator {
  std::variant<row_ir::opcode, std::string> op;
  std::optional<int32_t> expression_id{};
};

void handle_evaluation_status(packed_evaluation_status packed,
                              std::span<error_operator const> operators,
                              std::string_view operation,
                              std::string_view row_label = "row");

}  // namespace cudf::detail
