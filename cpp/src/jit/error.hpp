/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/evaluation_status.hpp>

#include <cstdint>
#include <span>
#include <string_view>

namespace cudf::detail::row_ir {
enum class opcode : int32_t;
}  // namespace cudf::detail::row_ir

namespace cudf::detail {

void handle_evaluation_status(packed_evaluation_status packed,
                              std::span<row_ir::opcode const> operators,
                              std::string_view operation);

}  // namespace cudf::detail
