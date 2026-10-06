/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/row_operator/primitive_common.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>

namespace cudf::detail {
bool is_primitive_row_op_compatible(cudf::table_view const& table)
{
  return std::all_of(
    table.begin(), table.end(), [](auto const& col) { return cudf::is_numeric(col.type()); });
}
}  // namespace cudf::detail
