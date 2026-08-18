/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/common/utils.hpp"

#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf::groupby::detail {

std::pair<rmm::device_buffer, bitmask_type const*> compute_row_bitmask(table_view const& keys,
                                                                       cuda::stream_ref stream)
{
  auto const mr = cudf::get_current_device_resource_ref();
  if (keys.num_columns() == 0 || !cudf::has_nulls(keys)) {
    return {rmm::device_buffer{0, stream, mr}, nullptr};
  }
  // Single-column fast path: reuse the column's null mask directly.
  if (keys.num_columns() == 1) {
    auto const& col = keys.column(0);
    if (col.offset() == 0) { return {rmm::device_buffer{0, stream, mr}, col.null_mask()}; }
    auto buf = cudf::copy_bitmask(col, stream, mr);
    auto ptr = static_cast<bitmask_type const*>(buf.data());
    return {std::move(buf), ptr};
  }
  auto [buf, null_count] = cudf::bitmask_and(keys, stream, mr);
  if (null_count == 0) { return {rmm::device_buffer{0, stream, mr}, nullptr}; }
  return {std::move(buf), static_cast<bitmask_type const*>(buf.data())};
}

}  // namespace cudf::groupby::detail
