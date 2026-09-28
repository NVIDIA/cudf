/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <algorithm>
#include <memory>
namespace cudf {

/// Column factory that adopts child columns.
std::unique_ptr<cudf::column> make_structs_column(
  size_type num_rows,
  std::vector<std::unique_ptr<column>>&& child_columns,
  size_type null_count,
  rmm::device_buffer&& null_mask,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(null_count <= 0 || !null_mask.is_empty(),
               "Struct column with nulls must be nullable.");

  CUDF_EXPECTS(std::all_of(child_columns.begin(),
                           child_columns.end(),
                           [&](auto const& child_col) { return num_rows == child_col->size(); }),
               "Child columns must have the same number of rows as the Struct column.");

  if (!null_mask.is_empty()) {
    for (auto& child : child_columns) {
      child = structs::detail::superimpose_and_sanitize_nulls(
        static_cast<bitmask_type const*>(null_mask.data()),
        null_count,
        std::move(child),
        stream,
        mr);
    }
  }

  return std::make_unique<column>(cudf::data_type{type_id::STRUCT},
                                  num_rows,
                                  rmm::device_buffer{},  // Empty data buffer. Structs hold no data.
                                  std::move(null_mask),
                                  null_count,
                                  std::move(child_columns));
}

/// Column factory that adopts child columns and constructs struct hierarchy
std::unique_ptr<cudf::column> create_structs_hierarchy(
  size_type num_rows,
  std::vector<std::unique_ptr<column>>&& child_columns,
  size_type null_count,
  rmm::device_buffer&& null_mask,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(null_count <= 0 || !null_mask.is_empty(),
               "Struct column with nulls must be nullable.");

  CUDF_EXPECTS(null_mask.is_empty() || null_mask.size() == bitmask_allocation_size_bytes(num_rows),
               "Number of bits in null_mask should equal number of rows in input columns");

  CUDF_EXPECTS(std::all_of(child_columns.begin(),
                           child_columns.end(),
                           [&](auto const& child_col) { return num_rows == child_col->size(); }),
               "Child columns must have the same number of rows as the Struct column.");

  return std::make_unique<column>(cudf::data_type{type_id::STRUCT},
                                  num_rows,
                                  rmm::device_buffer{},  // Empty data buffer. Structs hold no data.
                                  std::move(null_mask),
                                  null_count,
                                  std::move(child_columns));
}

}  // namespace cudf
