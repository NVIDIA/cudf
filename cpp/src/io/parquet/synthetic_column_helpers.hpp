/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reader_impl_chunking.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream_ref>

#include <cstddef>
#include <memory>
#include <span>

namespace cudf::io::parquet::detail {

struct row_group_info;

[[nodiscard]] std::unique_ptr<column> synthesize_source_index_column(
  std::span<std::size_t const> num_rows_per_source,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);
[[nodiscard]] std::unique_ptr<column> synthesize_row_group_index_column(
  column_view const& source_indices, cuda::stream_ref stream, rmm::device_async_resource_ref mr);
[[nodiscard]] std::unique_ptr<column> synthesize_row_index_column(
  std::span<row_group_info const> row_groups,
  row_range const& read_info,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::io::parquet::detail
