/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <benchmarks/io/cuio_common.hpp>

#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

constexpr cudf::size_type num_cols = 64;

void parquet_read_common(cudf::size_type num_rows_to_read,
                         cudf::size_type num_cols_to_read,
                         cuio_source_sink_pair& source_sink,
                         nvbench::state& state);

// Writes a single-column file with an explicitly controlled row group and page layout
[[nodiscard]] cuio_source_sink_pair write_file_shape_parquet_file(
  cudf::type_id dtype,
  cudf::size_type num_rows,
  cudf::size_type num_row_groups,
  cudf::size_type pages_per_row_group,
  io_type source_type,
  bool write_page_index);

// A mix of string, integral, float, decimal and list types used to fill a schema of the
// requested width
[[nodiscard]] std::vector<cudf::type_id> const& mixed_dtypes();

// Writes a mixed-type file with a fixed number of rows per row group
[[nodiscard]] cuio_source_sink_pair write_mixed_dtype_parquet_file(cudf::size_type num_cols,
                                                                   cudf::size_type num_row_groups,
                                                                   io_type source_type,
                                                                   bool write_page_index);

// Writes a negligible single-row file with deterministically named columns
[[nodiscard]] cuio_source_sink_pair write_named_resolution_parquet_file(cudf::size_type num_cols,
                                                                        io_type source_type);
