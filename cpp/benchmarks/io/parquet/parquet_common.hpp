/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <benchmarks/io/cuio_common.hpp>

#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <optional>

constexpr cudf::size_type num_cols = 64;

/**
 * @brief Map a `null_percent` axis value to a null probability.
 *
 * `-1` produces no validity mask at all, so the column is written `required`. `0` still produces
 * a mask, so the column is written `optional` -- the distinction matters because a decoder's
 * null handling keys off the definition levels a mask implies, not off the null count.
 */
std::optional<double> null_probability_from_percent(int64_t null_percent);

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
