/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/functional>

#include <memory>

/**
 * @file
 * @brief Column APIs for interleave and tile
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup column_reshape
 * @{
 */

/**
 * @brief Interleave columns of a table into a single column.
 *
 * Converts the column major table `input` into a row major column.
 * Example:
 * ```
 * in     = [[A1, A2, A3], [B1, B2, B3]]
 * return = [A1, B1, A2, B2, A3, B3]
 * ```
 *
 * @throws cudf::logic_error if input contains no columns.
 * @throws cudf::logic_error if input columns dtypes are not identical.
 *
 * @param input Table containing columns to interleave
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The interleaved columns as a single column
 */
std::unique_ptr<column> interleave_columns(
  table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Repeats the rows from `input` table `count` times to form a new table.
 *
 * `output.num_columns() == input.num_columns()`
 * `output.num_rows() == input.num_rows() * count`
 *
 * ```
 * input  = [[8, 4, 7], [5, 2, 3]]
 * count  = 2
 * return = [[8, 4, 7, 8, 4, 7], [5, 2, 3, 5, 2, 3]]
 * ```
 *
 * @param input Table containing rows to be repeated
 * @param count Number of times to tile "rows". Must be non-negative
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 *
 * @return The table containing the tiled "rows"
 */
std::unique_ptr<table> tile(
  table_view const& input,
  size_type count,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Configures whether byte casting flips endianness
 */
enum class flip_endianness : bool { NO, YES };

/**
 * @brief Converts a column's elements to lists of bytes
 *
 * ```
 * input<int32>  = [8675, 309]
 * configuration = flip_endianness::YES
 * return        = [[0x00, 0x00, 0x21, 0xe3], [0x00, 0x00, 0x01, 0x35]]
 * ```
 *
 * @param input_column Column to be converted to lists of bytes
 * @param endian_configuration Whether to retain or flip the endianness of the elements
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @return The column containing the lists of bytes
 */
std::unique_ptr<column> byte_cast(
  column_view const& input_column,
  flip_endianness endian_configuration,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Copies a table into a contiguous column-major device array.
 *
 * This function copies a `table_view` with columns of the same fixed-width type
 * into a 2D device array stored in column-major order.
 *
 * The output buffer must be preallocated and passed as a `device_span` using
 * a `device_span<cuda::std::byte>`. It must be large enough to hold
 * `num_rows * num_columns * sizeof(dtype)` bytes.
 *
 * @throws cudf::logic_error if columns do not all have the same type
 * @throws cudf::logic_error if the dtype of the columns is not a fixed-width type
 * @throws std::invalid_argument if the output span is too small
 *
 * @param input A table with fixed-width, non-nullable columns of the same type
 * @param output A span representing preallocated device memory for the output
 * @param stream CUDA stream used for memory operations
 */
void table_to_array(table_view const& input,
                    device_span<cuda::std::byte> output,
                    rmm::cuda_stream_view stream = cudf::get_default_stream());

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
