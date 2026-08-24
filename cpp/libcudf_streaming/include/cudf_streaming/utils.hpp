/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstdlib>
#include <string>

namespace cudf_streaming {

/**
 * @brief Converts the element at a specific index in a `cudf::column_view` to a string.
 *
 * @param col The column view containing the data.
 * @param index The index of the element to convert.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resource for device memory allocation.
 * @return A string representation of the element at the specified index.
 */
std::string str(cudf::column_view col,
                cudf::size_type index,
                rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Converts all elements in a `cudf::column_view` to a string.
 *
 * @param col The column view containing the data.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resource for device memory allocation.
 * @return A string representation of all elements in the column.
 */
std::string str(cudf::column_view col,
                rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Converts all rows in a `cudf::table_view` to a string.
 *
 * @param tbl The table view containing the data.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resource for device memory allocation.
 * @return A string representation of all rows in the table.
 */
std::string str(cudf::table_view tbl,
                rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Estimate the memory usage of a column.
 *
 * @param col The column to estimate the memory usage of.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return The estimated memory usage of the column.
 */
std::size_t estimated_memory_usage(cudf::column_view const& col, rmm::cuda_stream_view stream);

/**
 * @brief Estimate the memory usage of a table.
 *
 * @param tbl The table to estimate the memory usage of.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return The estimated memory usage of the table.
 */
std::size_t estimated_memory_usage(cudf::table_view const& tbl, rmm::cuda_stream_view stream);

}  // namespace cudf_streaming
