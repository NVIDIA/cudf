/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <vector>

/**
 * @file random_table_generator.cuh
 * @brief Random table and column generation utilities for external sorting example
 *
 * This header provides utilities to generate random tables and columns with various
 * data types for testing and demonstration purposes. The implementation uses thrust
 * algorithms for GPU-accelerated generation.
 */

namespace cudf {
namespace examples {

/**
 * @brief Generate a table with n_columns random columns and m_rows
 *
 * Creates a table with a mix of different data types to demonstrate libcudf's
 * type system and provide realistic test data. The column types cycle through:
 * INT32, FLOAT64, INT64, FLOAT32, INT16.
 *
 * Each column type uses different value ranges to create diverse datasets:
 * - INT32: Large integers with column-specific offsets
 * - FLOAT64: Double precision floats in range [0, 1000+offset]
 * - INT64: Large long integers around 1e9-1e10 range
 * - FLOAT32: Single precision floats with small ranges
 * - INT16: Short integers with moderate ranges
 *
 * @param n_columns Number of columns to generate
 * @param m_rows Number of rows in each column
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocations
 * @return Unique pointer to the generated table
 */
std::unique_ptr<cudf::table> generate_random_table(std::vector<cudf::data_type> const& types,
                                                   cudf::size_type n_columns,
                                                   cudf::size_type m_rows,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr);

void write_random_table(std::string const& directory,
                        int num_files,
                        cudf::size_type num_rows,
                        cudf::size_type num_cols,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);

}  // namespace examples
}  // namespace cudf
