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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <vector>

/**
 * @file parquet_io.hpp
 * @brief Parquet I/O utilities for external sorting example
 *
 * This header provides utilities for reading parquet files and implementing
 * external sorting using sample sort algorithm.
 */

namespace cudf {
namespace examples {

/**
 * @brief Read multiple parquet files sequentially
 *
 * Reads a list of parquet files sequentially without concatenating them.
 * Returns a vector of individual tables.
 *
 * @param filepaths List of parquet file paths to read
 * @param stream CUDA stream for operations
 * @return Vector of tables, one per file
 */
std::unique_ptr<cudf::table> read_parquet_file(std::string const& filepath,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

void write_parquet_file(std::string const& filepath,
                        cudf::table_view table_view,
                        rmm::cuda_stream_view stream);

std::string construct_file_path(std::string const& input_dir, int filenum);

}  // namespace examples
}  // namespace cudf
