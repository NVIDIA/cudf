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
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cudf {
namespace examples {

/**
 * @brief Sample splitters from a sorted table
 *
 * Extracts n equally spaced rows from the first column of a sorted table
 * to use as splitters for external sorting.
 *
 * @param sorted_table The table sorted by first column
 * @param num_splitters Number of splitters to extract
 * @param stream CUDA stream for operations
 * @return Column containing the sampled splitter values
 */
std::unique_ptr<cudf::column> sample_splitters(cudf::table_view const& tv,
                                               cudf::size_type num_splitters,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

}  // namespace examples
}  // namespace cudf
