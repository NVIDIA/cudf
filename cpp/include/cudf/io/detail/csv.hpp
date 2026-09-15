/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/csv.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace io::detail::csv {

/**
 * @brief Reads the entire dataset.
 *
 * @param source Input `datasource` object to read the dataset from
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 *
 * @return The set of columns along with table metadata
 */
table_with_metadata read_csv(std::unique_ptr<cudf::io::datasource>&& source,
                             csv_reader_options const& options,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr);

/**
 * @brief Write an entire dataset to CSV format.
 *
 * @param sink Output sink
 * @param table The set of columns
 * @param column_names Column names for the output CSV
 * @param options Settings for controlling behavior
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void write_csv(data_sink* sink,
               table_view const& table,
               host_span<std::string const> column_names,
               csv_writer_options const& options,
               cuda::stream_ref stream);

}  // namespace io::detail::csv
}  // namespace cudf
