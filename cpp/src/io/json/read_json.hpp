/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace io::json::detail {

// Some magic numbers
constexpr int num_subchunks            = 10;  // per chunk_size
constexpr size_t min_subchunk_size     = 10000;
constexpr int max_subchunks_prealloced = 3;

/**
 * @brief Read from array of data sources into RMM buffer. The size of the returned device span
          can be larger than the number of bytes requested from the list of sources when
          the range to be read spans across multiple sources. This is due to the delimiter
          characters inserted after the end of each accessed source.
 *
 * @param buffer Device span buffer to which data is read
 * @param sources Array of data sources
 * @param range_offset Number of bytes to skip from source start
 * @param range_size Number of bytes to read from source
 * @param delimiter Delimiter character for JSONL inputs
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @returns A subspan of the input device span containing data read
 */
device_span<char> ingest_raw_input(device_span<char> buffer,
                                   host_span<std::unique_ptr<datasource>> sources,
                                   size_t range_offset,
                                   size_t range_size,
                                   char delimiter,
                                   cuda::stream_ref stream);

/**
 * @brief Reads and returns the entire data set in batches.
 *
 * @param sources Input `datasource` objects to read the dataset from
 * @param reader_opts Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 *
 * @return cudf::table object that contains the array of cudf::column.
 */
table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

}  // namespace io::json::detail
}  // namespace CUDF_EXPORT cudf
