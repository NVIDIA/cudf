/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/avro.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace io::detail::avro {

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
table_with_metadata read_avro(std::unique_ptr<cudf::io::datasource>&& source,
                              avro_reader_options const& options,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

}  // namespace io::detail::avro
}  // namespace cudf
