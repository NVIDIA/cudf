/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <vector>

// Helper function to test correctness of JSON byte range reading.
// We split the input source files into a set of byte range chunks each of size
// `chunk_size` and return an array of partial tables constructed from each chunk
template <typename IndexType = std::int32_t>
std::vector<cudf::io::table_with_metadata> split_byte_range_reading(
  cudf::host_span<std::unique_ptr<cudf::io::datasource>> sources,
  cudf::host_span<std::unique_ptr<cudf::io::datasource>> csources,
  cudf::io::json_reader_options const& reader_opts,
  cudf::io::json_reader_options const& creader_opts,
  IndexType chunk_size,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/// Returns length of each string in the column
rmm::device_uvector<cudf::size_type> string_offset_to_length(
  cudf::strings_column_view const& column, cuda::stream_ref stream);
