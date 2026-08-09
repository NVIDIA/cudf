/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

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
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/// Returns length of each string in the column
rmm::device_uvector<cudf::size_type> string_offset_to_length(
  cudf::strings_column_view const& column, rmm::cuda_stream_view stream);

/// Returns the column's offsets, widened to size_type
///
/// A strings column stores its offsets 32 bits wide until it is large enough to need 64, so the
/// buffer cannot simply be read as size_type.
rmm::device_uvector<cudf::size_type> string_offsets(cudf::strings_column_view const& column,
                                                    rmm::cuda_stream_view stream);
