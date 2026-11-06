/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

/**
 * @file io_utils.cpp
 * @brief Definitions for IO utilities for hybrid_scan examples
 */

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_bytes(cudf::io::datasource& datasource)
{
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_footer_to_host(datasource);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_bytes(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes)
{
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_page_index_to_host(datasource, page_index_bytes);
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<cudf::device_span<uint8_t const>>,
           std::future<void>>
fetch_byte_ranges_async(cudf::io::datasource& datasource,
                        cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
                        cuda::stream_ref stream,
                        rmm::device_async_resource_ref mr)
{
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_byte_ranges_to_device_async(
    datasource, byte_ranges, cudf::io::parquet::io_submission_policy::SERIALIZE, stream, mr);
}
