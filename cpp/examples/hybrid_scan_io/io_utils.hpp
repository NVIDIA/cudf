/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <future>
#include <tuple>
#include <vector>

/**
 * @file io_utils.hpp
 * @brief IO utilities for hybrid_scan examples
 */

/**
 * @brief Fetches a host buffer of Parquet footer bytes from the input data source
 *
 * @param datasource Input data source
 * @return Host buffer containing footer bytes
 */
std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_bytes(cudf::io::datasource& datasource);

/**
 * @brief Fetches a host buffer of Parquet page index from the input data source
 *
 * @param datasource Input datasource
 * @param page_index_bytes Byte range of page index
 * @return Host buffer containing page index bytes
 */
std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_bytes(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes);

/**
 * @brief Fetches a list of byte ranges from a host buffer into device buffers
 *
 * @param datasource Input datasource
 * @param byte_ranges Byte ranges to fetch
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return A tuple containing the device buffers, the device spans of the fetched data, and a future
 * to wait on the read tasks
 */
std::tuple<std::vector<rmm::device_buffer>,
           std::vector<cudf::device_span<uint8_t const>>,
           std::future<void>>
fetch_byte_ranges_async(cudf::io::datasource& datasource,
                        cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
                        cuda::stream_ref stream,
                        rmm::device_async_resource_ref mr);
