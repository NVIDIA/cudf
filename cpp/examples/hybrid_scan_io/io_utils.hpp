/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <future>
#include <limits>
#include <tuple>
#include <utility>
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

/**
 * @brief Fetches dictionary pages, trimming every upper-bound range to exactly one dictionary page
 *
 * Reads each range on the host (capping a range that only bounds its page at `max_upper_bound_size`
 * bytes), measures a real dictionary page with `dictionary_page_length`, and copies only the
 * verified page bytes to the device, leaving an empty span for a chunk with no dictionary page.
 * Positions are preserved so the returned spans stay aligned with `dictionary_page_ranges`, which
 * is what `filter_row_groups_with_dictionary_pages` expects.
 *
 * @param datasource Input datasource
 * @param dictionary_page_ranges Dictionary page ranges from `dictionary_pages_byte_ranges`
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @param max_upper_bound_size Most bytes to read of a range that only bounds its dictionary page
 *
 * @return Owning device buffers and one device span per input range
 */
std::pair<std::vector<rmm::device_buffer>, std::vector<cudf::device_span<uint8_t const>>>
fetch_dictionary_pages(cudf::io::datasource& datasource,
                       cudf::host_span<cudf::io::parquet::experimental::dictionary_page_range const>
                         dictionary_page_ranges,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr,
                       int64_t max_upper_bound_size =
                         cudf::io::parquet::experimental::default_max_dictionary_page_read_size);
