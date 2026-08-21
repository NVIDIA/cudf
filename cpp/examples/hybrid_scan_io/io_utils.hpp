/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

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
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);

/**
 * @brief Owns the datasources, footer buffers, and byte spans for a multifile read.
 *
 * Call `fetch_footers()` after construction. Keeping datasource and footer setup separate allows
 * examples to time those operations independently.
 */
struct multifile_inputs {
  explicit multifile_inputs(cudf::io::source_info const& source_info);

  void fetch_footers();

  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  std::vector<std::reference_wrapper<cudf::io::datasource>> datasource_refs;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footer_buffers;
  std::vector<cudf::host_span<uint8_t const>> footer_byte_spans;
};

/**
 * @brief Owns multifile device buffers and the corresponding per-source and flattened spans.
 */
struct multisource_device_data {
  std::vector<rmm::device_buffer> buffers;
  std::vector<std::vector<cudf::device_span<uint8_t const>>> per_source_spans;
  std::vector<cudf::device_span<uint8_t const>> flat_spans;
};

/**
 * @brief Regroups flattened byte ranges using the source map returned by Hybrid Scan.
 */
[[nodiscard]] std::vector<std::vector<cudf::io::text::byte_range_info>> group_byte_ranges_by_source(
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  std::size_t num_sources);

/**
 * @brief Fetches source-mapped multifile byte ranges and flattens the resulting device spans.
 */
[[nodiscard]] multisource_device_data fetch_multisource_device_data(
  multifile_inputs const& inputs,
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Fetches source-grouped multifile byte ranges.
 */
[[nodiscard]] multisource_device_data fetch_multisource_device_data(
  multifile_inputs const& inputs,
  std::vector<std::vector<cudf::io::text::byte_range_info>> const& byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
