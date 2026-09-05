/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io_utils.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

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

std::pair<std::vector<rmm::device_buffer>, std::vector<cudf::device_span<uint8_t const>>>
fetch_dictionary_pages(cudf::io::datasource& datasource,
                       cudf::host_span<cudf::io::parquet::experimental::dictionary_page_range const>
                         dictionary_page_ranges,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr,
                       int64_t max_upper_bound_size)
{
  using cudf::io::parquet::experimental::dictionary_page_extent;

  auto const read_ranges = cudf::io::parquet::experimental::dictionary_page_byte_ranges_to_read(
    dictionary_page_ranges, max_upper_bound_size);

  auto buffers = std::vector<rmm::device_buffer>{};
  auto spans   = std::vector<cudf::device_span<uint8_t const>>{};
  buffers.reserve(read_ranges.size());
  spans.reserve(read_ranges.size());

  // Keep the host reads alive until the async copies below have completed.
  auto host_reads = std::vector<std::unique_ptr<cudf::io::datasource::buffer>>{};

  for (std::size_t i = 0; i < read_ranges.size(); ++i) {
    auto const read_size = read_ranges[i].size();

    // A chunk the reader will not prune with has an empty range; keep an empty span in its place.
    if (read_size == 0) {
      buffers.emplace_back();
      spans.emplace_back();
      continue;
    }

    auto host_buffer = datasource.host_read(read_ranges[i].offset(), read_size);
    auto const bytes = cudf::host_span<uint8_t const>{host_buffer->data(), host_buffer->size()};

    // An exact range is already one page; an upper-bound range has to be measured and trimmed, and
    // may hold no dictionary page at all.
    auto const page_size =
      (dictionary_page_ranges[i].extent == dictionary_page_extent::upper_bound_if_present)
        ? cudf::io::parquet::experimental::dictionary_page_length(bytes).value_or(0)
        : static_cast<int64_t>(bytes.size());

    if (page_size == 0) {
      buffers.emplace_back();
      spans.emplace_back();
      continue;
    }

    auto const page_bytes = static_cast<std::size_t>(page_size);
    auto device_buffer    = rmm::device_buffer{bytes.data(), page_bytes, stream, mr};
    spans.emplace_back(static_cast<uint8_t const*>(device_buffer.data()), page_bytes);
    buffers.emplace_back(std::move(device_buffer));
    host_reads.emplace_back(std::move(host_buffer));
  }

  stream.sync();  // host_reads are freed on return, so the copies must finish first
  return {std::move(buffers), std::move(spans)};
}
