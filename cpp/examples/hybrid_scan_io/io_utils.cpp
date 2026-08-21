/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io_utils.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

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
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_byte_ranges_to_device_async(datasource, byte_ranges, stream, mr);
}

multifile_inputs::multifile_inputs(cudf::io::source_info const& source_info)
  : datasources{cudf::io::make_datasources(source_info)}
{
  datasource_refs.reserve(datasources.size());
  std::transform(datasources.begin(),
                 datasources.end(),
                 std::back_inserter(datasource_refs),
                 [](auto const& datasource) { return std::ref(*datasource); });
}

void multifile_inputs::fetch_footers()
{
  footer_buffers = cudf::io::parquet::fetch_footers_to_host(datasource_refs);
  footer_byte_spans.clear();
  footer_byte_spans.reserve(footer_buffers.size());
  std::transform(footer_buffers.begin(),
                 footer_buffers.end(),
                 std::back_inserter(footer_byte_spans),
                 [](auto const& buffer) { return cudf::host_span<uint8_t const>{*buffer}; });
}

std::vector<std::vector<cudf::io::text::byte_range_info>> group_byte_ranges_by_source(
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  std::size_t num_sources)
{
  auto const& [byte_ranges, source_map] = byte_ranges_and_source_map;
  CUDF_EXPECTS(byte_ranges.size() == source_map.size(), "Invalid source map size");

  auto byte_ranges_per_source =
    std::vector<std::vector<cudf::io::text::byte_range_info>>(num_sources);
  for (auto range_index = std::size_t{0}; range_index < byte_ranges.size(); ++range_index) {
    auto const source_index = source_map[range_index];
    CUDF_EXPECTS(
      source_index >= 0 and static_cast<std::size_t>(source_index) < byte_ranges_per_source.size(),
      "Invalid source index");
    byte_ranges_per_source[source_index].push_back(byte_ranges[range_index]);
  }
  return byte_ranges_per_source;
}

multisource_device_data fetch_multisource_device_data(
  multifile_inputs const& inputs,
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const byte_ranges_per_source =
    group_byte_ranges_by_source(byte_ranges_and_source_map, inputs.datasources.size());
  return fetch_multisource_device_data(inputs, byte_ranges_per_source, stream, mr);
}

multisource_device_data fetch_multisource_device_data(
  multifile_inputs const& inputs,
  std::vector<std::vector<cudf::io::text::byte_range_info>> const& byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto [buffers, per_source_spans, tasks] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    inputs.datasource_refs, byte_ranges_per_source, stream, mr);
  tasks.get();

  auto flat_spans = std::vector<cudf::device_span<uint8_t const>>{};
  for (auto const& source_spans : per_source_spans) {
    flat_spans.insert(flat_spans.end(), source_spans.begin(), source_spans.end());
  }
  return {std::move(buffers), std::move(per_source_spans), std::move(flat_spans)};
}
