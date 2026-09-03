/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_composer.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/aligned_resource_adaptor.hpp>

#include <unordered_set>
#include <vector>

/**
 * @file hybrid_scan_composer.cpp
 * @brief Definitions for hybrid scan composer function(s)
 */

namespace {

using cudf::io::parquet::experimental::hybrid_scan_reader;

std::unique_ptr<hybrid_scan_reader> setup_reader(cudf::io::datasource& datasource,
                                                 cudf::io::parquet_reader_options const& options)
{
  // Fetch footer bytes and setup reader
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(datasource);
  auto reader              = std::make_unique<hybrid_scan_reader>(
    cudf::host_span<uint8_t const>{static_cast<uint8_t const*>(footer_buffer->data()),
                                                footer_buffer->size()},
    options);

  auto const page_index_byte_range = reader->page_index_byte_range();
  if (not page_index_byte_range.is_empty()) {
    auto const page_index_buffer =
      cudf::io::parquet::fetch_page_index_to_host(datasource, page_index_byte_range);
    reader->setup_page_index(cudf::host_span<uint8_t const>{
      static_cast<uint8_t const*>(page_index_buffer->data()), page_index_buffer->size()});
  }
  return reader;
}

std::vector<cudf::size_type> apply_row_group_filters(
  cudf::io::datasource& datasource,
  hybrid_scan_reader const& reader,
  cudf::host_span<cudf::size_type> input_row_group_indices,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  cudf::io::parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // Span to track current row group indices
  auto current_row_group_indices = input_row_group_indices;

  // Filter row groups with stats
  auto stats_filtered_row_groups = std::vector<cudf::size_type>{};
  stats_filtered_row_groups.reserve(current_row_group_indices.size());

  if (filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_STATS)) {
    stats_filtered_row_groups =
      reader.filter_row_groups_with_stats(current_row_group_indices, options, stream);
    current_row_group_indices = stats_filtered_row_groups;
  }

  // Filter row groups with dictionary pages
  auto dict_page_filtered_row_groups = std::vector<cudf::size_type>{};
  dict_page_filtered_row_groups.reserve(current_row_group_indices.size());

  if (filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_DICT_PAGES)) {
    auto const dict_page_byte_ranges =
      reader.dictionary_pages_byte_ranges(current_row_group_indices, options);

    if (not dict_page_byte_ranges.empty()) {
      auto [dictionary_page_buffers, dictionary_page_data, dict_read_tasks] =
        cudf::io::parquet::fetch_byte_ranges_to_device_async(
          datasource,
          dict_page_byte_ranges,
          cudf::io::parquet::io_submission_policy::SERIALIZE,
          stream,
          mr);
      dict_read_tasks.get();

      dict_page_filtered_row_groups = reader.filter_row_groups_with_dictionary_pages(
        dictionary_page_data, current_row_group_indices, options, stream);

      current_row_group_indices = dict_page_filtered_row_groups;
    }
  }

  // Filter row groups with bloom filters
  auto bloom_filtered_row_groups = std::vector<cudf::size_type>{};
  bloom_filtered_row_groups.reserve(current_row_group_indices.size());

  if (filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS)) {
    auto const bloom_filter_byte_ranges =
      reader.bloom_filters_byte_ranges(current_row_group_indices, options);

    if (not bloom_filter_byte_ranges.empty()) {
      // Fetch 32-byte aligned bloom filter data buffers from the input file buffer
      auto constexpr bloom_filter_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
      auto aligned_mr = rmm::mr::aligned_resource_adaptor(mr, bloom_filter_alignment);
      auto [bloom_filter_buffers, bloom_filter_data, bloom_read_tasks] =
        cudf::io::parquet::fetch_byte_ranges_to_device_async(
          datasource,
          bloom_filter_byte_ranges,
          cudf::io::parquet::io_submission_policy::SERIALIZE,
          stream,
          aligned_mr);
      bloom_read_tasks.get();

      bloom_filtered_row_groups = reader.filter_row_groups_with_bloom_filters(
        bloom_filter_data, current_row_group_indices, options, stream);

      current_row_group_indices = bloom_filtered_row_groups;
    }
  }

  return std::vector<cudf::size_type>(current_row_group_indices.begin(),
                                      current_row_group_indices.end());
}

std::unique_ptr<cudf::table> single_step_materialize(
  cudf::io::datasource& datasource,
  hybrid_scan_reader const& reader,
  cudf::host_span<cudf::size_type> current_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const all_column_chunk_byte_ranges =
    reader.all_column_chunks_byte_ranges(current_row_group_indices, options);
  auto [all_column_chunk_buffers, all_column_chunk_data, all_column_chunk_read_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      datasource,
      all_column_chunk_byte_ranges,
      cudf::io::parquet::io_submission_policy::SERIALIZE,
      stream,
      mr);
  all_column_chunk_read_tasks.get();

  return reader
    .materialize_all_columns(current_row_group_indices, all_column_chunk_data, options, stream, mr)
    .tbl;
}

}  // namespace

std::unique_ptr<cudf::table> hybrid_scan(cudf::io::parquet_reader_options const& options,
                                         std::unordered_set<hybrid_scan_filter_type> const& filters,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  // Input file buffer span
  auto const io_source = options.get_source();
  auto datasource      = std::move(cudf::io::make_datasources(io_source).front());
  auto datasource_ref  = std::ref(*datasource);

  // Setup reader and page index
  auto reader           = setup_reader(datasource_ref, options);
  auto const reader_ref = std::cref(*reader);

  // Start with all row groups
  auto row_group_indices         = reader->all_row_groups(options);
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(row_group_indices);

  // Filter row groups
  if (options.get_filter().has_value()) {
    row_group_indices = apply_row_group_filters(
      datasource_ref, reader_ref, current_row_group_indices, filters, options, stream, mr);
    current_row_group_indices = cudf::host_span<cudf::size_type>(row_group_indices);
  }

  // Materialize table in single step
  return single_step_materialize(
    datasource_ref, reader_ref, current_row_group_indices, options, stream, mr);
}
