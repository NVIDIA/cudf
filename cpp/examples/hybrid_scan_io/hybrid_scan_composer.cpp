/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_composer.hpp"

#include "common_utils.hpp"
#include "io_utils.hpp"
#include "timer.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/aligned_resource_adaptor.hpp>

#include <nvtx3/nvtx3.hpp>

#include <unordered_set>

/**
 * @file hybrid_scan_commons.cpp
 * @brief Definitions for common hybrid scan related functions for hybrid_scan examples
 */

using cudf::io::parquet::experimental::hybrid_scan_reader;

namespace {

/**
 * @brief Sets up the a hybrid scan reader instance
 *
 * @param datasource Data source
 * @param options Parquet reader options
 * @param verbose Whether to print verbose output
 *
 * @return Hybrid scan reader
 */
std::unique_ptr<hybrid_scan_reader> setup_reader(cudf::io::datasource& datasource,
                                                 cudf::io::parquet_reader_options const& options,
                                                 bool verbose)
{
  if (verbose) { std::cout << "READER: Setup reader...\n"; }

  timer timer;
  // Fetch footer bytes and setup reader
  auto const footer_buffer = fetch_footer_bytes(datasource);
  auto reader              = std::make_unique<hybrid_scan_reader>(*footer_buffer, options);
  if (verbose) { timer.print_elapsed_millis(); }

  return std::move(reader);
}

/**
 * @brief Sets up the page index for the hybrid scan reader
 *
 * @param datasource Data source
 * @param reader Hybrid scan reader
 * @param single_step_read Whether to use single step read mode
 * @param verbose Whether to print verbose output
 */
void setup_page_index(cudf::io::datasource& datasource,
                      hybrid_scan_reader const& reader,
                      bool single_step_read,
                      bool verbose)
{
  if (verbose) { std::cout << "READER: Setup page index...\n"; }

  timer timer;
  auto const page_index_byte_range = reader.page_index_byte_range();
  CUDF_EXPECTS(
    not page_index_byte_range.is_empty() or single_step_read,
    "Parquet source does not contain a page index, needed by the Hybrid Scan for two-step read");
  if (not page_index_byte_range.is_empty()) {
    auto const page_index_buffer = fetch_page_index_bytes(datasource, page_index_byte_range);
    reader.setup_page_index(*page_index_buffer);
  }
  if (verbose) { timer.print_elapsed_millis(); }
}

/**
 * @brief Applies specified row group filters
 *
 * @param datasource Data source
 * @param reader Hybrid scan reader
 * @param filters Set of hybrid scanfilters to apply
 * @param input_row_group_indices Span of input row group indices
 * @param options Parquet reader options
 * @param verbose Whether to print verbose output
 * @param stream CUDA stream
 *
 * @return Filtered row group indices
 */
std::vector<cudf::size_type> apply_row_group_filters(
  cudf::io::datasource& datasource,
  hybrid_scan_reader const& reader,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  cudf::host_span<cudf::size_type> input_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  bool verbose,
  cuda::stream_ref stream)
{
  // Span to track current row group indices
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);

  if (verbose) {
    std::cout << "Input row group indices size: " << current_row_group_indices.size() << "\n";
  }

  timer timer;

  // Filter row groups with stats
  auto stats_filtered_row_group_indices = std::vector<cudf::size_type>{};
  if (filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_STATS)) {
    if (verbose) { std::cout << "READER: Filter row groups with stats...\n"; }

    timer.reset();
    stats_filtered_row_group_indices =
      reader.filter_row_groups_with_stats(current_row_group_indices, options, stream);

    current_row_group_indices = stats_filtered_row_group_indices;
    if (verbose) {
      std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
      timer.print_elapsed_millis();
    }
  }

  // Get dictionary page byte ranges from the reader
  std::vector<cudf::io::text::byte_range_info> dict_page_byte_ranges;
  if (filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_DICT_PAGES)) {
    if (verbose) { std::cout << "READER: Get dictionary page byte ranges...\n"; }
    timer.reset();
    dict_page_byte_ranges = reader.dictionary_pages_byte_ranges(current_row_group_indices, options);
    if (verbose) { timer.print_elapsed_millis(); }
  }

  auto temp_mr = cudf::get_current_device_resource_ref();

  // Filter row groups with dictionary pages
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (not dict_page_byte_ranges.empty()) {
    if (verbose) { std::cout << "READER: Filter row groups with dictionary pages...\n"; }
    timer.reset();

    // Fetch dictionary page buffers and corresponding device spans from the input file buffer
    nvtxRangePush("fetch_dict_page_byte_ranges");
    auto [dictionary_page_buffers, dictionary_page_data, dict_read_tasks] =
      fetch_byte_ranges_async(datasource, dict_page_byte_ranges, stream, temp_mr);
    dict_read_tasks.get();
    nvtxRangePop();

    dictionary_page_filtered_row_group_indices = reader.filter_row_groups_with_dictionary_pages(
      dictionary_page_data, current_row_group_indices, options, stream);

    current_row_group_indices = dictionary_page_filtered_row_group_indices;
    if (verbose) {
      std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
      timer.print_elapsed_millis();
    }
  } else if (verbose) {
    std::cout << "SKIP: Row group filtering with dictionary pages...\n\n";
  }

  // Get bloom filter byte ranges from the reader
  std::vector<cudf::io::text::byte_range_info> bloom_filter_byte_ranges;
  if (filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS)) {
    if (verbose) { std::cout << "READER: Get bloom filter byte ranges...\n"; }
    timer.reset();
    bloom_filter_byte_ranges = reader.bloom_filters_byte_ranges(current_row_group_indices, options);
    if (verbose) { timer.print_elapsed_millis(); }
  }

  // Filter row groups with bloom filters
  std::vector<cudf::size_type> bloom_filtered_row_group_indices;
  bloom_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (not bloom_filter_byte_ranges.empty()) {
    // Fetch 32-byte aligned bloom filter data buffers from the input file buffer
    auto constexpr bloom_filter_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
    auto aligned_mr = rmm::mr::aligned_resource_adaptor(temp_mr, bloom_filter_alignment);
    if (verbose) { std::cout << "READER: Filter row groups with bloom filters...\n"; }
    timer.reset();
    nvtxRangePush("fetch_bloom_filter_byte_ranges");
    auto [bloom_filter_buffers, bloom_filter_data, bloom_read_tasks] =
      fetch_byte_ranges_async(datasource, bloom_filter_byte_ranges, stream, aligned_mr);
    bloom_read_tasks.get();
    nvtxRangePop();

    bloom_filtered_row_group_indices = reader.filter_row_groups_with_bloom_filters(
      bloom_filter_data, current_row_group_indices, options, stream);

    current_row_group_indices = bloom_filtered_row_group_indices;
    if (verbose) {
      std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
      timer.print_elapsed_millis();
    }
  } else if (verbose) {
    std::cout << "SKIP: Row group filtering with bloom filters...\n\n";
  }

  if (verbose) {
    std::cout << "Filtered row group indices size: " << current_row_group_indices.size() << "\n";
    timer.print_elapsed_millis();
  }

  return std::vector<cudf::size_type>(current_row_group_indices.begin(),
                                      current_row_group_indices.end());
}

/**
 * @brief Materializes all parquet columns in single step mode
 *
 * @param datasource Data source
 * @param reader Hybrid scan reader
 * @param current_row_group_indices Span of current row group indices
 * @param options Parquet reader options
 * @param verbose Whether to print verbose output
 * @param stream CUDA stream
 * @param mr Device memory resource to allocate memory for the read table
 *
 * @return Unique pointer to the read table
 */
std::unique_ptr<cudf::table> single_step_materialize(
  cudf::io::datasource& datasource,
  hybrid_scan_reader const& reader,
  cudf::host_span<cudf::size_type> current_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  bool verbose,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (verbose) { std::cout << "READER: Single step materialize...\n"; }

  timer timer;

  auto const all_column_chunk_byte_ranges =
    reader.all_column_chunks_byte_ranges(current_row_group_indices, options);

  nvtxRangePush("fetch_all_col_byte_ranges");
  auto [all_column_chunk_buffers, all_column_chunk_data, all_column_chunk_read_tasks] =
    fetch_byte_ranges_async(datasource, all_column_chunk_byte_ranges, stream, mr);
  all_column_chunk_read_tasks.get();
  nvtxRangePop();

  auto read_table = reader
                      .materialize_all_columns(
                        current_row_group_indices, all_column_chunk_data, options, stream, mr)
                      .tbl;

  if (verbose) { timer.print_elapsed_millis(); }

  return std::move(read_table);
}

/**
 * @brief Materializes all parquet columns in two step mode
 *
 * @param datasource Data source
 * @param reader Hybrid scan reader
 * @param filters Set of hybrid scan filters to apply
 * @param current_row_group_indices Span of current row group indices
 * @param options Parquet reader options
 * @param verbose Whether to print verbose output
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Unique pointer to the read table
 */
std::unique_ptr<cudf::table> two_step_materialize(
  cudf::io::datasource& datasource,
  hybrid_scan_reader const& reader,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  cudf::host_span<cudf::size_type> current_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  bool verbose,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // Check whether to prune filter column data pages
  using cudf::io::parquet::experimental::use_data_page_mask;
  auto const prune_filter_data_pages =
    filters.contains(hybrid_scan_filter_type::FILTER_COLUMN_PAGES_WITH_PAGE_INDEX);

  if (verbose) {
    if (prune_filter_data_pages) {
      std::cout << "READER: Filter data pages of filter columns with page index stats...\n";
    } else {
      std::cout << "SKIP: Filter column data page filtering with page index stats...\n\n";
    }
  }

  timer timer;

  // Build initial row mask
  auto row_mask = [&]() {
    if (prune_filter_data_pages) {
      return reader.build_row_mask_with_page_index_stats(
        current_row_group_indices, options, stream, mr);
    } else {
      return reader.build_all_true_row_mask(current_row_group_indices, stream, mr);
    }
  }();
  if (verbose) {
    timer.print_elapsed_millis();
    std::cout << "READER: Materialize filter columns...\n";
  }

  timer.reset();

  // Get column chunk byte ranges from the reader
  auto const filter_column_chunk_byte_ranges =
    reader.filter_column_chunks_byte_ranges(current_row_group_indices, options);
  nvtxRangePush("fetch_filter_col_byte_ranges");
  auto [filter_column_chunk_buffers, filter_column_chunk_data, filter_col_read_tasks] =
    fetch_byte_ranges_async(datasource, filter_column_chunk_byte_ranges, stream, mr);
  filter_col_read_tasks.get();
  nvtxRangePop();

  auto row_mask_mutable_view = row_mask->mutable_view();
  auto filter_table =
    reader
      .materialize_filter_columns(
        current_row_group_indices,
        filter_column_chunk_data,
        row_mask_mutable_view,
        prune_filter_data_pages ? use_data_page_mask::YES : use_data_page_mask::NO,
        options,
        stream,
        mr)
      .tbl;
  if (verbose) { timer.print_elapsed_millis(); }

  // Check whether to prune payload column data pages
  auto const prune_payload_data_pages =
    filters.contains(hybrid_scan_filter_type::PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK);

  if (verbose) {
    std::cout << "READER: Materialize payload columns...\n";
    if (prune_payload_data_pages) {
      std::cout << "READER: Filter data pages of payload columns with row mask...\n";
    } else {
      std::cout << "SKIP: Payload column data page filtering with row mask...\n\n";
    }
  }

  timer.reset();

  // Get column chunk byte ranges from the reader
  auto const payload_column_chunk_byte_ranges =
    reader.payload_column_chunks_byte_ranges(current_row_group_indices, options);
  nvtxRangePush("fetch_payload_col_byte_ranges");
  auto [payload_column_chunk_buffers, payload_column_chunk_data, payload_col_read_tasks] =
    fetch_byte_ranges_async(datasource, payload_column_chunk_byte_ranges, stream, mr);
  payload_col_read_tasks.get();
  nvtxRangePop();

  auto payload_table =
    reader
      .materialize_payload_columns(
        current_row_group_indices,
        payload_column_chunk_data,
        row_mask->view(),
        prune_payload_data_pages ? use_data_page_mask::YES : use_data_page_mask::NO,
        options,
        stream,
        mr)
      .tbl;

  if (verbose) { timer.print_elapsed_millis(); }

  return combine_tables(std::move(filter_table), std::move(payload_table));
}

}  // namespace

template <bool single_step_read, bool use_page_index>
std::unique_ptr<cudf::table> hybrid_scan(
  io_source const& io_source,
  std::optional<cudf::ast::operation const> filter_expression,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  bool verbose,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const has_filter_expr = filter_expression.has_value();
  if (not single_step_read and not has_filter_expr) {
    throw std::runtime_error("Filter expression must be provided for two-step hybrid scan");
  }

  auto options = cudf::io::parquet_reader_options::builder().build();
  if (has_filter_expr) { options.set_filter(filter_expression.value()); }

  // Input file buffer span
  auto datasource     = std::move(cudf::io::make_datasources(io_source.get_source_info()).front());
  auto datasource_ref = std::ref(*datasource);

  // Setup reader
  auto reader           = setup_reader(datasource_ref, options, verbose);
  auto const reader_ref = std::cref(*reader);

  // Setup page index if needed
  if constexpr (use_page_index) {
    setup_page_index(datasource_ref, reader_ref, single_step_read, verbose);
  }

  // Start with all row groups
  auto row_group_indices         = reader->all_row_groups(options);
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(row_group_indices);

  // Filter row groups
  if (has_filter_expr) {
    row_group_indices = apply_row_group_filters(
      datasource_ref, reader_ref, filters, current_row_group_indices, options, verbose, stream);
    current_row_group_indices = cudf::host_span<cudf::size_type>(row_group_indices);
  }

  // Materialize filter and payload columns separately
  if constexpr (single_step_read) {
    return single_step_materialize(
      datasource_ref, reader_ref, current_row_group_indices, options, verbose, stream, mr);
  } else {
    return two_step_materialize(
      datasource_ref, reader_ref, filters, current_row_group_indices, options, verbose, stream, mr);
  }
}

// Specialization for two-step read without page index
template <bool single_step_read, bool use_page_index>
  requires(not single_step_read and not use_page_index)
std::unique_ptr<cudf::table> inline hybrid_scan(
  io_source const& io_source,
  std::optional<cudf::ast::operation const> filter_expression,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  bool verbose,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  static_assert(single_step_read or use_page_index,
                "Hybrid scan requires parquet page index for two-step parquet read");
  return nullptr;
}

// Instantiations for hybrid_scan template

template std::unique_ptr<cudf::table> hybrid_scan<true, false>(
  io_source const&,
  std::optional<cudf::ast::operation const>,
  std::unordered_set<hybrid_scan_filter_type> const&,
  bool,
  cuda::stream_ref,
  rmm::device_async_resource_ref);

template std::unique_ptr<cudf::table> hybrid_scan<true, true>(
  io_source const&,
  std::optional<cudf::ast::operation const>,
  std::unordered_set<hybrid_scan_filter_type> const&,
  bool,
  cuda::stream_ref,
  rmm::device_async_resource_ref);

template std::unique_ptr<cudf::table> hybrid_scan<false, true>(
  io_source const&,
  std::optional<cudf::ast::operation const>,
  std::unordered_set<hybrid_scan_filter_type> const&,
  bool,
  cuda::stream_ref,
  rmm::device_async_resource_ref);
