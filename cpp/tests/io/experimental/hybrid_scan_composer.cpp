/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_composer.hpp"

#include "hybrid_scan_common.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/aligned_resource_adaptor.hpp>

#include <vector>

using cudf::io::parquet::experimental::hybrid_scan_reader;

namespace {

/**
 * @brief Sets up the a hybrid scan reader instance and page index if present
 *
 * @param datasource Data source
 * @param options Parquet reader options
 * @param verbose Whether to print verbose output
 *
 * @return Hybrid scan reader
 */

std::unique_ptr<hybrid_scan_reader> setup_reader(cudf::io::datasource& datasource,
                                                 cudf::io::parquet_reader_options const& options)
{
  // Fetch footer bytes and setup reader
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(datasource);
  auto reader              = std::make_unique<hybrid_scan_reader>(*footer_buffer, options);

  auto const page_index_byte_range = reader->page_index_byte_range();
  if (not page_index_byte_range.is_empty()) {
    auto const page_index_buffer =
      cudf::io::parquet::fetch_page_index_to_host(datasource, page_index_byte_range);
    reader->setup_page_index(*page_index_buffer);
  }

  return reader;
}

/**
 * @brief Apply hybrid scan row group filters
 *
 * @param datasource Input datasource
 * @param options Reader options
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Filtered row group indices
 */
auto apply_hybrid_scan_filters(cudf::io::datasource& datasource,
                               hybrid_scan_reader const& reader,
                               cudf::io::parquet_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  // Get all row groups from the reader
  auto input_row_group_indices = reader.all_row_groups(options);

  // Span to track current row group indices
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);

  // Filter row groups with stats
  auto stats_filtered_row_group_indices =
    reader.filter_row_groups_with_stats(current_row_group_indices, options, stream);

  // Update current row group indices
  current_row_group_indices = stats_filtered_row_group_indices;

  // Get bloom filter and dictionary page byte ranges from the reader
  auto [bloom_filter_byte_ranges, dict_page_byte_ranges] =
    reader.secondary_filters_byte_ranges(current_row_group_indices, options);

  // If we have dictionary page byte ranges, filter row groups with dictionary pages
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (dict_page_byte_ranges.size()) {
    // Fetch dictionary page buffers from the input file buffer
    auto [dict_page_buffers, dict_page_data, dict_read_tasks] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        datasource, dict_page_byte_ranges, stream, mr);
    dict_read_tasks.get();

    // Filter row groups with dictionary pages
    dictionary_page_filtered_row_group_indices = reader.filter_row_groups_with_dictionary_pages(
      dict_page_data, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = dictionary_page_filtered_row_group_indices;
  }

  // If we have bloom filter byte ranges, filter row groups with bloom filters
  std::vector<cudf::size_type> bloom_filtered_row_group_indices;
  bloom_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (bloom_filter_byte_ranges.size()) {
    // Fetch 32 byte aligned bloom filter data buffers from the input file buffer
    auto aligned_mr = rmm::mr::aligned_resource_adaptor(cudf::get_current_device_resource_ref(),
                                                        bloom_filter_alignment);

    auto [bloom_filter_buffers, bloom_filter_data, bloom_read_tasks] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        datasource, bloom_filter_byte_ranges, stream, aligned_mr);
    bloom_read_tasks.get();

    // Filter row groups with bloom filters
    bloom_filtered_row_group_indices = reader.filter_row_groups_with_bloom_filters(
      bloom_filter_data, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = bloom_filtered_row_group_indices;
  }

  return std::vector<cudf::size_type>(current_row_group_indices.begin(),
                                      current_row_group_indices.end());
}

}  // namespace

std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> hybrid_scan(
  cudf::io::datasource& datasource,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& payload_column_names,
  bool case_sensitive_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  rmm::mr::aligned_resource_adaptor& aligned_mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder()
                                               .filter(filter_expression)
                                               .case_sensitive_names(case_sensitive_names);

  // Set payload column names if provided
  if (payload_column_names.has_value()) { options.set_column_names(payload_column_names.value()); }

  auto const reader = setup_reader(datasource, options);
  auto reader_ref   = std::ref(*reader);

  auto const filtered_row_group_indices =
    apply_hybrid_scan_filters(datasource, reader_ref, options, stream, mr);

  auto const current_row_group_indices =
    cudf::host_span<cudf::size_type const>(filtered_row_group_indices);

  // Build a row mask for the filtered row groups using page index stats, or all true if no filter
  auto row_mask =
    options.get_filter().has_value()
      ? reader->build_row_mask_with_page_index_stats(current_row_group_indices, options, stream, mr)
      : reader->build_all_true_row_mask(current_row_group_indices, stream, mr);

  // Get column chunk byte ranges from the reader
  auto const filter_column_chunk_byte_ranges =
    reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers and spans from the input buffer
  auto [filter_col_buffers, filter_col_data, filter_col_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      datasource, filter_column_chunk_byte_ranges, stream, mr);
  filter_col_tasks.get();

  // Materialize the table with only the filter columns
  auto row_mask_view = row_mask->mutable_view();
  auto [filter_table, filter_metadata] =
    reader->materialize_filter_columns(current_row_group_indices,
                                       filter_col_data,
                                       row_mask_view,
                                       cudf::io::parquet::experimental::use_data_page_mask::YES,
                                       options,
                                       stream,
                                       mr);

  // Get column chunk byte ranges from the reader
  auto const payload_column_chunk_byte_ranges =
    reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers and spans from the input buffer
  auto [payload_col_buffers, payload_col_data, payload_col_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      datasource, payload_column_chunk_byte_ranges, stream, mr);
  payload_col_tasks.get();

  // Materialize the table with only the payload columns
  auto [payload_table, payload_metadata] =
    reader->materialize_payload_columns(current_row_group_indices,
                                        payload_col_data,
                                        row_mask_view,
                                        cudf::io::parquet::experimental::use_data_page_mask::YES,
                                        options,
                                        stream,
                                        mr);

  return std::tuple{std::move(filter_table), std::move(payload_table)};
}

std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> chunked_hybrid_scan(
  cudf::io::datasource& datasource,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& payload_column_names,
  bool case_sensitive_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  rmm::mr::aligned_resource_adaptor& aligned_mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder()
                                               .filter(filter_expression)
                                               .case_sensitive_names(case_sensitive_names);

  // Set payload column names if provided
  if (payload_column_names.has_value()) { options.set_column_names(payload_column_names.value()); }

  auto const reader = setup_reader(datasource, options);
  auto reader_ref   = std::ref(*reader);

  auto const filtered_row_group_indices =
    apply_hybrid_scan_filters(datasource, reader_ref, options, stream, mr);

  auto const current_row_group_indices =
    cudf::host_span<cudf::size_type const>(filtered_row_group_indices);

  // Build a row mask for the filtered row groups using page index stats, or all true if no filter
  auto row_mask =
    options.get_filter().has_value()
      ? reader->build_row_mask_with_page_index_stats(current_row_group_indices, options, stream, mr)
      : reader->build_all_true_row_mask(current_row_group_indices, stream, mr);

  auto filter_tables  = std::vector<std::unique_ptr<cudf::table>>{};
  auto payload_tables = std::vector<std::unique_ptr<cudf::table>>{};

  // Helper to materialize filter and payload columns for a row group pass
  std::size_t rows_materialized = 0;
  auto const materialize_pass   = [&](cudf::host_span<cudf::size_type const> row_group_indices) {
    // Sliced row mask view for the current pass
    auto const rows_in_pass = reader->total_rows_in_row_groups(row_group_indices);
    auto* null_mask         = row_mask->nullable() ? row_mask->mutable_view().null_mask() : nullptr;
    auto const slice_null_count =
      cudf::null_count(null_mask, rows_materialized, rows_materialized + rows_in_pass, stream);
    auto row_mask_view = cudf::mutable_column_view(row_mask->type(),
                                                   rows_in_pass,
                                                   row_mask->mutable_view().data<bool>(),
                                                   null_mask,
                                                   slice_null_count,
                                                   rows_materialized);

    // Materialize filter columns
    auto const filter_byte_ranges =
      reader->filter_column_chunks_byte_ranges(row_group_indices, options);
    auto [filter_buffers, filter_data, filter_tasks] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        datasource, filter_byte_ranges, stream, mr);
    filter_tasks.get();

    reader->setup_chunking_for_filter_columns(
      1024,
      10240,
      row_group_indices,
      row_mask_view,
      cudf::io::parquet::experimental::use_data_page_mask::YES,
      filter_data,
      options,
      stream,
      mr);

    while (reader->has_next_table_chunk()) {
      filter_tables.push_back(reader->materialize_filter_columns_chunk(row_mask_view).tbl);
    }

    // Materialize payload columns
    auto const payload_byte_ranges =
      reader->payload_column_chunks_byte_ranges(row_group_indices, options);
    auto [payload_buffers, payload_data, payload_tasks] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        datasource, payload_byte_ranges, stream, mr);
    payload_tasks.get();

    reader->setup_chunking_for_payload_columns(
      1024,
      10240,
      row_group_indices,
      row_mask_view,
      cudf::io::parquet::experimental::use_data_page_mask::YES,
      payload_data,
      options,
      stream,
      mr);

    while (reader->has_next_table_chunk()) {
      payload_tables.push_back(reader->materialize_payload_columns_chunk(row_mask_view).tbl);
    }

    // Update the number of rows materialized
    rows_materialized += rows_in_pass;
  };

  if (current_row_group_indices.size() > 1) {
    auto const row_group_split = current_row_group_indices.size() / 2;
    materialize_pass(current_row_group_indices.subspan(0, row_group_split));
    materialize_pass(current_row_group_indices.subspan(
      row_group_split, current_row_group_indices.size() - row_group_split));
  } else {
    materialize_pass(current_row_group_indices);
  }

  auto filter_table  = concatenate_tables(std::move(filter_tables), stream, mr);
  auto payload_table = concatenate_tables(std::move(payload_tables), stream, mr);

  return std::tuple{std::move(filter_table), std::move(payload_table)};
}

std::unique_ptr<cudf::table> hybrid_scan_single_step(
  cudf::io::datasource& datasource,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  bool case_sensitive_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder()
                                               .filter(filter_expression)
                                               .case_sensitive_names(case_sensitive_names)
                                               .build();

  if (column_names.has_value()) { options.set_column_names(column_names.value()); }

  auto const reader = setup_reader(datasource, options);
  auto reader_ref   = std::ref(*reader);

  auto const filtered_row_group_indices =
    apply_hybrid_scan_filters(datasource, reader_ref, options, stream, mr);

  auto const current_row_group_indices =
    cudf::host_span<cudf::size_type const>(filtered_row_group_indices);

  // Get all column chunk byte ranges from the reader
  auto const all_column_chunk_byte_ranges =
    reader->all_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers and spans from the input buffer
  auto [all_col_buffers, all_col_data, all_col_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      datasource, all_column_chunk_byte_ranges, stream, mr);
  all_col_tasks.get();

  // Materialize the table with all columns
  return reader
    ->materialize_all_columns(current_row_group_indices, all_col_data, options, stream, mr)
    .tbl;
}

std::unique_ptr<cudf::table> chunked_hybrid_scan_single_step(
  cudf::io::datasource& datasource,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  bool case_sensitive_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder()
                                               .filter(filter_expression)
                                               .case_sensitive_names(case_sensitive_names)
                                               .build();

  if (column_names.has_value()) { options.set_column_names(column_names.value()); }

  auto const reader = setup_reader(datasource, options);
  auto reader_ref   = std::ref(*reader);

  auto filtered_row_group_indices =
    apply_hybrid_scan_filters(datasource, reader_ref, options, stream, mr);

  auto current_row_group_indices = cudf::host_span<cudf::size_type>(filtered_row_group_indices);

  // Helper to split the materialization of all columns into chunks
  auto tables   = std::vector<std::unique_ptr<cudf::table>>{};
  auto metadata = cudf::io::table_metadata{};
  auto const materialize_all_columns =
    [&](cudf::host_span<cudf::size_type const> row_group_indices) {
      // Get column chunk byte ranges from the reader and fetch device buffers
      auto const all_column_chunk_byte_ranges =
        reader->all_column_chunks_byte_ranges(row_group_indices, options);
      auto [all_column_chunk_buffers, all_column_chunk_data, all_column_chunk_tasks] =
        cudf::io::parquet::fetch_byte_ranges_to_device_async(
          datasource, all_column_chunk_byte_ranges, stream, mr);
      all_column_chunk_tasks.get();

      // Setup chunking for all columns and materialize the columns
      reader->setup_chunking_for_all_columns(
        1024, 10240, row_group_indices, all_column_chunk_data, options, stream, mr);

      while (reader->has_next_table_chunk()) {
        auto chunk = reader->materialize_all_columns_chunk();
        tables.push_back(std::move(chunk.tbl));
        metadata = std::move(chunk.metadata);
      }
    };

  if (current_row_group_indices.size() > 1) {
    auto const row_group_split = current_row_group_indices.size() / 2;
    materialize_all_columns(current_row_group_indices.subspan(0, row_group_split));
    materialize_all_columns(current_row_group_indices.subspan(
      row_group_split, current_row_group_indices.size() - row_group_split));
  } else {
    materialize_all_columns(current_row_group_indices);
  }

  return concatenate_tables(std::move(tables), stream, mr);
}
