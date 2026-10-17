/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_multifile_composer.hpp"

#include "hybrid_scan_common.hpp"

#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

using cudf::io::parquet::experimental::use_data_page_mask;

namespace {

/**
 * @brief Materializes all (or selected) columns of the given row groups in chunks
 *
 * @throws cudf::logic_error if the options have no filter and a chunk other than the first one is
 * empty
 */
std::vector<std::unique_ptr<cudf::table>> chunked_materialize_all_columns(
  cudf::io::parquet::experimental::hybrid_scan_multifile& reader,
  multifile_inputs const& inputs,
  cudf::host_span<std::vector<cudf::size_type> const> row_groups,
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  cudf::io::parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto all_column_chunks = fetch_multisource_device_data(
    inputs, reader.all_column_chunks_byte_ranges(row_groups, options), stream, mr);
  reader.setup_chunking_for_all_columns(chunk_read_limit,
                                        pass_read_limit,
                                        row_groups,
                                        all_column_chunks.flat_spans,
                                        options,
                                        stream,
                                        mr);

  auto tables = std::vector<std::unique_ptr<cudf::table>>{};
  while (reader.has_next_table_chunk()) {
    auto chunk = reader.materialize_all_columns_chunk().tbl;
    // Without a filter, only the first chunk can be empty, so this also ends a reader that never
    // finishes
    CUDF_EXPECTS(options.get_filter().has_value() or tables.empty() or chunk->num_rows() > 0,
                 "Number of rows in the new chunk is zero.");
    tables.push_back(std::move(chunk));
  }
  return tables;
}

}  // namespace

std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> hybrid_scan_multifile(
  cudf::io::source_info const& source_info,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& payload_column_names,
  bool case_sensitive_names,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto options = cudf::io::parquet_reader_options::builder()
                   .filter(filter_expression)
                   .case_sensitive_names(case_sensitive_names)
                   .build();
  if (payload_column_names.has_value()) { options.set_column_names(payload_column_names.value()); }

  auto inputs = multifile_inputs(source_info);
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const input_row_groups = reader.all_row_groups(options);
  auto const row_groups = reader.filter_row_groups_with_stats(input_row_groups, options, stream);
  auto row_mask = reader.build_row_mask_with_page_index_stats(row_groups, options, stream, mr);

  auto filter_column_chunks = fetch_multisource_device_data(
    inputs, reader.filter_column_chunks_byte_ranges(row_groups, options), stream, mr);
  auto row_mask_view = row_mask->mutable_view();
  auto filter_result = reader.materialize_filter_columns(row_groups,
                                                         filter_column_chunks.flat_spans,
                                                         row_mask_view,
                                                         use_data_page_mask::YES,
                                                         options,
                                                         stream,
                                                         mr);

  auto payload_column_chunks = fetch_multisource_device_data(
    inputs, reader.payload_column_chunks_byte_ranges(row_groups, options), stream, mr);
  auto payload_result = reader.materialize_payload_columns(row_groups,
                                                           payload_column_chunks.flat_spans,
                                                           row_mask_view,
                                                           use_data_page_mask::YES,
                                                           options,
                                                           stream,
                                                           mr);

  return std::tuple{std::move(filter_result.tbl), std::move(payload_result.tbl)};
}

std::unique_ptr<cudf::table> hybrid_scan_multifile_single_step(
  cudf::io::source_info const& source_info,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  bool case_sensitive_names,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto options = cudf::io::parquet_reader_options::builder()
                   .filter(filter_expression)
                   .case_sensitive_names(case_sensitive_names)
                   .build();
  if (column_names.has_value()) { options.set_column_names(column_names.value()); }

  auto inputs = multifile_inputs(source_info);
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const input_row_groups = reader.all_row_groups(options);
  auto const row_groups = reader.filter_row_groups_with_stats(input_row_groups, options, stream);

  auto all_column_chunks = fetch_multisource_device_data(
    inputs, reader.all_column_chunks_byte_ranges(row_groups, options), stream, mr);
  return reader
    .materialize_all_columns(row_groups, all_column_chunks.flat_spans, options, stream, mr)
    .tbl;
}

std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
chunked_hybrid_scan_multifile(cudf::io::source_info const& source_info,
                              cudf::ast::operation const& filter_expression,
                              std::optional<std::vector<std::string>> const& payload_column_names,
                              bool case_sensitive_names,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  auto options = cudf::io::parquet_reader_options::builder()
                   .filter(filter_expression)
                   .case_sensitive_names(case_sensitive_names)
                   .build();
  if (payload_column_names.has_value()) { options.set_column_names(payload_column_names.value()); }

  auto inputs = multifile_inputs(source_info);
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const input_row_groups = reader.all_row_groups(options);
  auto const row_groups = reader.filter_row_groups_with_stats(input_row_groups, options, stream);
  auto row_mask = reader.build_row_mask_with_page_index_stats(row_groups, options, stream, mr);

  // Non-trivial chunk and pass read limits so chunking and multi-pass reads trigger across the
  // single row mask spanning all sources
  auto constexpr chunk_read_limit = std::size_t{256 * 1024};
  auto constexpr pass_read_limit  = std::size_t{1024 * 1024};

  auto filter_tables  = std::vector<std::unique_ptr<cudf::table>>{};
  auto payload_tables = std::vector<std::unique_ptr<cudf::table>>{};

  auto filter_column_chunks = fetch_multisource_device_data(
    inputs, reader.filter_column_chunks_byte_ranges(row_groups, options), stream, mr);
  auto row_mask_view = row_mask->mutable_view();
  reader.setup_chunking_for_filter_columns(chunk_read_limit,
                                           pass_read_limit,
                                           row_groups,
                                           row_mask_view,
                                           use_data_page_mask::YES,
                                           filter_column_chunks.flat_spans,
                                           options,
                                           stream,
                                           mr);
  while (reader.has_next_table_chunk()) {
    filter_tables.push_back(reader.materialize_filter_columns_chunk(row_mask_view).tbl);
  }

  auto payload_column_chunks = fetch_multisource_device_data(
    inputs, reader.payload_column_chunks_byte_ranges(row_groups, options), stream, mr);
  reader.setup_chunking_for_payload_columns(chunk_read_limit,
                                            pass_read_limit,
                                            row_groups,
                                            row_mask_view,
                                            use_data_page_mask::YES,
                                            payload_column_chunks.flat_spans,
                                            options,
                                            stream,
                                            mr);
  while (reader.has_next_table_chunk()) {
    payload_tables.push_back(reader.materialize_payload_columns_chunk(row_mask_view).tbl);
  }

  return std::tuple{concatenate_tables(std::move(filter_tables), stream, mr),
                    concatenate_tables(std::move(payload_tables), stream, mr)};
}

std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
chunked_sparse_hybrid_scan_multifile(
  cudf::io::source_info const& source_info,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& payload_column_names,
  bool case_sensitive_names,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto options = cudf::io::parquet_reader_options::builder()
                   .filter(filter_expression)
                   .case_sensitive_names(case_sensitive_names)
                   .build();
  if (payload_column_names.has_value()) { options.set_column_names(payload_column_names.value()); }

  auto inputs = multifile_inputs(source_info);
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const input_row_groups = reader.all_row_groups(options);
  auto const row_groups = reader.filter_row_groups_with_stats(input_row_groups, options, stream);
  auto row_mask = reader.build_row_mask_with_page_index_stats(row_groups, options, stream, mr);

  auto constexpr chunk_read_limit = std::size_t{256 * 1024};
  auto constexpr pass_read_limit  = std::size_t{1024 * 1024};

  auto filter_tables  = std::vector<std::unique_ptr<cudf::table>>{};
  auto payload_tables = std::vector<std::unique_ptr<cudf::table>>{};

  auto filter_column_chunks = fetch_multisource_device_data(
    inputs, reader.filter_column_chunks_byte_ranges(row_groups, options), stream, mr);
  auto row_mask_view = row_mask->mutable_view();
  reader.setup_chunking_for_filter_columns(chunk_read_limit,
                                           pass_read_limit,
                                           row_groups,
                                           row_mask_view,
                                           use_data_page_mask::YES,
                                           filter_column_chunks.flat_spans,
                                           options,
                                           stream,
                                           mr);
  while (reader.has_next_table_chunk()) {
    filter_tables.push_back(reader.materialize_filter_columns_chunk(row_mask_view).tbl);
  }

  auto const payload_page_ranges =
    reader.payload_pages_byte_ranges(row_groups, row_mask->view(), options, stream);
  auto payload_page_data = fetch_multisource_device_data(inputs, payload_page_ranges, stream, mr);

  reader.setup_chunking_for_payload_columns(chunk_read_limit,
                                            pass_read_limit,
                                            row_groups,
                                            row_mask->view(),
                                            payload_page_data.flat_spans,
                                            options,
                                            stream,
                                            mr);
  while (reader.has_next_table_chunk()) {
    payload_tables.push_back(reader.materialize_payload_columns_chunk(row_mask->view()).tbl);
  }

  return std::tuple{concatenate_tables(std::move(filter_tables), stream, mr),
                    concatenate_tables(std::move(payload_tables), stream, mr)};
}

std::unique_ptr<cudf::table> chunked_hybrid_scan_multifile_single_step(
  cudf::io::source_info const& source_info,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  bool case_sensitive_names,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto options = cudf::io::parquet_reader_options::builder()
                   .filter(filter_expression)
                   .case_sensitive_names(case_sensitive_names)
                   .build();
  if (column_names.has_value()) { options.set_column_names(column_names.value()); }

  auto inputs = multifile_inputs(source_info);
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const input_row_groups = reader.all_row_groups(options);
  auto const row_groups = reader.filter_row_groups_with_stats(input_row_groups, options, stream);

  auto constexpr chunk_read_limit = std::size_t{256 * 1024};
  auto constexpr pass_read_limit  = std::size_t{1024 * 1024};

  return concatenate_tables(
    chunked_materialize_all_columns(
      reader, inputs, row_groups, chunk_read_limit, pass_read_limit, options, stream, mr),
    stream,
    mr);
}

std::vector<std::unique_ptr<cudf::table>> chunked_hybrid_scan_multifile_passes(
  cudf::io::source_info const& source_info,
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const options = cudf::io::parquet_reader_options::builder().build();

  auto inputs = multifile_inputs(source_info);
  auto reader =
    cudf::io::parquet::experimental::hybrid_scan_multifile{inputs.footer_byte_spans, options};
  setup_page_indexes(reader, inputs);

  auto const passes = reader.construct_row_group_passes(
    cudf::io::parquet::experimental::read_columns_mode::ALL_COLUMNS,
    reader.all_row_groups(options),
    pass_read_limit,
    options);

  auto tables = std::vector<std::unique_ptr<cudf::table>>{};
  for (auto const& pass : passes) {
    auto pass_tables = chunked_materialize_all_columns(
      reader, inputs, pass, chunk_read_limit, pass_read_limit, options, stream, mr);
    std::move(pass_tables.begin(), pass_tables.end(), std::back_inserter(tables));
  }

  return tables;
}
