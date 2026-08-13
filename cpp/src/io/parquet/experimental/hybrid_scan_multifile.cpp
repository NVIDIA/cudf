/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_impl.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/utilities/error.hpp>

#include <numeric>

namespace cudf::io::parquet::experimental {

hybrid_scan_multifile::hybrid_scan_multifile(
  cudf::host_span<cudf::host_span<uint8_t const> const> footer_bytes,
  parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(footer_bytes, options)}
{
}

hybrid_scan_multifile::hybrid_scan_multifile(cudf::host_span<FileMetaData const> parquet_metadata,
                                             parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(parquet_metadata, options)}
{
}

hybrid_scan_multifile::~hybrid_scan_multifile() = default;

std::vector<FileMetaData> hybrid_scan_multifile::parquet_metadatas() const
{
  return _impl->parquet_metadatas();
}

std::vector<text::byte_range_info> hybrid_scan_multifile::page_index_byte_ranges() const
{
  return _impl->page_index_byte_ranges();
}

void hybrid_scan_multifile::setup_page_indexes(
  cudf::host_span<cudf::host_span<uint8_t const> const> page_index_bytes) const
{
  CUDF_FUNC_RANGE();
  _impl->setup_page_indexes(page_index_bytes);
}

std::vector<std::vector<size_type>> hybrid_scan_multifile::all_row_groups(
  parquet_reader_options const& options) const
{
  return _impl->all_row_groups(options);
}

size_type hybrid_scan_multifile::total_rows_in_row_groups(
  cudf::host_span<std::vector<size_type> const> row_group_indices) const
{
  if (row_group_indices.empty()) { return 0; }
  return _impl->total_rows_in_row_groups(row_group_indices);
}

void hybrid_scan_multifile::reset_column_selection() const { _impl->reset_column_selection(); }

std::vector<std::vector<size_type>> hybrid_scan_multifile::filter_row_groups_with_byte_range(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();
  return _impl->filter_row_groups_with_byte_range(row_group_indices, options);
}

std::vector<std::vector<size_type>> hybrid_scan_multifile::filter_row_groups_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();
  return _impl->filter_row_groups_with_stats(row_group_indices, options, stream);
}

std::pair<std::vector<text::byte_range_info>, std::vector<text::byte_range_info>>
hybrid_scan_multifile::secondary_filters_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();
  return _impl->secondary_filters_byte_ranges(row_group_indices, options);
}

std::unique_ptr<cudf::column> hybrid_scan_multifile::build_all_true_row_mask(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->build_all_true_row_mask(row_group_indices, stream, mr);
}

std::unique_ptr<cudf::column> hybrid_scan_multifile::build_row_mask_with_page_index_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->build_row_mask_with_page_index_stats(row_group_indices, options, stream, mr);
}

std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::filter_column_chunks_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();
  return _impl->filter_column_chunks_byte_ranges(row_group_indices, options);
}

table_with_metadata hybrid_scan_multifile::materialize_filter_columns(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::mutable_column_view& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->materialize_filter_columns(
    row_group_indices, column_chunk_data, row_mask, mask_data_pages, options, stream, mr);
}

std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::payload_column_chunks_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();
  return _impl->payload_column_chunks_byte_ranges(row_group_indices, options);
}

table_with_metadata hybrid_scan_multifile::materialize_payload_columns(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->materialize_payload_columns(
    row_group_indices, column_chunk_data, row_mask, mask_data_pages, options, stream, mr);
}

std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::all_column_chunks_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();
  return _impl->all_column_chunks_byte_ranges(row_group_indices, options);
}

table_with_metadata hybrid_scan_multifile::materialize_all_columns(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->materialize_all_columns(row_group_indices, column_chunk_data, options, stream, mr);
}

void hybrid_scan_multifile::setup_chunking_for_filter_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  _impl->setup_chunking_for_filter_columns(chunk_read_limit,
                                           pass_read_limit,
                                           row_group_indices,
                                           row_mask,
                                           mask_data_pages,
                                           column_chunk_data,
                                           options,
                                           stream,
                                           mr);
}

table_with_metadata hybrid_scan_multifile::materialize_filter_columns_chunk(
  cudf::mutable_column_view& row_mask) const
{
  CUDF_FUNC_RANGE();
  return _impl->materialize_filter_columns_chunk(row_mask);
}

void hybrid_scan_multifile::setup_chunking_for_payload_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  _impl->setup_chunking_for_payload_columns(chunk_read_limit,
                                            pass_read_limit,
                                            row_group_indices,
                                            row_mask,
                                            mask_data_pages,
                                            column_chunk_data,
                                            options,
                                            stream,
                                            mr);
}

table_with_metadata hybrid_scan_multifile::materialize_payload_columns_chunk(
  cudf::column_view const& row_mask) const
{
  CUDF_FUNC_RANGE();
  return _impl->materialize_payload_columns_chunk(row_mask);
}

void hybrid_scan_multifile::setup_chunking_for_all_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  _impl->setup_chunking_for_all_columns(
    chunk_read_limit, pass_read_limit, row_group_indices, column_chunk_data, options, stream, mr);
}

table_with_metadata hybrid_scan_multifile::materialize_all_columns_chunk() const
{
  CUDF_FUNC_RANGE();
  return _impl->materialize_all_columns_chunk();
}

bool hybrid_scan_multifile::has_next_table_chunk() const { return _impl->has_next_table_chunk(); }

std::vector<std::vector<std::vector<size_type>>> hybrid_scan_multifile::construct_row_group_passes(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::size_t pass_read_limit) const
{
  CUDF_FUNC_RANGE();

  auto const total_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    std::size_t{0},
                    [](auto sum, auto const& rgs) { return sum + rgs.size(); });
  CUDF_EXPECTS(
    total_row_groups > 0, "Empty input row group indices encountered", std::invalid_argument);

  auto [passes, source_map] =
    _impl->construct_row_group_passes(row_group_indices, total_row_groups, pass_read_limit);

  if (pass_read_limit == 0) { return {passes}; }

  auto source_passes = std::vector<std::vector<std::vector<size_type>>>{};
  source_passes.reserve(passes.size());

  if (row_group_indices.size() == 1) {
    for (auto& pass : passes) {
      source_passes.emplace_back();
      source_passes.back().push_back(std::move(pass));
    }
    return source_passes;
  }

  auto source_map_it = source_map.begin();
  for (auto const& pass : passes) {
    auto source_pass = std::vector<std::vector<size_type>>(row_group_indices.size());
    for (auto const row_group_index : pass) {
      source_pass[*source_map_it++].push_back(row_group_index);
    }
    source_passes.push_back(std::move(source_pass));
  }

  return source_passes;
}

std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::dictionary_pages_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();
  return _impl->dictionary_pages_byte_ranges(row_group_indices, options);
}

std::vector<std::vector<size_type>> hybrid_scan_multifile::filter_row_groups_with_dictionary_pages(
  cudf::host_span<cudf::device_span<uint8_t const> const> dictionary_page_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();
  return _impl->filter_row_groups_with_dictionary_pages(
    dictionary_page_data, row_group_indices, options, stream);
}

}  // namespace cudf::io::parquet::experimental
