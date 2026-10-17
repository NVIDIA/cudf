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
  std::span<cudf::host_span<uint8_t const> const> footer_bytes,
  parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(footer_bytes, options)}
{
}

hybrid_scan_multifile::hybrid_scan_multifile(std::span<FileMetaData const> parquet_metadata,
                                             parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(parquet_metadata, options)}
{
}

hybrid_scan_multifile::hybrid_scan_multifile(std::vector<FileMetaData>&& parquet_metadata,
                                             parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(std::move(parquet_metadata), options)}
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
  std::span<cudf::host_span<uint8_t const> const> page_index_bytes) const
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
  std::span<std::vector<size_type> const> row_group_indices) const
{
  if (row_group_indices.empty()) { return 0; }
  return _impl->total_rows_in_row_groups(row_group_indices);
}

void hybrid_scan_multifile::reset_column_selection() { _impl->reset_column_selection(); }

std::vector<std::vector<size_type>> hybrid_scan_multifile::filter_row_groups_with_byte_range(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();
  return _impl->filter_row_groups_with_byte_range(row_group_indices, options);
}

std::vector<std::vector<size_type>> hybrid_scan_multifile::filter_row_groups_with_stats(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return _impl->filter_row_groups_with_stats(row_group_indices, options, stream);
}

std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::bloom_filters_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_FUNC_RANGE();
  return _impl->bloom_filters_byte_ranges(row_group_indices, options);
}

std::vector<std::vector<size_type>> hybrid_scan_multifile::filter_row_groups_with_bloom_filters(
  std::span<cudf::device_span<uint8_t const> const> bloom_filter_data,
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return _impl->filter_row_groups_with_bloom_filters(
    bloom_filter_data, row_group_indices, options, stream);
}

std::unique_ptr<cudf::column> hybrid_scan_multifile::build_all_true_row_mask(
  std::span<std::vector<size_type> const> row_group_indices,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return _impl->build_all_true_row_mask(row_group_indices, stream, mr);
}

std::unique_ptr<cudf::column> hybrid_scan_multifile::build_row_mask_with_page_index_stats(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return _impl->build_row_mask_with_page_index_stats(row_group_indices, options, stream, mr);
}

template <read_columns_mode mode>
std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_FUNC_RANGE();
  return _impl->column_chunks_byte_ranges<mode>(row_group_indices, options);
}

std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::filter_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  return column_chunks_byte_ranges<read_columns_mode::FILTER_COLUMNS>(row_group_indices, options);
}

template <read_columns_mode mode>
table_with_metadata hybrid_scan_multifile::materialize_table(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  row_mask_view_t<mode> const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return _impl->materialize_table<mode>(
    row_group_indices, column_chunk_data, row_mask, mask_data_pages, options, stream, mr);
}

table_with_metadata hybrid_scan_multifile::materialize_filter_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::mutable_column_view& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  return materialize_table<read_columns_mode::FILTER_COLUMNS>(
    row_group_indices, column_chunk_data, row_mask, mask_data_pages, options, stream, mr);
}

std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::payload_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  return column_chunks_byte_ranges<read_columns_mode::PAYLOAD_COLUMNS>(row_group_indices, options);
}

std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::sparse_payload_pages_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return _impl->sparse_payload_pages_byte_ranges(row_group_indices, row_mask, options, stream);
}

std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::payload_pages_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  return sparse_payload_pages_byte_ranges(row_group_indices, row_mask, options, stream);
}

table_with_metadata hybrid_scan_multifile::materialize_payload_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  return materialize_table<read_columns_mode::PAYLOAD_COLUMNS>(
    row_group_indices, column_chunk_data, row_mask, mask_data_pages, options, stream, mr);
}

std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>
hybrid_scan_multifile::all_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  return column_chunks_byte_ranges<read_columns_mode::ALL_COLUMNS>(row_group_indices, options);
}

table_with_metadata hybrid_scan_multifile::materialize_all_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  return materialize_table<read_columns_mode::ALL_COLUMNS>(
    row_group_indices, column_chunk_data, {}, use_data_page_mask::NO, options, stream, mr);
}

template <read_columns_mode mode>
void hybrid_scan_multifile::setup_chunked_table_materialization(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  _impl->setup_chunked_table_materialization<mode>(chunk_read_limit,
                                                   pass_read_limit,
                                                   row_group_indices,
                                                   column_chunk_data,
                                                   row_mask,
                                                   mask_data_pages,
                                                   options,
                                                   stream,
                                                   mr);
}

void hybrid_scan_multifile::setup_chunking_for_filter_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  setup_chunked_table_materialization<read_columns_mode::FILTER_COLUMNS>(chunk_read_limit,
                                                                         pass_read_limit,
                                                                         row_group_indices,
                                                                         column_chunk_data,
                                                                         row_mask,
                                                                         mask_data_pages,
                                                                         options,
                                                                         stream,
                                                                         mr);
}

template <read_columns_mode mode>
table_with_metadata hybrid_scan_multifile::materialize_table_chunk(
  row_mask_view_t<mode> const& row_mask)
{
  CUDF_FUNC_RANGE();
  return _impl->materialize_table_chunk<mode>(row_mask);
}

table_with_metadata hybrid_scan_multifile::materialize_filter_columns_chunk(
  cudf::mutable_column_view& row_mask)
{
  return materialize_table_chunk<read_columns_mode::FILTER_COLUMNS>(row_mask);
}

void hybrid_scan_multifile::setup_chunking_for_payload_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  setup_chunked_table_materialization<read_columns_mode::PAYLOAD_COLUMNS>(chunk_read_limit,
                                                                          pass_read_limit,
                                                                          row_group_indices,
                                                                          column_chunk_data,
                                                                          row_mask,
                                                                          mask_data_pages,
                                                                          options,
                                                                          stream,
                                                                          mr);
}

void hybrid_scan_multifile::setup_chunked_sparse_payload_materialization(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> page_data,
  cudf::column_view const& row_mask,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  _impl->setup_chunked_sparse_payload_materialization(
    chunk_read_limit, pass_read_limit, row_group_indices, page_data, row_mask, options, stream, mr);
}

void hybrid_scan_multifile::setup_chunking_for_payload_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  std::span<cudf::device_span<uint8_t const> const> page_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  setup_chunked_sparse_payload_materialization(
    chunk_read_limit, pass_read_limit, row_group_indices, page_data, row_mask, options, stream, mr);
}

table_with_metadata hybrid_scan_multifile::materialize_payload_columns_chunk(
  cudf::column_view const& row_mask)
{
  return materialize_table_chunk<read_columns_mode::PAYLOAD_COLUMNS>(row_mask);
}

void hybrid_scan_multifile::setup_chunking_for_all_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  setup_chunked_table_materialization<read_columns_mode::ALL_COLUMNS>(chunk_read_limit,
                                                                      pass_read_limit,
                                                                      row_group_indices,
                                                                      column_chunk_data,
                                                                      {},
                                                                      use_data_page_mask::NO,
                                                                      options,
                                                                      stream,
                                                                      mr);
}

table_with_metadata hybrid_scan_multifile::materialize_all_columns_chunk()
{
  return materialize_table_chunk<read_columns_mode::ALL_COLUMNS>({});
}

bool hybrid_scan_multifile::has_next_table_chunk() { return _impl->has_next_table_chunk(); }

std::vector<std::vector<std::vector<size_type>>> hybrid_scan_multifile::construct_row_group_passes(
  read_columns_mode columns_mode,
  std::span<std::vector<size_type> const> row_group_indices,
  std::size_t pass_read_limit,
  parquet_reader_options const& options)
{
  CUDF_FUNC_RANGE();

  auto const total_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    std::size_t{0},
                    [](auto sum, auto const& rgs) { return sum + rgs.size(); });

  auto [passes, source_map] = _impl->construct_row_group_passes(
    columns_mode, row_group_indices, total_row_groups, pass_read_limit, options);

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
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_FUNC_RANGE();
  return _impl->dictionary_pages_byte_ranges(row_group_indices, options);
}

std::vector<std::vector<size_type>> hybrid_scan_multifile::filter_row_groups_with_dictionary_pages(
  std::span<cudf::device_span<uint8_t const> const> dictionary_page_data,
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return _impl->filter_row_groups_with_dictionary_pages(
    dictionary_page_data, row_group_indices, options, stream);
}

// Explicit instantiations of the generic columns APIs for each columns selection mode
#define INSTANTIATE_COLUMNS_APIS(mode)                                                            \
  template std::pair<std::vector<text::byte_range_info>, std::vector<size_type>>                  \
  hybrid_scan_multifile::column_chunks_byte_ranges<mode>(std::span<std::vector<size_type> const>, \
                                                         parquet_reader_options const&);          \
  template table_with_metadata hybrid_scan_multifile::materialize_table<mode>(                    \
    std::span<std::vector<size_type> const>,                                                      \
    std::span<cudf::device_span<uint8_t const> const>,                                            \
    row_mask_view_t<mode> const&,                                                                 \
    use_data_page_mask,                                                                           \
    parquet_reader_options const&,                                                                \
    cuda::stream_ref,                                                                             \
    cudf::memory_resources);                                                                      \
  template void hybrid_scan_multifile::setup_chunked_table_materialization<mode>(                 \
    std::size_t,                                                                                  \
    std::size_t,                                                                                  \
    std::span<std::vector<size_type> const>,                                                      \
    std::span<cudf::device_span<uint8_t const> const>,                                            \
    cudf::column_view const&,                                                                     \
    use_data_page_mask,                                                                           \
    parquet_reader_options const&,                                                                \
    cuda::stream_ref,                                                                             \
    cudf::memory_resources);                                                                      \
  template table_with_metadata hybrid_scan_multifile::materialize_table_chunk<mode>(              \
    row_mask_view_t<mode> const&);

INSTANTIATE_COLUMNS_APIS(read_columns_mode::FILTER_COLUMNS)
INSTANTIATE_COLUMNS_APIS(read_columns_mode::PAYLOAD_COLUMNS)
INSTANTIATE_COLUMNS_APIS(read_columns_mode::ALL_COLUMNS)

#undef INSTANTIATE_COLUMNS_APIS

}  // namespace cudf::io::parquet::experimental
