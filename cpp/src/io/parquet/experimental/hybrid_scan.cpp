/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_impl.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/host_vector.h>

namespace cudf::io::parquet::experimental {

hybrid_scan_metadata::hybrid_scan_metadata(cudf::host_span<uint8_t const> footer_bytes,
                                           parquet_reader_options const& options)
  : _metadata{std::make_shared<detail::aggregate_reader_metadata>(
      std::vector<cudf::host_span<uint8_t const>>{footer_bytes},
      options.is_enabled_use_arrow_schema(),
      options.is_enabled_allow_mismatched_pq_schemas())}
{
}

hybrid_scan_metadata::hybrid_scan_metadata(FileMetaData const& parquet_metadata,
                                           parquet_reader_options const& options)
  : _metadata{std::make_shared<detail::aggregate_reader_metadata>(
      std::vector<FileMetaData>{parquet_metadata},
      options.is_enabled_use_arrow_schema(),
      options.is_enabled_allow_mismatched_pq_schemas())}
{
}

hybrid_scan_metadata::~hybrid_scan_metadata() = default;

hybrid_scan_reader::hybrid_scan_reader(cudf::host_span<uint8_t const> footer_bytes,
                                       parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(
      std::vector<cudf::host_span<uint8_t const>>{footer_bytes}, options)}
{
}

hybrid_scan_reader::hybrid_scan_reader(FileMetaData const& parquet_metadata,
                                       parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(
      std::vector<FileMetaData>{parquet_metadata}, options)}
{
}

hybrid_scan_reader::hybrid_scan_reader(hybrid_scan_metadata metadata)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(std::move(metadata._metadata))}
{
}

hybrid_scan_reader::~hybrid_scan_reader() = default;

[[nodiscard]] text::byte_range_info hybrid_scan_reader::page_index_byte_range() const
{
  return _impl->page_index_byte_ranges().front();
}

[[nodiscard]] FileMetaData hybrid_scan_reader::parquet_metadata() const
{
  return _impl->parquet_metadatas().front();
}

void hybrid_scan_reader::setup_page_index(cudf::host_span<uint8_t const> page_index_bytes) const
{
  CUDF_FUNC_RANGE();
  return _impl->setup_page_indexes(std::vector<cudf::host_span<uint8_t const>>{page_index_bytes});
}

std::vector<cudf::size_type> hybrid_scan_reader::all_row_groups(
  parquet_reader_options const& options) const
{
  CUDF_EXPECTS(options.get_row_groups().size() <= 1,
               "Encountered invalid size of row group indices in parquet reader options");

  // If row groups are specified in parquet reader options, return them as is
  if (options.get_row_groups().size() == 1) { return options.get_row_groups().front(); }

  return _impl->all_row_groups(options).front();
}

std::size_t hybrid_scan_reader::total_rows_in_row_groups(
  std::span<size_type const> row_group_indices) const
{
  if (row_group_indices.empty()) { return 0; }

  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return _impl->total_rows_in_row_groups(input_row_group_indices);
}

void hybrid_scan_reader::reset_column_selection() { _impl->reset_column_selection(); }

std::vector<size_type> hybrid_scan_reader::filter_row_groups_with_byte_range(
  std::span<size_type const> row_group_indices, parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->filter_row_groups_with_byte_range(input_row_group_indices, options).front();
}

std::vector<cudf::size_type> hybrid_scan_reader::filter_row_groups_with_stats(
  std::span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->filter_row_groups_with_stats(input_row_group_indices, options, stream).front();
}

std::vector<text::byte_range_info> hybrid_scan_reader::bloom_filters_byte_ranges(
  std::span<size_type const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->bloom_filters_byte_ranges(input_row_group_indices, options).first;
}

std::vector<text::byte_range_info> hybrid_scan_reader::dictionary_pages_byte_ranges(
  std::span<size_type const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->dictionary_pages_byte_ranges(input_row_group_indices, options).first;
}

std::vector<cudf::size_type> hybrid_scan_reader::filter_row_groups_with_dictionary_pages(
  std::span<cudf::device_span<uint8_t const> const> dictionary_page_data,
  std::span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl
    ->filter_row_groups_with_dictionary_pages(
      dictionary_page_data, input_row_group_indices, options, stream)
    .front();
}

std::vector<cudf::size_type> hybrid_scan_reader::filter_row_groups_with_bloom_filters(
  std::span<cudf::device_span<uint8_t const> const> bloom_filter_data,
  std::span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl
    ->filter_row_groups_with_bloom_filters(
      bloom_filter_data, input_row_group_indices, options, stream)
    .front();
}

std::unique_ptr<cudf::column> hybrid_scan_reader::build_all_true_row_mask(
  std::span<size_type const> row_group_indices,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->build_all_true_row_mask(input_row_group_indices, stream, mr);
}

std::unique_ptr<cudf::column> hybrid_scan_reader::build_row_mask_with_page_index_stats(
  std::span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->build_row_mask_with_page_index_stats(input_row_group_indices, options, stream, mr);
}

template <read_columns_mode mode>
std::vector<byte_range_info> hybrid_scan_reader::column_chunks_byte_ranges(
  std::span<size_type const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->column_chunks_byte_ranges<mode>(input_row_group_indices, options).first;
}

std::vector<byte_range_info> hybrid_scan_reader::filter_column_chunks_byte_ranges(
  std::span<size_type const> row_group_indices, parquet_reader_options const& options)
{
  return column_chunks_byte_ranges<read_columns_mode::FILTER_COLUMNS>(row_group_indices, options);
}

template <read_columns_mode mode>
table_with_metadata hybrid_scan_reader::materialize_table(
  std::span<size_type const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  row_mask_view_t<mode> const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->materialize_table<mode>(
    input_row_group_indices, column_chunk_data, row_mask, mask_data_pages, options, stream, mr);
}

table_with_metadata hybrid_scan_reader::materialize_filter_columns(
  std::span<size_type const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::mutable_column_view& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return materialize_table<read_columns_mode::FILTER_COLUMNS>(
    row_group_indices, column_chunk_data, row_mask, mask_data_pages, options, stream, mr);
}

std::vector<byte_range_info> hybrid_scan_reader::payload_column_chunks_byte_ranges(
  std::span<size_type const> row_group_indices, parquet_reader_options const& options)
{
  return column_chunks_byte_ranges<read_columns_mode::PAYLOAD_COLUMNS>(row_group_indices, options);
}

table_with_metadata hybrid_scan_reader::materialize_payload_columns(
  std::span<size_type const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return materialize_table<read_columns_mode::PAYLOAD_COLUMNS>(
    row_group_indices, column_chunk_data, row_mask, mask_data_pages, options, stream, mr);
}

std::vector<byte_range_info> hybrid_scan_reader::all_column_chunks_byte_ranges(
  std::span<size_type const> row_group_indices, parquet_reader_options const& options)
{
  return column_chunks_byte_ranges<read_columns_mode::ALL_COLUMNS>(row_group_indices, options);
}

table_with_metadata hybrid_scan_reader::materialize_all_columns(
  std::span<size_type const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return materialize_table<read_columns_mode::ALL_COLUMNS>(
    row_group_indices, column_chunk_data, {}, use_data_page_mask::NO, options, stream, mr);
}

template <read_columns_mode mode>
void hybrid_scan_reader::setup_chunked_table_materialization(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<size_type const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Temporary vector with row group indices from the first source
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  _impl->setup_chunked_table_materialization<mode>(chunk_read_limit,
                                                   pass_read_limit,
                                                   input_row_group_indices,
                                                   column_chunk_data,
                                                   row_mask,
                                                   mask_data_pages,
                                                   options,
                                                   stream,
                                                   mr);
}

void hybrid_scan_reader::setup_chunking_for_filter_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<size_type const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
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
table_with_metadata hybrid_scan_reader::materialize_table_chunk(
  row_mask_view_t<mode> const& row_mask)
{
  CUDF_FUNC_RANGE();

  return _impl->materialize_table_chunk<mode>(row_mask);
}

table_with_metadata hybrid_scan_reader::materialize_filter_columns_chunk(
  cudf::mutable_column_view& row_mask)
{
  return materialize_table_chunk<read_columns_mode::FILTER_COLUMNS>(row_mask);
}

void hybrid_scan_reader::setup_chunking_for_payload_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<size_type const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
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

table_with_metadata hybrid_scan_reader::materialize_payload_columns_chunk(
  cudf::column_view const& row_mask)
{
  return materialize_table_chunk<read_columns_mode::PAYLOAD_COLUMNS>(row_mask);
}

void hybrid_scan_reader::setup_chunking_for_all_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<size_type const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
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

table_with_metadata hybrid_scan_reader::materialize_all_columns_chunk()
{
  return materialize_table_chunk<read_columns_mode::ALL_COLUMNS>({});
}

std::vector<std::vector<cudf::size_type>> hybrid_scan_reader::construct_row_group_passes(
  read_columns_mode columns_mode,
  std::span<cudf::size_type const> row_group_indices,
  std::size_t pass_read_limit,
  parquet_reader_options const& options)
{
  CUDF_FUNC_RANGE();

  auto const total_row_groups = row_group_indices.size();
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl
    ->construct_row_group_passes(
      columns_mode, input_row_group_indices, total_row_groups, pass_read_limit, options)
    .first;
}

bool hybrid_scan_reader::has_next_table_chunk() { return _impl->has_next_table_chunk(); }

// Explicit instantiations of the generic columns APIs for each columns selection mode
#define INSTANTIATE_COLUMNS_APIS(mode)                                                       \
  template std::vector<byte_range_info> hybrid_scan_reader::column_chunks_byte_ranges<mode>( \
    std::span<size_type const>, parquet_reader_options const&);                              \
  template table_with_metadata hybrid_scan_reader::materialize_table<mode>(                  \
    std::span<size_type const>,                                                              \
    std::span<cudf::device_span<uint8_t const> const>,                                       \
    row_mask_view_t<mode> const&,                                                            \
    use_data_page_mask,                                                                      \
    parquet_reader_options const&,                                                           \
    cuda::stream_ref,                                                                        \
    rmm::device_async_resource_ref);                                                         \
  template void hybrid_scan_reader::setup_chunked_table_materialization<mode>(               \
    std::size_t,                                                                             \
    std::size_t,                                                                             \
    std::span<size_type const>,                                                              \
    std::span<cudf::device_span<uint8_t const> const>,                                       \
    cudf::column_view const&,                                                                \
    use_data_page_mask,                                                                      \
    parquet_reader_options const&,                                                           \
    cuda::stream_ref,                                                                        \
    rmm::device_async_resource_ref);                                                         \
  template table_with_metadata hybrid_scan_reader::materialize_table_chunk<mode>(            \
    row_mask_view_t<mode> const&);

INSTANTIATE_COLUMNS_APIS(read_columns_mode::FILTER_COLUMNS)
INSTANTIATE_COLUMNS_APIS(read_columns_mode::PAYLOAD_COLUMNS)
INSTANTIATE_COLUMNS_APIS(read_columns_mode::ALL_COLUMNS)

#undef INSTANTIATE_COLUMNS_APIS

}  // namespace cudf::io::parquet::experimental
