/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_impl.hpp"

#include "cudf/io/text/byte_range_info.hpp"
#include "hybrid_scan_helpers.hpp"
#include "io/parquet/reader_impl_chunking_utils.cuh"
#include "io/parquet/synthetic_column_helpers.hpp"

#include <cudf/copying.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <ranges>
#include <tuple>
#include <utility>

namespace cudf::io::parquet::experimental::detail {

using io::detail::inline_column_buffer;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::decode_kernel_mask;
using parquet::detail::file_intermediate_data;
using parquet::detail::PageInfo;
using parquet::detail::PageNestingDecodeInfo;
using text::byte_range_info;

namespace {

/**
 * @brief Tests the logical type for a fixed length byte array column to see if it should be
 * treated as a string.
 *
 * Currently the only logical type that has special handling is DECIMAL. Other valid types in the
 * future would be UUID (still treated as string) and FLOAT16 (which for now would also be treated
 * as a string).
 *
 * @param logical_type The logical type to test
 * @return Boolean indicating if the logical type should be treated as a string
 */
[[maybe_unused]] inline bool is_treat_fixed_length_as_string(
  cuda::std::optional<LogicalType> const& logical_type)
{
  if (!logical_type.has_value()) { return true; }
  return logical_type->type != LogicalType::DECIMAL;
}

/**
 * @brief Get the output types from the output buffer template
 *
 * @param output_buffer_template Output buffer template
 * @return Output types
 */
[[nodiscard]] std::vector<cudf::data_type> get_output_types(
  std::span<inline_column_buffer const> output_buffer_template)
{
  std::vector<cudf::data_type> output_dtypes;
  output_dtypes.reserve(output_buffer_template.size());
  std::transform(output_buffer_template.begin(),
                 output_buffer_template.end(),
                 std::back_inserter(output_dtypes),
                 [](auto const& col) { return col.type; });
  return output_dtypes;
}

/**
 * @brief Construct a vector of empty-like buffers from the input buffers
 *
 * @param buffers Input buffers
 * @return Vector of empty-like buffers
 */
[[nodiscard]] std::vector<inline_column_buffer> make_empty_like_column_buffers(
  std::span<inline_column_buffer const> buffers)
{
  std::vector<inline_column_buffer> empty_buffers;
  empty_buffers.reserve(buffers.size());
  std::transform(
    buffers.begin(), buffers.end(), std::back_inserter(empty_buffers), [](auto const& buffer) {
      return inline_column_buffer::empty_like(buffer);
    });
  return empty_buffers;
}

/**
 * @brief Count the number of row groups in the input
 *
 * @param row_group_indices Row group indices
 * @return Number of row groups
 */
[[nodiscard]] inline size_type count_row_groups(
  std::span<std::vector<size_type> const> row_group_indices)
{
  return std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    size_type{0},
    [](auto sum, auto const& rgs) { return sum + static_cast<size_type>(rgs.size()); });
}

/**
 * @brief Get the byte range of a column chunk's dictionary page, if present
 *
 * @param column Column chunk metadata with a valid offset index
 * @return Dictionary page offset and size, or `std::nullopt` when no dictionary page is present
 */
[[nodiscard]] std::optional<std::pair<int64_t, int64_t>> dictionary_page_range(
  ColumnChunk const& column)
{
  auto const& page_locations = column.offset_index->page_locations;
  if (column.meta_data.dictionary_page_offset > 0) {
    auto const offset = column.meta_data.dictionary_page_offset;
    return std::pair{offset, column.meta_data.data_page_offset - offset};
  }
  if (not page_locations.empty() and
      column.meta_data.data_page_offset < page_locations.front().offset) {
    auto const offset = column.meta_data.data_page_offset;
    return std::pair{offset, page_locations.front().offset - offset};
  }
  return std::nullopt;
}

}  // namespace

void hybrid_scan_reader_impl::mark_buffers_nullable_for_pruned_pages()
{
  auto const& pass               = *_pass_itm_data;
  auto buffers_with_pruned_pages = std::vector<bool>(_output_buffers.size(), false);
  auto pruned_page_indices =
    std::views::iota(std::size_t{0}, _pass_page_mask.size()) |
    std::views::filter([&](auto page_idx) { return not _pass_page_mask[page_idx]; });
  std::ranges::for_each(pruned_page_indices, [&](auto page_idx) {
    auto const& chunk        = pass.chunks[pass.pages[page_idx].chunk_idx];
    auto const& input_column = _input_columns[chunk.src_col_index];
    buffers_with_pruned_pages[input_column.nesting.front()] = true;
  });

  // Helper to mark a buffer and its children nullable
  auto const mark_buffers_nullable = [](auto const& self,
                                        std::span<inline_column_buffer> buffers) -> void {
    for (auto& buffer : buffers) {
      // Page pruning synthesizes null rows at every nesting level except list elements.
      if ((buffer.user_data & parquet::detail::PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) == 0) {
        buffer.is_nullable = true;
      }
      self(self, buffer.children);
    }
  };

  // Mark buffers with pruned pages as nullable
  auto buffers_with_pruned_page_indices =
    std::views::iota(std::size_t{0}, _output_buffers.size()) |
    std::views::filter([&](auto buffer_idx) { return buffers_with_pruned_pages[buffer_idx]; });
  std::ranges::for_each(buffers_with_pruned_page_indices, [&](auto buffer_idx) {
    mark_buffers_nullable(mark_buffers_nullable,
                          std::span<inline_column_buffer>{&_output_buffers[buffer_idx], 1});
    mark_buffers_nullable(
      mark_buffers_nullable,
      std::span<inline_column_buffer>{&_output_buffers_template[buffer_idx], 1});
  });
}

hybrid_scan_reader_impl::hybrid_scan_reader_impl(
  cudf::host_span<cudf::host_span<uint8_t const> const> footer_bytes,
  parquet_reader_options const& options)
{
  _metadata = std::make_shared<aggregate_reader_metadata>(
    footer_bytes, options.is_enabled_use_arrow_schema(), has_cols_from_mismatched_sources(options));

  _extended_metadata = static_cast<aggregate_reader_metadata*>(_metadata.get());
}

hybrid_scan_reader_impl::hybrid_scan_reader_impl(
  cudf::host_span<FileMetaData const> parquet_metadatas, parquet_reader_options const& options)
{
  _metadata =
    std::make_shared<aggregate_reader_metadata>(parquet_metadatas,
                                                options.is_enabled_use_arrow_schema(),
                                                has_cols_from_mismatched_sources(options));
  _extended_metadata = static_cast<aggregate_reader_metadata*>(_metadata.get());
}

hybrid_scan_reader_impl::hybrid_scan_reader_impl(
  std::shared_ptr<aggregate_reader_metadata> metadata)
{
  CUDF_EXPECTS(metadata != nullptr, "Shared parquet metadata must not be null");
  _metadata          = std::move(metadata);
  _extended_metadata = static_cast<aggregate_reader_metadata*>(_metadata.get());
}

std::vector<FileMetaData> hybrid_scan_reader_impl::parquet_metadatas() const
{
  return _extended_metadata->parquet_metadatas();
}

std::vector<byte_range_info> hybrid_scan_reader_impl::page_index_byte_ranges() const
{
  return _extended_metadata->page_index_byte_ranges();
}

void hybrid_scan_reader_impl::setup_page_indexes(
  cudf::host_span<cudf::host_span<uint8_t const> const> page_index_bytes) const
{
  _extended_metadata->setup_page_indexes(page_index_bytes);
}

void hybrid_scan_reader_impl::select_columns(read_columns_mode read_columns_mode,
                                             parquet_reader_options const& options)
{
  // Initialize reader configuration.
  initialize_reader_config(options);

  // Build column selection options directly from the user options.
  auto selection_options = make_column_selection_options(options);

  if (read_columns_mode == read_columns_mode::ALL_COLUMNS) {
    if (_is_all_columns_selected) { return; }

    // Select only columns required by the options and filter
    auto const select_column_names = get_column_projection(options);

    // Select only columns required by the options and filter.
    // Using as is from:
    // https://github.com/NVIDIA/cudf/blob/a8b25cd205dc5d04b9918dcb0b3abd6b8c4e4a74/cpp/src/io/parquet/reader_impl.cpp#L556-L569
    std::optional<std::vector<std::string>> filter_only_columns_names;
    if (options.get_filter().has_value() and select_column_names.has_value()) {
      filter_only_columns_names = parquet::detail::get_column_names_in_expression(
        options.get_filter(), *select_column_names, options, _extended_metadata->get_schema_tree());
      _num_filter_only_columns = filter_only_columns_names->size();
    }
    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _metadata->select_columns(select_column_names, filter_only_columns_names, selection_options);

    _is_all_columns_selected     = true;
    _is_filter_columns_selected  = false;
    _is_payload_columns_selected = false;
  } else if (read_columns_mode == read_columns_mode::FILTER_COLUMNS) {
    if (_is_filter_columns_selected) { return; }
    // Must not ignore missing filter columns
    selection_options.ignore_missing_columns = false;

    _filter_columns_names = cudf::io::parquet::detail::get_column_names_in_expression(
      options.get_filter(), {}, options, _extended_metadata->get_schema_tree());
    // Select only filter columns using the base `select_columns` method
    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _extended_metadata->select_columns(_filter_columns_names, {}, selection_options);

    _is_filter_columns_selected  = true;
    _is_payload_columns_selected = false;
    _is_all_columns_selected     = false;
  } else {
    if (_is_payload_columns_selected) { return; }

    auto select_column_names = get_column_projection(options);
    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _extended_metadata->select_payload_columns(
        select_column_names, _filter_columns_names, selection_options);

    _is_payload_columns_selected = true;
    _is_filter_columns_selected  = false;
    _is_all_columns_selected     = false;
  }

  // Reset the materialization step flag
  _output_chunk_produced = false;

  CUDF_EXPECTS(_input_columns.size() > 0 and _output_buffers.size() > 0, "No columns selected");

  // Save original output-buffer schema for reuse across materialization passes.
  _original_output_buffers_template = make_empty_like_column_buffers(_output_buffers);

  // Initialize mutable output-buffer template for this materialization pass.
  reset_output_buffers_template();
}

void hybrid_scan_reader_impl::reset_output_buffers_template()
{
  _output_buffers_template = make_empty_like_column_buffers(_original_output_buffers_template);
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::all_row_groups(
  parquet_reader_options const& options) const
{
  return _extended_metadata->all_row_groups(options);
}

std::size_t hybrid_scan_reader_impl::total_rows_in_row_groups(
  std::span<std::vector<size_type> const> row_group_indices) const
{
  return _extended_metadata->total_rows_in_row_groups(row_group_indices);
}

void hybrid_scan_reader_impl::reset_column_selection()
{
  _is_all_columns_selected     = false;
  _is_filter_columns_selected  = false;
  _is_payload_columns_selected = false;
}

std::pair<parquet_filter_normalizer, std::vector<cudf::data_type>>
hybrid_scan_reader_impl::prepare_filter_and_output_types(parquet_reader_options const& options)
{
  CUDF_EXPECTS(options.get_filter().has_value(), "Empty input filter expression encountered");

  select_columns(read_columns_mode::FILTER_COLUMNS, options);

  // Normalize the input expression (must be done after column selection)
  auto expr_conv     = build_normalized_expression(options);
  auto output_dtypes = get_output_types(_output_buffers_template);

  return {std::move(expr_conv), std::move(output_dtypes)};
}

void hybrid_scan_reader_impl::prepare_materialization(read_columns_mode read_columns_mode,
                                                      std::size_t num_sources,
                                                      parquet_reader_options const& options,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  reset_internal_state();
  initialize_options(options, num_sources, stream, mr);
  select_columns(read_columns_mode, options);
  reset_output_buffers_template();
}

std::vector<std::vector<cudf::size_type>>
hybrid_scan_reader_impl::filter_row_groups_with_byte_range(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  if (options.get_skip_bytes() == 0 and not options.get_num_bytes().has_value()) {
    return std::vector<std::vector<cudf::size_type>>{row_group_indices.begin(),
                                                     row_group_indices.end()};
  }

  return _extended_metadata->filter_row_groups_with_byte_range(
    row_group_indices, options.get_skip_bytes(), options.get_num_bytes());
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::filter_row_groups_with_stats(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->filter_row_groups_with_stats(row_group_indices,
                                                          output_dtypes,
                                                          _output_column_schemas,
                                                          expr_conv.get_converted_expr().value(),
                                                          stream);
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::dictionary_pages_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->dictionary_pages_byte_ranges(row_group_indices,
                                                          output_dtypes,
                                                          _output_column_schemas,
                                                          expr_conv.get_converted_expr().value());
}

std::pair<std::vector<byte_range_info>, std::vector<size_type>>
hybrid_scan_reader_impl::bloom_filters_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->bloom_filters_byte_ranges(row_group_indices,
                                                       output_dtypes,
                                                       _output_column_schemas,
                                                       expr_conv.get_converted_expr().value());
}

std::vector<std::vector<size_type>>
hybrid_scan_reader_impl::filter_row_groups_with_dictionary_pages(
  std::span<cudf::device_span<uint8_t const> const> dictionary_page_data,
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  // Collect literal and operator pairs for each input column with an (in)equality predicate
  auto const [literals, operators] =
    dictionary_literals_collector{expr_conv.get_converted_expr().value().get(), output_dtypes}
      .get_literals_and_operators();

  // Return all row groups if no dictionary page filtering is needed
  if (literals.empty() or std::all_of(literals.begin(), literals.end(), [](auto& col_literals) {
        return col_literals.empty();
      })) {
    return std::vector<std::vector<size_type>>(row_group_indices.begin(), row_group_indices.end());
  }

  // Collect schema indices of input columns with a non-empty (in)equality literal/operator vector
  std::vector<cudf::size_type> dictionary_col_schemas;
  thrust::copy_if(thrust::host,
                  _output_column_schemas.begin(),
                  _output_column_schemas.end(),
                  literals.begin(),
                  std::back_inserter(dictionary_col_schemas),
                  [](auto& dict_literals) { return not dict_literals.empty(); });

  // Prepare dictionary column chunks and decode page headers
  auto [has_compressed_data, chunks, pages] = prepare_dictionaries(
    row_group_indices, dictionary_page_data, dictionary_col_schemas, options, stream);

  // Decompress dictionary pages if needed and store uncompressed buffers here
  auto const mr                          = cudf::get_current_device_resource_ref();
  auto decompressed_dictionary_page_data = std::optional<rmm::device_buffer>{};
  if (has_compressed_data) {
    // Use the `decompress_page_data` utility to decompress dictionary pages (passed as pass_pages)
    decompressed_dictionary_page_data =
      std::get<0>(parquet::detail::decompress_page_data(chunks, pages, {}, {}, stream, mr));
    pages.host_to_device_async(stream);
  }

  // Filter row groups using dictionary pages
  return _extended_metadata->filter_row_groups_with_dictionary_pages(
    chunks,
    pages,
    row_group_indices,
    literals,
    operators,
    output_dtypes,
    dictionary_col_schemas,
    expr_conv.get_converted_expr().value(),
    stream);
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::filter_row_groups_with_bloom_filters(
  std::span<cudf::device_span<uint8_t const> const> bloom_filter_data,
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->filter_row_groups_with_bloom_filters(
    bloom_filter_data,
    row_group_indices,
    output_dtypes,
    _output_column_schemas,
    expr_conv.get_converted_expr().value(),
    stream);
}

std::unique_ptr<cudf::column> hybrid_scan_reader_impl::build_all_true_row_mask(
  std::span<std::vector<size_type> const> row_group_indices,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  return _extended_metadata->build_all_true_row_mask(row_group_indices, stream, mr);
}

std::unique_ptr<cudf::column> hybrid_scan_reader_impl::build_row_mask_with_page_index_stats(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->build_row_mask_with_page_index_stats(
    row_group_indices,
    output_dtypes,
    _output_column_schemas,
    expr_conv.get_converted_expr().value(),
    stream,
    mr);
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::get_input_column_chunk_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices) const
{
  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_row_groups    = std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    std::size_t{0},
    [](std::size_t sum, auto const& row_groups) { return sum + row_groups.size(); });
  auto const num_chunks = num_row_groups * num_input_columns;

  // Association between each column chunk and its source
  auto chunk_source_map = std::vector<size_type>{};
  chunk_source_map.reserve(num_chunks);

  // Keep track of column chunk byte ranges
  auto column_chunk_byte_ranges = std::vector<byte_range_info>{};
  column_chunk_byte_ranges.reserve(num_chunks);

  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{row_group_indices.size()},
                [&](auto const source_idx) {
                  auto const& row_groups = row_group_indices[source_idx];
                  for (auto const row_group_index : row_groups) {
                    // generate ColumnChunkDesc objects for everything to be decoded (all input
                    // columns)
                    for (auto const& col : _input_columns) {
                      // look up metadata
                      auto const& col_meta = _extended_metadata->get_column_metadata(
                        row_group_index, source_idx, col.schema_idx);

                      auto const chunk_offset =
                        (col_meta.dictionary_page_offset != 0)
                          ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
                          : col_meta.data_page_offset;

                      auto const chunk_size = col_meta.total_compressed_size;
                      column_chunk_byte_ranges.emplace_back(chunk_offset, chunk_size);

                      // Map each column chunk to its column index and its source index
                      chunk_source_map.emplace_back(static_cast<size_type>(source_idx));
                    }
                  }
                });

  return {std::move(column_chunk_byte_ranges), std::move(chunk_source_map)};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::filter_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  select_columns(read_columns_mode::FILTER_COLUMNS, options);
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::payload_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  select_columns(read_columns_mode::PAYLOAD_COLUMNS, options);
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::payload_pages_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  parquet_reader_options const& options,
  cuda::stream_ref stream)
{
  CUDF_EXPECTS(row_group_indices.size() == _extended_metadata->get_num_sources(),
               "Row group source count must match the number of input sources");
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when planning payload pages");

  select_columns(read_columns_mode::PAYLOAD_COLUMNS, options);

  auto column_schemas = std::vector<size_type>{};
  column_schemas.reserve(_input_columns.size());
  std::transform(_input_columns.begin(),
                 _input_columns.end(),
                 std::back_inserter(column_schemas),
                 [](auto const& col) { return col.schema_idx; });
  CUDF_EXPECTS(_extended_metadata->page_index_presence(row_group_indices, column_schemas).second,
               "Page-level I/O for payload columns requires offset indexes to be present");

  auto const num_columns = _input_columns.size();
  auto const num_chunks =
    static_cast<std::size_t>(count_row_groups(row_group_indices)) * num_columns;

  // The data page mask is ordered by column (all pages of a column, then the next column) so
  // accumulate page counts per column to locate each column's portion of the mask.
  auto mask_offsets = std::vector<std::size_t>(num_columns + 1, 0);

  // For each source
  for (std::size_t source_idx = 0; source_idx < row_group_indices.size(); ++source_idx) {
    // For each row group in the source
    auto colchunk_offsets = std::vector<std::optional<size_type>>(num_columns);
    for (auto const row_group_idx : row_group_indices[source_idx]) {
      // For each selected column chunk in the row group
      auto const& row_group = _extended_metadata->get_row_group(row_group_idx, source_idx);
      for (std::size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
        auto const schema_idx =
          _extended_metadata->map_schema_index(column_schemas[col_idx], source_idx);
        auto& colchunk_offset = colchunk_offsets[col_idx];
        colchunk_offset =
          parquet::detail::find_colchunk_iter_offset(row_group, schema_idx, colchunk_offset);
        // Accumulate page counts per column
        mask_offsets[col_idx + 1] +=
          row_group.columns[colchunk_offset.value()].offset_index->page_locations.size();
      }
    }
  }

  // Accumulate page counts per column
  std::partial_sum(mask_offsets.begin(), mask_offsets.end(), mask_offsets.begin());

  // Compute the data page mask
  auto const mask_size = mask_offsets.back();
  auto data_page_mask  = _extended_metadata->compute_data_page_mask(
    row_mask, row_group_indices, _input_columns, 0, stream);
  CUDF_EXPECTS(data_page_mask.empty() or data_page_mask.size() == mask_size,
               "Computed data page mask does not match offset indexes");

  // Generate page ranges (row group wise) and the corresponding source map
  auto page_ranges = std::vector<byte_range_info>{};
  auto source_map  = std::vector<cudf::size_type>{};
  page_ranges.reserve(mask_size + num_chunks);
  source_map.reserve(mask_size + num_chunks);

  // For each source
  for (std::size_t source_idx = 0; source_idx < row_group_indices.size(); ++source_idx) {
    auto colchunk_offsets = std::vector<std::optional<size_type>>(num_columns);
    // For each row group in the source
    for (auto const row_group_idx : row_group_indices[source_idx]) {
      auto const& row_group = _extended_metadata->get_row_group(row_group_idx, source_idx);
      // For each selected column chunk in the row group
      for (std::size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
        auto const schema_idx =
          _extended_metadata->map_schema_index(column_schemas[col_idx], source_idx);
        auto& colchunk_offset = colchunk_offsets[col_idx];
        colchunk_offset =
          parquet::detail::find_colchunk_iter_offset(row_group, schema_idx, colchunk_offset);
        auto const& column_chunk   = row_group.columns[colchunk_offset.value()];
        auto const& page_locations = column_chunk.offset_index->page_locations;
        auto const mask_offset     = mask_offsets[col_idx];
        auto const any_data_page_retained =
          data_page_mask.empty() or
          std::any_of(data_page_mask.begin() + mask_offset,
                      data_page_mask.begin() + mask_offset + page_locations.size(),
                      cuda::std::identity{});

        // Helper lambda to add a page's byte range and source index to the output vectors
        auto add_page_range = [&](bool is_page_retained, int64_t offset, int64_t size) {
          page_ranges.emplace_back(offset, is_page_retained ? size : 0);
          source_map.push_back(static_cast<size_type>(source_idx));
        };

        // Add dictionary page range if any of the data pages are also retained
        if (auto const dict_page_range = dictionary_page_range(column_chunk);
            dict_page_range.has_value()) {
          add_page_range(any_data_page_retained, dict_page_range->first, dict_page_range->second);
        }

        // Add data page ranges
        auto mask_iter = data_page_mask.cbegin() + mask_offset;
        for (auto const& location : page_locations) {
          add_page_range(data_page_mask.empty() or *mask_iter++,
                         location.offset,
                         static_cast<int64_t>(location.compressed_page_size));
        }
        // Update the mask offset for the next column
        mask_offsets[col_idx] += page_locations.size();
      }
    }
  }

  return {std::move(page_ranges), std::move(source_map)};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::all_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  select_columns(read_columns_mode::ALL_COLUMNS, options);
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

table_with_metadata hybrid_scan_reader_impl::materialize_filter_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::mutable_column_view& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(options.get_filter().has_value(), "Empty input filter expression encountered");

  prepare_materialization(
    read_columns_mode::FILTER_COLUMNS, row_group_indices.size(), options, stream, mr);

  // Normalize the input expression (must be done after prepare_materialization)
  _expr_conv = build_normalized_expression(options);

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::READ_ALL, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::FILTER_COLUMNS, row_mask);
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::READ_ALL, row_group_indices, column_chunk_data, data_page_mask);

  return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::FILTER_COLUMNS, row_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_payload_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when materializing payload column");

  prepare_materialization(
    read_columns_mode::PAYLOAD_COLUMNS, row_group_indices.size(), options, stream, mr);

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::READ_ALL, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::PAYLOAD_COLUMNS, row_mask);
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (not row_mask.is_empty() and mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::READ_ALL, row_group_indices, column_chunk_data, data_page_mask);

  return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::PAYLOAD_COLUMNS, row_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_all_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  prepare_materialization(
    read_columns_mode::ALL_COLUMNS, row_group_indices.size(), options, stream, mr);

  // Normalize the input expression after materialization preparation.
  _expr_conv = build_normalized_expression(options);

  prepare_data(read_mode::READ_ALL, row_group_indices, column_chunk_data, {});

  // Use the main reader's function
  auto result = reader_impl::read_chunk_internal(read_mode::READ_ALL);

  // base read_chunk_internal() does not update the output chunk produced flag
  _output_chunk_produced = true;

  return result;
}

void hybrid_scan_reader_impl::setup_chunking_for_filter_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(options.get_filter().has_value(), "Empty input filter expression encountered");
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");

  prepare_materialization(
    read_columns_mode::FILTER_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Normalize the input expression (must be done after prepare_materialization)
  _expr_conv = build_normalized_expression(options);

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::CHUNKED_READ, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return;
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::CHUNKED_READ, row_group_indices, column_chunk_data, data_page_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_filter_columns_chunk(
  cudf::mutable_column_view& row_mask)
{
  CUDF_EXPECTS(_file_preprocessed, "Chunking for filter columns not yet setup");

  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }

  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});
  return read_chunk_internal(read_mode::CHUNKED_READ, read_columns_mode::FILTER_COLUMNS, row_mask);
}

void hybrid_scan_reader_impl::setup_chunking_for_payload_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when materializing payload column");

  prepare_materialization(
    read_columns_mode::PAYLOAD_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::CHUNKED_READ, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return;
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (not row_mask.is_empty() and mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::CHUNKED_READ, row_group_indices, column_chunk_data, data_page_mask);
}

void hybrid_scan_reader_impl::setup_chunking_for_payload_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  std::span<cudf::device_span<uint8_t const> const> page_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when materializing payload column");

  prepare_materialization(
    read_columns_mode::PAYLOAD_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::CHUNKED_READ, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return;
  }

  // Check if offset indexes are present
  auto const num_columns = _input_columns.size();
  auto column_schemas    = std::vector<size_type>{};
  column_schemas.reserve(num_columns);
  std::transform(_input_columns.begin(),
                 _input_columns.end(),
                 std::back_inserter(column_schemas),
                 [](auto const& col) { return col.schema_idx; });
  CUDF_EXPECTS(_extended_metadata->page_index_presence(row_group_indices, column_schemas).second,
               "Page-level I/O for payload columns requires offset indexes to be present");

  // Mark that we are using page-level I/O for payload columns
  _sparse_page_io = true;

  // Data page mask in sparse mode will be computed directly from the page data span inside
  // `prepare_data() -> setup_sparse_compressed_data()`
  prepare_data(read_mode::CHUNKED_READ, row_group_indices, page_data, {});
}

table_with_metadata hybrid_scan_reader_impl::materialize_payload_columns_chunk(
  cudf::column_view const& row_mask)
{
  CUDF_EXPECTS(_file_preprocessed, "Chunking for payload columns not yet setup");

  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }
  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});
  return read_chunk_internal(read_mode::CHUNKED_READ, read_columns_mode::PAYLOAD_COLUMNS, row_mask);
}

void hybrid_scan_reader_impl::setup_chunking_for_all_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  prepare_materialization(
    read_columns_mode::ALL_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Normalize the input expression (must be done after column selection)
  _expr_conv = build_normalized_expression(options);

  prepare_data(read_mode::CHUNKED_READ, row_group_indices, column_chunk_data, {});
}

table_with_metadata hybrid_scan_reader_impl::materialize_all_columns_chunk()
{
  CUDF_EXPECTS(_file_preprocessed, "Chunking for all columns not yet setup");

  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }
  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});

  // Use the main reader's function for reading all columns
  auto result = reader_impl::read_chunk_internal(read_mode::CHUNKED_READ);

  // base read_chunk_internal() does not update the output chunk produced flag
  _output_chunk_produced = true;

  return result;
}

std::pair<std::vector<std::vector<cudf::size_type>>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::construct_row_group_passes(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::size_t total_row_groups,
  std::size_t pass_read_limit) const
{
  CUDF_EXPECTS(
    total_row_groups > 0, "Empty input row group indices encountered", std::invalid_argument);

  CUDF_EXPECTS(row_group_indices.size() == _extended_metadata->get_num_sources(),
               "Mismatch in the number of row group indices vectors and the number of input "
               "datasources",
               std::invalid_argument);

  if (pass_read_limit == 0) {
    return {
      std::vector<std::vector<cudf::size_type>>{row_group_indices.begin(), row_group_indices.end()},
      std::vector<cudf::size_type>{}};
  }

  CUDF_EXPECTS(
    pass_read_limit > 0, "Pass read limit must be greater than 0", std::invalid_argument);

  auto row_group_ids   = std::vector<std::pair<size_type, size_type>>{};
  auto row_group_sizes = std::vector<cudf::io::parquet::detail::row_group_size_info>{};
  row_group_ids.reserve(total_row_groups);
  row_group_sizes.reserve(total_row_groups);

  std::for_each(cuda::counting_iterator<cudf::size_type>(0),
                cuda::counting_iterator<cudf::size_type>(row_group_indices.size()),
                [&](auto const source_index) {
                  for (auto const rg_index : row_group_indices[source_index]) {
                    row_group_ids.emplace_back(rg_index, source_index);
                    // TODO(mh): Compute the row group size information over the selected columns
                    // instead
                    row_group_sizes.push_back(_extended_metadata->get_row_group_size_info(
                      rg_index, source_index, std::nullopt));
                  }
                });

  auto const comp_read_limit = static_cast<std::size_t>(
    pass_read_limit * cudf::io::parquet::detail::input_limit_compression_reserve);

  auto const pass_data =
    cudf::io::parquet::detail::compute_row_group_passes(row_group_sizes, comp_read_limit, 0);

  // Convert offset-based pass boundaries back to vectors of row group indices
  auto const& offsets = pass_data.pass_row_group_offsets;
  auto passes         = std::vector<std::vector<cudf::size_type>>{};
  passes.reserve(offsets.size() - 1);
  auto row_group_source_map       = std::vector<cudf::size_type>{};
  auto const has_multiple_sources = row_group_indices.size() > 1;
  if (has_multiple_sources) { row_group_source_map.reserve(row_group_ids.size()); }
  std::transform(offsets.begin(),
                 offsets.end() - 1,
                 offsets.begin() + 1,
                 std::back_inserter(passes),
                 [&](auto const start, auto const end) {
                   auto pass = std::vector<cudf::size_type>{};
                   pass.reserve(end - start);
                   std::for_each(row_group_ids.begin() + start,
                                 row_group_ids.begin() + end,
                                 [&](auto const& row_group_id) {
                                   pass.emplace_back(row_group_id.first);
                                   if (has_multiple_sources) {
                                     row_group_source_map.emplace_back(row_group_id.second);
                                   }
                                 });
                   return pass;
                 });
  return {std::move(passes), std::move(row_group_source_map)};
}

bool hybrid_scan_reader_impl::has_next_table_chunk()
{
  CUDF_EXPECTS(_file_preprocessed, "Chunking not yet setup");
  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});

  // current_input_pass will only be incremented to be == num_passes after
  // the last chunk in the last subpass in the last pass has been returned
  // if not has_more_work then check if this is the first pass in an empty
  // table and return true so it could be read once.
  return has_more_work() or is_first_output_chunk();
}

void hybrid_scan_reader_impl::reset_internal_state()
{
  _row_mask_offset   = 0;
  _file_itm_data     = file_intermediate_data{};
  _file_preprocessed = false;
  _has_offset_index  = false;
  _pass_itm_data.reset();
  _pass_page_mask.clear();
  _subpass_page_mask.reset();
  _output_metadata.reset();
  _sparse_page_io = false;

  _options.timestamp_type = cudf::data_type{};
  _options.decimal_width  = type_id::EMPTY;
  _options.num_rows       = std::nullopt;
  _options.row_group_indices.clear();
  _options.use_jit_filter              = false;
  _options.case_sensitive_names        = true;
  _options.prepend_source_index_column = false;
  _options.prepend_row_index_column    = false;

  _num_sources             = 0;
  _input_pass_read_limit   = 0;
  _output_chunk_read_limit = 0;
  _strings_to_categorical  = false;
  _reader_column_schema.reset();
  _expr_conv = parquet_filter_normalizer{};
  _mr        = cudf::get_current_device_resource_ref();
}

void hybrid_scan_reader_impl::initialize_reader_config(parquet_reader_options const& options)
{
  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  _options.timestamp_type              = cudf::data_type{options.get_timestamp_type().id()};
  _options.decimal_width               = options.get_decimal_width();
  _options.use_jit_filter              = options.is_enabled_use_jit_filter();
  _options.case_sensitive_names        = options.is_enabled_case_sensitive_names();
  _options.prepend_source_index_column = options.is_enabled_prepend_source_index_column();
  _options.prepend_row_index_column    = options.is_enabled_prepend_row_index_column();
}

void hybrid_scan_reader_impl::initialize_options(parquet_reader_options const& options,
                                                 std::size_t num_sources,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  // Binary columns can be read as binary or strings
  _reader_column_schema = options.get_column_schema();

  _num_sources = num_sources;

  // CUDA stream to use for internal operations
  _stream = stream;

  // Device memory resource to use for allocations
  _mr = mr;
}

parquet_filter_normalizer hybrid_scan_reader_impl::build_normalized_expression(
  parquet_reader_options const& options)
{
  if (not options.get_filter().has_value()) { return parquet_filter_normalizer{}; }

  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = parquet_filter_normalizer(options.get_filter(),
                                             metadata,
                                             _extended_metadata->get_schema_tree(),
                                             options,
                                             options.is_enabled_case_sensitive_names());
  CUDF_EXPECTS(expr_conv.get_converted_expr().has_value(),
               "Columns names in filter expression must be convertible to index references");
  return expr_conv;
}

void hybrid_scan_reader_impl::prepare_data(
  read_mode mode,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  host_span<bool const> data_page_mask)
{
  // if we have not preprocessed at the whole-file level, do that now
  if (not _file_preprocessed) {
    // setup file level information
    // - read row group information
    // - setup information on (parquet) chunks
    // - compute schedule of input passes
    prepare_row_groups(read_mode::READ_ALL, row_group_indices);
  }

  // handle any chunking work (ratcheting through the subpasses and chunks within
  // our current pass) if in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    handle_chunking(mode, column_chunk_data, data_page_mask);
  }
}

template <typename RowMaskView>
table_with_metadata hybrid_scan_reader_impl::read_chunk_internal(
  read_mode mode, read_columns_mode read_columns_mode, RowMaskView row_mask)
{
  // If `_output_metadata` has been constructed, just copy it over.
  auto out_metadata = _output_metadata ? table_metadata{*_output_metadata} : table_metadata{};
  out_metadata.schema_info.resize(_output_buffers.size());

  // output cudf columns as determined by the top level schema
  auto out_columns = std::vector<std::unique_ptr<column>>{};
  out_columns.reserve(_output_buffers.size());

  // Copy number of total input row groups and number of surviving row groups from predicate
  // pushdown.
  out_metadata.num_input_row_groups = _file_itm_data.num_input_row_groups;
  // Copy the number surviving row groups from each predicate pushdown only if the filter has value
  if (_expr_conv.get_converted_expr().has_value()) {
    out_metadata.num_row_groups_after_stats_filter =
      _file_itm_data.surviving_row_groups.after_stats_filter;
    out_metadata.num_row_groups_after_bloom_filter =
      _file_itm_data.surviving_row_groups.after_bloom_filter;
  }

  // no work to do (this can happen on the first pass if we have no rows to read)
  if (!has_more_work()) {
    // Empty dataframe case: Simply initialize to a list of zeros
    out_metadata.num_rows_per_source =
      std::vector<std::size_t>(_file_itm_data.num_rows_per_source.size(), 0);

    // Finalize output
    return finalize_output(read_columns_mode, out_metadata, out_columns, row_mask);
  }

  auto& pass            = *_pass_itm_data;
  auto& subpass         = *pass.subpass;
  auto const& read_info = subpass.output_chunk_read_info[subpass.current_output_chunk];

  // computes:
  // PageNestingInfo::batch_size for each level of nesting, for each page, taking row bounds into
  // account. PageInfo::skipped_values, which tells us where to start decoding in the input to
  // respect the user bounds. It is only necessary to do this second pass if uses_custom_row_bounds
  // is set (if the user has specified artificial bounds).
  if (uses_custom_row_bounds(mode)) {
    compute_page_sizes(subpass.pages,
                       pass.chunks,
                       subpass_page_mask_span(),
                       read_info.skip_rows,
                       read_info.num_rows,
                       false,  // num_rows is already computed
                       pass.level_type_size,
                       _stream);
  }

  // preprocess strings
  preprocess_chunk_strings(mode, read_info);

  // Allocate memory buffers for the output columns.
  allocate_columns(mode, read_info.skip_rows, read_info.num_rows);

  // Parse data into the output buffers.
  decode_page_data(mode, read_info.skip_rows, read_info.num_rows);

  // Create the final output cudf columns.
  for (std::size_t i = 0; i < _output_buffers.size(); ++i) {
    auto metadata           = _reader_column_schema.has_value()
                                ? std::make_optional<reader_column_schema>((*_reader_column_schema)[i])
                                : std::nullopt;
    auto const& schema      = _extended_metadata->get_schema(_output_column_schemas[i]);
    auto const logical_type = schema.logical_type.value_or(LogicalType{});
    // FIXED_LEN_BYTE_ARRAY never read as string.
    // TODO: if we ever decide that the default reader behavior is to treat unannotated BINARY
    // as binary and not strings, this test needs to change.
    if (schema.type == Type::FIXED_LEN_BYTE_ARRAY and logical_type.type != LogicalType::DECIMAL) {
      metadata = std::make_optional<reader_column_schema>();
      metadata->set_convert_binary_to_strings(false);
      metadata->set_type_length(schema.type_length);
    }
    // Only construct `out_metadata` if `_output_metadata` has not been cached.
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(make_column(_output_buffers[i], &col_name, metadata, _stream));
    } else {
      out_columns.emplace_back(make_column(_output_buffers[i], nullptr, metadata, _stream));
    }
  }

  out_columns =
    cudf::structs::detail::enforce_null_consistency(std::move(out_columns), _stream, _mr);

  // Compute the output number of rows per source
  if (mode == read_mode::CHUNKED_READ) {
    out_metadata.num_rows_per_source =
      calculate_output_num_rows_per_source(read_info.skip_rows, read_info.num_rows);
  } else {
    // Move is okay here as we are reading in one go.
    out_metadata.num_rows_per_source = std::move(_file_itm_data.num_rows_per_source);
  }

  // Add empty columns if needed. Filter output columns based on filter.
  return finalize_output(read_columns_mode, out_metadata, out_columns, row_mask);
}

template <typename RowMaskView>
table_with_metadata hybrid_scan_reader_impl::finalize_output(
  read_columns_mode read_columns_mode,
  table_metadata& out_metadata,
  std::vector<std::unique_ptr<column>>& out_columns,
  RowMaskView row_mask)
{
  // Create empty columns as needed (this can happen if we've ended up with no actual data to
  // read)
  for (std::size_t i = out_columns.size(); i < _output_buffers.size(); ++i) {
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], &col_name, _stream, _mr));
    } else {
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], nullptr, _stream, _mr));
    }
  }

  if (!_output_metadata) {
    populate_metadata(out_metadata);
    // Finally, save the output table metadata into `_output_metadata` for reuse next time.
    _output_metadata = std::make_unique<table_metadata>(out_metadata);
  }

  // Row-range of the current output chunk relative to the current row group selection.
  auto const read_info =
    (_file_itm_data._current_input_pass < _file_itm_data.num_passes())
      ? _pass_itm_data->subpass
          ->output_chunk_read_info[_pass_itm_data->subpass->current_output_chunk]
      : cudf::io::parquet::detail::row_range{0, 0};

  // advance output chunk/subpass/pass info for non-empty tables if and only if we are in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    auto& pass    = *_pass_itm_data;
    auto& subpass = *pass.subpass;
    subpass.current_output_chunk++;
  }

  // increment the output chunk count
  _file_itm_data._output_chunk_count++;

  apply_decimal_width_cast(out_columns);

  // Prepend the source and row index columns to filter columns only
  if (read_columns_mode == read_columns_mode::FILTER_COLUMNS) {
    if (_options.prepend_row_index_column) {
      out_columns.emplace(
        out_columns.begin(),
        synthesize_row_index_column(_file_itm_data.row_groups, read_info, _stream, _mr));
      out_metadata.schema_info.emplace(out_metadata.schema_info.begin(),
                                       column_name_info{.name = "row_index", .is_nullable = false});
    }
    if (_options.prepend_source_index_column) {
      out_columns.emplace(out_columns.begin(),
                          parquet::detail::synthesize_source_index_column(
                            out_metadata.num_rows_per_source, _stream, _mr));
      out_metadata.schema_info.emplace(
        out_metadata.schema_info.begin(),
        column_name_info{.name = "source_index", .is_nullable = false});
    }
  }

  // Create a table from the output columns.
  auto read_table = std::make_unique<table>(std::move(out_columns));

  // If the input row mask is empty, all rows are pruned anyway.
  if (row_mask.is_empty()) {
    _output_chunk_produced = true;
    return {std::move(read_table), std::move(out_metadata)};
  }

  CUDF_EXPECTS(row_mask.type().id() == type_id::BOOL8, "Input row mask must be a boolean column");

  // Get the current row mask offset
  auto const mask_offset = _row_mask_offset;
  // Update the row mask offset and the output chunk produced flag
  _row_mask_offset += read_table->num_rows();
  _output_chunk_produced = true;

  // Clear the number of rows per source as it is not valid after filtering
  out_metadata.num_rows_per_source.clear();

  // For filter columns, apply the filter expression and update the input row mask
  if constexpr (std::is_same_v<RowMaskView, cudf::mutable_column_view>) {
    CUDF_EXPECTS(read_columns_mode == read_columns_mode::FILTER_COLUMNS, "Invalid read mode");

    // Compute the final filter expression incorporating any column reference offsets in _expr_conv
    auto const final_filter      = compute_offset_filter();
    auto const final_filter_expr = final_filter.get_converted_expr();

    auto final_row_mask = cudf::detail::compute_column(*read_table,
                                                       final_filter_expr.value().get(),
                                                       _stream,
                                                       cudf::get_current_device_resource_ref());
    CUDF_EXPECTS(final_row_mask->view().type().id() == type_id::BOOL8,
                 "Predicate filter should return a boolean");

    // Apply the final row mask to get the final output table
    auto output_table = cudf::detail::apply_mask(
      read_table->view(), *final_row_mask, cudf::detail::mask_type::RETENTION, _stream, _mr);

    // Update the input row mask to reflect the final row mask.
    update_row_mask(final_row_mask->view(), row_mask, mask_offset, _stream);

    // Return the final output table and metadata
    return {std::move(output_table), std::move(out_metadata)};
  }
  // For payload columns, simply apply the input row mask to the table.
  else {
    CUDF_EXPECTS(read_columns_mode == read_columns_mode::PAYLOAD_COLUMNS, "Invalid read mode");

    CUDF_EXPECTS(mask_offset + read_table->num_rows() <= row_mask.size(),
                 "Encountered invalid sized row mask to apply");
    auto effective_row_mask =
      (read_table->num_rows() == row_mask.size())
        ? row_mask
        : cudf::split(row_mask, {mask_offset, mask_offset + read_table->num_rows()}, _stream)[1];
    auto output_table = cudf::detail::apply_mask(
      read_table->view(), effective_row_mask, cudf::detail::mask_type::RETENTION, _stream, _mr);
    return {std::move(output_table), std::move(out_metadata)};
  }
}

void hybrid_scan_reader_impl::set_pass_page_mask(std::span<bool const> data_page_mask)
{
  auto const& pass   = _pass_itm_data;
  auto const& chunks = pass->chunks;

  _pass_page_mask        = cudf::detail::make_empty_host_vector<bool>(pass->pages.size(), _stream);
  auto const num_columns = _input_columns.size();

  // Handle the empty page mask case
  if (data_page_mask.empty()) {
    std::fill(_pass_page_mask.begin(), _pass_page_mask.end(), true);
    return;
  }

  std::size_t num_inserted_data_pages = 0;
  std::for_each(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{_input_columns.size()},
    [&](auto col_idx) {
      for (std::size_t chunk_idx = col_idx; chunk_idx < chunks.size(); chunk_idx += num_columns) {
        // Number of data pages in this column chunk
        auto const num_data_pages_this_col_chunk = chunks[chunk_idx].num_data_pages;

        // Make sure we have enough page mask for this column chunk
        CUDF_EXPECTS(
          data_page_mask.size() >= num_inserted_data_pages + num_data_pages_this_col_chunk,
          "Encountered invalid data page mask size");

        // Insert a true value for each dictionary page
        _pass_page_mask.insert(_pass_page_mask.end(), chunks[chunk_idx].num_dict_pages, true);

        // Insert page mask for this column chunk
        _pass_page_mask.insert(
          _pass_page_mask.end(),
          data_page_mask.begin() + num_inserted_data_pages,
          data_page_mask.begin() + num_inserted_data_pages + num_data_pages_this_col_chunk);
        // Update the number of inserted data pages
        num_inserted_data_pages += num_data_pages_this_col_chunk;
      }
    });

  // Make sure we inserted exactly the number of pages for this pass
  CUDF_EXPECTS(_pass_page_mask.size() == pass->pages.size(),
               "Encountered mismatch in number of pass pages and page mask size");

  // Mark output buffers nullable when page pruning produces nulls
  mark_buffers_nullable_for_pruned_pages();
}

void hybrid_scan_reader_impl::set_sparse_pass_page_mask(
  std::span<cudf::device_span<uint8_t const> const> page_data)
{
  auto const& pass   = _pass_itm_data;
  auto const& chunks = pass->chunks;

  _pass_page_mask = cudf::detail::make_empty_host_vector<bool>(pass->pages.size(), _stream);

  // Find the first logical page-data span for every column chunk.
  auto page_offsets = std::vector<std::size_t>{};
  page_offsets.reserve(chunks.size());
  auto const num_logical_pages = std::accumulate(
    chunks.begin(), chunks.end(), std::size_t{0}, [&](auto offset, auto const& chunk) {
      page_offsets.push_back(offset);
      return offset + chunk.num_dict_pages + chunk.num_data_pages;
    });
  CUDF_EXPECTS(page_data.size() == num_logical_pages,
               "Sparse page span count does not match the number of logical pages");

  auto const num_columns = _input_columns.size();
  // Build the internal mask in column/chunk order.
  std::for_each(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{num_columns},
    [&](auto col_idx) {
      for (std::size_t chunk_idx = col_idx; chunk_idx < chunks.size(); chunk_idx += num_columns) {
        auto const& chunk        = chunks[chunk_idx];
        auto const data_page_idx = page_offsets[chunk_idx] + chunk.num_dict_pages;
        auto const data_page_end = data_page_idx + chunk.num_data_pages;

        // Retain a dictionary whenever the column chunk has a retained data page.
        if (chunk.num_dict_pages > 0) {
          _pass_page_mask.push_back(std::any_of(page_data.begin() + data_page_idx,
                                                page_data.begin() + data_page_end,
                                                [](auto const& page) { return not page.empty(); }));
        }
        // Insert page-mask values directly from the corresponding data-page spans.
        std::transform(page_data.begin() + data_page_idx,
                       page_data.begin() + data_page_end,
                       std::back_inserter(_pass_page_mask),
                       [](auto const& page) { return not page.empty(); });
      }
    });
  // Make sure we inserted exactly the number of pages for this pass.
  CUDF_EXPECTS(_pass_page_mask.size() == pass->pages.size(),
               "Encountered mismatch in number of pass pages and page mask size");

  // Mark output buffers nullable when page pruning produces nulls
  mark_buffers_nullable_for_pruned_pages();
}

}  // namespace cudf::io::parquet::experimental::detail
