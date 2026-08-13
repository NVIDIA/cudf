/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file hybrid_scan_impl.hpp
 * @brief cuDF-IO experimental Parquet reader class implementation header
 */

#pragma once

#include "hybrid_scan_helpers.hpp"
#include "io/parquet/parquet_gpu.hpp"
#include "io/parquet/reader_impl.hpp"

#include <cudf/io/detail/utils.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/host_vector.h>

#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

using cudf::io::parquet::detail::ColumnChunkDesc;
using cudf::io::parquet::detail::PageInfo;
using text::byte_range_info;

/**
 * @brief Implementation of the experimental Parquet reader optimized for Hybrid Scan operation
 */
class hybrid_scan_reader_impl : public parquet::detail::reader_impl {
 public:
  /**
   * @brief Constructor for the experimental parquet reader implementation
   *
   * @param footer_bytes Span of parquet file footer byte spans, one per source
   * @param options Parquet reader options
   */
  explicit hybrid_scan_reader_impl(
    cudf::host_span<cudf::host_span<uint8_t const> const> footer_bytes,
    parquet_reader_options const& options);

  /**
   * @brief Constructor for the experimental parquet reader implementation
   *
   * @param parquet_metadatas Span of pre-populated Parquet file metadata, one per source
   * @param options Parquet reader options
   */
  explicit hybrid_scan_reader_impl(cudf::host_span<FileMetaData const> parquet_metadatas,
                                   parquet_reader_options const& options);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::parquet_metadatas
   */
  [[nodiscard]] std::vector<FileMetaData> parquet_metadatas() const;

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::page_index_byte_ranges
   */
  [[nodiscard]] std::vector<byte_range_info> page_index_byte_ranges() const;

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::setup_page_indexes
   */
  void setup_page_indexes(
    cudf::host_span<cudf::host_span<uint8_t const> const> page_index_bytes) const;

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::all_row_groups
   */
  [[nodiscard]] std::vector<std::vector<size_type>> all_row_groups(
    parquet_reader_options const& options) const;

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::total_rows_in_row_groups
   */
  [[nodiscard]] std::size_t total_rows_in_row_groups(
    std::span<std::vector<size_type> const> row_group_indices) const;

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan::reset_column_selection
   */
  void reset_column_selection();

  /**
   * @copydoc
   * cudf::io::parquet::experimental::hybrid_scan_multifile::filter_row_groups_with_byte_range
   */
  [[nodiscard]] std::vector<std::vector<cudf::size_type>> filter_row_groups_with_byte_range(
    std::span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options) const;

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::filter_row_groups_with_stats
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_stats(
    std::span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::secondary_filters_byte_ranges
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<byte_range_info>>
  secondary_filters_byte_ranges(std::span<std::vector<size_type> const> row_group_indices,
                                parquet_reader_options const& options);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::dictionary_pages_byte_ranges
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
  dictionary_pages_byte_ranges(cudf::host_span<std::vector<size_type> const> row_group_indices,
                               parquet_reader_options const& options);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan::filter_row_groups_with_dictionary_pages
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_dictionary_pages(
    std::span<cudf::device_span<uint8_t const> const> dictionary_page_data,
    std::span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan::filter_row_groups_with_bloom_filters
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_bloom_filters(
    std::span<cudf::device_span<uint8_t const> const> bloom_filter_data,
    std::span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::build_all_true_row_mask
   */
  [[nodiscard]] std::unique_ptr<cudf::column> build_all_true_row_mask(
    std::span<std::vector<size_type> const> row_group_indices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @copydoc
   * cudf::io::parquet::experimental::hybrid_scan_multifile::build_row_mask_with_page_index_stats
   */
  [[nodiscard]] std::unique_ptr<cudf::column> build_row_mask_with_page_index_stats(
    std::span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @brief Fetches byte ranges of column chunks of filter columns
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Pair of a vector of byte ranges to column chunks of filter columns and a vector of
   *         their corresponding input source file indices
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
  filter_column_chunks_byte_ranges(std::span<std::vector<size_type> const> row_group_indices,
                                   parquet_reader_options const& options);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::materialize_filter_columns
   */
  [[nodiscard]] table_with_metadata materialize_filter_columns(
    std::span<std::vector<size_type> const> row_group_indices,
    std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
    cudf::mutable_column_view& row_mask,
    use_data_page_mask mask_data_pages,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @brief Fetches byte ranges of column chunks of payload columns
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Pair of a vector of byte ranges to column chunks of payload columns and a vector of
   * their corresponding input source file indices
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
  payload_column_chunks_byte_ranges(std::span<std::vector<size_type> const> row_group_indices,
                                    parquet_reader_options const& options);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::materialize_payload_columns
   */
  [[nodiscard]] table_with_metadata materialize_payload_columns(
    std::span<std::vector<size_type> const> row_group_indices,
    std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
    cudf::column_view const& row_mask,
    use_data_page_mask mask_data_pages,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::all_column_chunks_byte_ranges
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
  all_column_chunks_byte_ranges(std::span<std::vector<size_type> const> row_group_indices,
                                parquet_reader_options const& options);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::materialize_all_columns
   */
  [[nodiscard]] table_with_metadata materialize_all_columns(
    std::span<std::vector<size_type> const> row_group_indices,
    std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @copydoc
   * cudf::io::parquet::experimental::hybrid_scan_multifile::setup_chunking_for_filter_columns
   */
  void setup_chunking_for_filter_columns(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    std::span<std::vector<size_type> const> row_group_indices,
    cudf::column_view const& row_mask,
    use_data_page_mask mask_data_pages,
    std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @copydoc
   * cudf::io::parquet::experimental::hybrid_scan_multifile::materialize_filter_columns_chunk
   */
  [[nodiscard]] table_with_metadata materialize_filter_columns_chunk(
    cudf::mutable_column_view& row_mask);

  /**
   * @copydoc
   * cudf::io::parquet::experimental::hybrid_scan_multifile::setup_chunking_for_payload_columns
   */
  void setup_chunking_for_payload_columns(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    std::span<std::vector<size_type> const> row_group_indices,
    cudf::column_view const& row_mask,
    use_data_page_mask mask_data_pages,
    std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @copydoc
   * cudf::io::parquet::experimental::hybrid_scan_multifile::materialize_payload_columns_chunk
   */
  [[nodiscard]] table_with_metadata materialize_payload_columns_chunk(
    cudf::column_view const& row_mask);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::setup_chunking_for_all_columns
   */
  void setup_chunking_for_all_columns(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    std::span<std::vector<size_type> const> row_group_indices,
    std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::materialize_all_columns_chunk
   */
  [[nodiscard]] table_with_metadata materialize_all_columns_chunk();

  /**
   * @brief Partition per-source row groups into read passes
   *
   * @throws std::invalid_argument if @p row_group_indices.size() is all empty or not equal to the
   * number of input datasources
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param total_row_groups Total number of row groups across all sources
   * @param pass_read_limit Memory limit to read and decompress row
   * group data
   *
   * @return Pair of a vector of flattened row group passes and a source index map. The source index
   * map is empty for single source input
   */
  [[nodiscard]] std::pair<std::vector<std::vector<cudf::size_type>>, std::vector<cudf::size_type>>
  construct_row_group_passes(cudf::host_span<std::vector<size_type> const> row_group_indices,
                             std::size_t total_row_groups,
                             std::size_t pass_read_limit) const;

  /**
   * @copydoc cudf::io::parquet::experimental::hybrid_scan_multifile::has_next_table_chunk
   */
  [[nodiscard]] bool has_next_table_chunk();

 private:
  /**
   * @brief Enum indicating whether we are reading the filter, payload, or all columns
   */
  enum class read_columns_mode { FILTER_COLUMNS, PAYLOAD_COLUMNS, ALL_COLUMNS };

  /**
   * @brief Populate the reader's `_options` config (and related members) from the user options.
   *
   * @param options Reader options
   */
  void initialize_reader_config(parquet_reader_options const& options);

  /**
   * @brief Initialize the necessary options related internal variables for use later on
   *
   * @param options Reader options
   * @param num_sources Number of input sources
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   */
  void initialize_options(parquet_reader_options const& options,
                          std::size_t num_sources,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr);

  /**
   * @brief Convert the input filter expression such that all column name references are replaced
   * with corresponding column references
   *
   * @param options Reader options
   * @return Converted expression
   */
  [[nodiscard]] named_to_reference_converter build_converted_expression(
    parquet_reader_options const& options);

  /**
   * @brief Set the page mask for the pass pages
   *
   * @param data_page_mask Input data page mask from page-pruning step
   */
  void set_pass_page_mask(std::span<bool const> data_page_mask);

  /**
   * @brief Select the columns to be read based on the read mode
   *
   * @param read_columns_mode Read mode indicating if we are reading filter or payload columns
   * @param options Reader options
   */
  void select_columns(read_columns_mode read_columns_mode, parquet_reader_options const& options);

  /**
   * @brief Get the byte ranges for the input column chunks
   *
   * @param row_group_indices The row groups indices to read
   * @return A pair of vectors containing the byte ranges and the source indices
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
  get_input_column_chunk_byte_ranges(
    std::span<std::vector<size_type> const> row_group_indices) const;

  /**
   * @brief Helper to prepare converted filter expression and output column data types
   *
   * @param options Parquet reader options
   * @return A pair of a converted filter expression and a vector of output column data types
   */
  std::pair<named_to_reference_converter, std::vector<cudf::data_type>>
  prepare_filter_and_output_types(parquet_reader_options const& options);

  /**
   * @brief Helper to prepare column materialization
   *
   * @param read_columns_mode Read mode indicating if we are reading filter or payload columns
   * @param num_sources Number of input sources
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the device memory for the output columns
   */
  void prepare_materialization(read_columns_mode read_columns_mode,
                               std::size_t num_sources,
                               parquet_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

  /**
   * @brief Perform the necessary data preprocessing for parsing file later on
   *
   * Only ever called once for filter and payload columns. This function prepares the input row
   * groups and computes the schedule of top level passes (see `pass_intermediate_data`) and the
   * schedule of subpasses (see `subpass_intermediate_data`).
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param row_group_indices Row group indices to read
   * @param column_chunk_data Device spans of buffers containing column chunk data
   * @param data_page_mask Input data page mask from page-pruning step
   */
  void prepare_data(read_mode mode,
                    std::span<std::vector<size_type> const> row_group_indices,
                    std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
                    host_span<bool const> data_page_mask);

  /**
   * @brief Create descriptors for filter column chunks and decode dictionary page headers
   *
   * @param row_group_indices The row groups to read
   * @param dictionary_page_data Device buffers containing dictionary page data
   * @param dictionary_col_schemas Schema indices of output columns with (in)equality predicate
   * @param options Parquet reader options
   * @param stream CUDA stream
   *
   * @return A tuple of a boolean indicating if any of the chunks have compressed data, a host
   * device vector of column chunk descriptors, and a host device vector of decoded dictionary page
   * headers
   */
  std::tuple<bool,
             cudf::detail::hostdevice_vector<ColumnChunkDesc>,
             cudf::detail::hostdevice_vector<PageInfo>>
  prepare_dictionaries(std::span<std::vector<size_type> const> row_group_indices,
                       std::span<cudf::device_span<uint8_t const> const> dictionary_page_data,
                       std::span<int const> dictionary_col_schemas,
                       parquet_reader_options const& options,
                       rmm::cuda_stream_view stream);

  /**
   * @brief Prepares the select input row groups and associated chunk information
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param row_group_indices Row group indices to read
   */
  void prepare_row_groups(read_mode mode,
                          std::span<std::vector<size_type> const> row_group_indices);

  /**
   * @brief Ratchet the pass/subpass/chunk process forward.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param column_chunk_data Device spans of buffers containing column chunk data
   * @param data_page_mask Input data page mask from page-pruning step for the current pass
   */
  void handle_chunking(read_mode mode,
                       std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
                       host_span<bool const> data_page_mask);

  /**
   * @brief Setup step for the next input read pass.
   *
   * A 'pass' is defined as a subset of row groups read out of the globally
   * requested set of all row groups.
   *
   * @param column_chunk_data Device spans of buffers containing column chunk data
   */
  void setup_next_pass(std::span<cudf::device_span<uint8_t const> const> column_chunk_data);

  /**
   * @brief Setup pointers to columns chunks to be processed for this pass.
   *
   * Does not decompress the chunk data.
   *
   * @param column_chunk_data Device spans of buffers containing column chunk data
   * @return boolean indicating if compressed chunks were found
   */
  bool setup_column_chunks(std::span<cudf::device_span<uint8_t const> const> column_chunk_data);

  /**
   * @brief Setup compressed column chunks data and decode page headers for the current pass.
   *
   * @param column_chunk_data Device spans of buffers containing column chunk data
   */
  void setup_compressed_data(std::span<cudf::device_span<uint8_t const> const> column_chunk_data);

  /**
   * @brief Reset the internal state of the reader.
   */
  void reset_internal_state();

  /**
   * @brief Finalize the output table by adding empty columns for the non-selected columns in
   * schema.
   *
   * @tparam RowMaskView View type of the row mask column
   *
   * @param[in] read_columns_mode Read mode indicating if we are reading filter or payload columns
   * @param[in,out] out_metadata The output table metadata
   * @param[in,out] out_columns The columns for building the output table
   * @param[in,out] row_mask Boolean column indicating which rows need to be read after page-pruning
   *                         for filter columns, or after materialize step for payload columns
   * @return The output table along with columns' metadata
   */
  template <typename RowMaskView>
  table_with_metadata finalize_output(read_columns_mode read_columns_mode,
                                      table_metadata& out_metadata,
                                      std::vector<std::unique_ptr<column>>& out_columns,
                                      RowMaskView row_mask);

  /**
   * @brief Read a chunk of data and return an output table.
   *
   * This function is called internally and expects all preprocessing steps have already been done.
   *
   * @tparam RowMaskView View type of the row mask column
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param[in] read_columns_mode Read mode indicating if we are reading filter or payload columns
   * @param[in,out] row_mask Boolean column indicating which rows need to be read after page-pruning
   *                         for filter columns, or after materialize step for payload columns
   * @return The output table along with columns' metadata
   */
  template <typename RowMaskView>
  table_with_metadata read_chunk_internal(read_mode mode,
                                          read_columns_mode read_columns_mode,
                                          RowMaskView row_mask);

  /**
   * @brief Check if all rows are pruned (all valid and false)
   *
   * @param row_mask Input row mask column
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return True if all rows are pruned
   */
  [[nodiscard]] bool are_all_rows_pruned(cudf::column_view const& row_mask,
                                         rmm::cuda_stream_view stream) const;

  /**
   * @brief Updates the output row mask such that out_row_mask[i + out_row_mask_offset] = true if
   * and only if in_row_mask[i] is valid and true
   *
   * Updates the output row mask to reflect the final valid and surviving rows from the input row
   * mask. This is inline with the masking behavior of cudf::detail::apply_boolean_mask
   *
   * @param in_row_mask Input row mask column
   * @param out_row_mask Output row mask column
   * @param out_row_mask_offset Offset into the output row mask column
   * @param stream CUDA stream
   */
  void update_row_mask(cudf::column_view const& in_row_mask,
                       cudf::mutable_column_view& out_row_mask,
                       cudf::size_type out_row_mask_offset,
                       rmm::cuda_stream_view stream);

  /**
   * @brief Check if this is the first output chunk
   *
   * @return True if this is the first output chunk
   */
  [[nodiscard]] bool is_first_output_chunk() const
  {
    return _file_itm_data._output_chunk_count == 0 and not _output_chunk_produced;
  }

  aggregate_reader_metadata* _extended_metadata;

  std::optional<std::vector<std::string>> _filter_columns_names;

  cudf::size_type _row_mask_offset{0};
  bool _output_chunk_produced{false};

  bool _is_filter_columns_selected{false};
  bool _is_payload_columns_selected{false};
  bool _is_all_columns_selected{false};
};

}  // namespace cudf::io::parquet::experimental::detail
