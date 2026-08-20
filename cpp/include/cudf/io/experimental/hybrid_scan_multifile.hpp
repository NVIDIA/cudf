/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>
#include <span>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Multi-file variant of the experimental Hybrid Scan Parquet reader.
 */

namespace cudf::io::parquet::experimental::detail {
/**
 * @brief Internal experimental Parquet reader optimized for highly selective filters, called a
 *        Hybrid Scan operation.
 */
class hybrid_scan_reader_impl;
}  // namespace cudf::io::parquet::experimental::detail

//! Using `byte_range_info` from cudf::io::text
using cudf::io::text::byte_range_info;

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental {
/**
 * @addtogroup io_readers
 * @{
 */

/**
 * @brief Multi-file variant of the experimental Hybrid Scan Parquet reader
 *
 * Vectorizes `hybrid_scan_reader` APIs to support multiple Parquet sources. Inputs and outputs are
 * indexed by source order except for the row mask which is a single BOOL8 column spanning all rows
 * from all sources concatenated in source order, then row-group order within a source.
 *
 * @note Detailed usage documentation will be added once all APIs are in place. This reader will
 * eventually move to `hybrid_scan.hpp` and the existing single-file reader (`hybrid_scan_reader`)
 * will become its subclass. Only keeping this separate here for now to reduce noise.
 */
class hybrid_scan_multifile {
 public:
  /**
   * @brief Constructor for the multi-file experimental Parquet reader
   *
   * @param footer_bytes Host span of Parquet file footer byte spans, one per source
   * @param options Parquet reader options
   */
  explicit hybrid_scan_multifile(cudf::host_span<cudf::host_span<uint8_t const> const> footer_bytes,
                                 parquet_reader_options const& options);

  /**
   * @brief Constructor for the multi-file experimental Parquet reader
   *
   * @param parquet_metadata Host span of pre-populated Parquet file metadata, one per source
   * @param options Parquet reader options
   */
  explicit hybrid_scan_multifile(cudf::host_span<FileMetaData const> parquet_metadata,
                                 parquet_reader_options const& options);

  /**
   * @brief Destructor for the multi-file experimental Parquet reader
   */
  ~hybrid_scan_multifile();

  /**
   * @brief Get parquet metadatas for all sources
   *
   * @return Vector of parquet metadata, one per source
   */
  [[nodiscard]] std::vector<FileMetaData> parquet_metadatas() const;

  /**
   * @brief Get byte ranges of the page index for all sources
   *
   * @return Vector of page index byte ranges, one per source
   */
  [[nodiscard]] std::vector<byte_range_info> page_index_byte_ranges() const;

  /**
   * @brief Setup the per-source page index within each Parquet file metadata
   *
   * @param page_index_bytes Host span of Parquet page index buffer bytes, one per source
   */
  void setup_page_indexes(
    cudf::host_span<cudf::host_span<uint8_t const> const> page_index_bytes) const;

  /**
   * @brief Get all available per-source row group indices from the parquet files
   *
   * @param options Parquet reader options
   * @return Vector of vectors of row group indices, one per source
   */
  [[nodiscard]] std::vector<std::vector<size_type>> all_row_groups(
    parquet_reader_options const& options) const;

  /**
   * @brief Get the total number of top-level rows in the per-source row groups
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @return Total number of top-level rows across all sources
   */
  [[nodiscard]] size_type total_rows_in_row_groups(
    cudf::host_span<std::vector<size_type> const> row_group_indices) const;

  /**
   * @brief Resets the current column selection
   *
   * Resets the current column selection state forcing column re-selection in subsequent filter,
   * byte range, setup chunking and materialization APIs. This is useful if the filter expression
   * has been cascaded (and-ed) to include new columns.
   */
  void reset_column_selection() const;

  /**
   * @brief Filter the row groups using the byte range specified by [`bytes_to_skip`,
   * `bytes_to_skip + bytes_to_read`)
   *
   * Filters the row groups such that only the row groups that start within the byte range are
   * selected. Note that the last selected row group may end beyond the byte range.
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param options Parquet reader options
   * @return Vector of vectors of filtered row group indices, one per source
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_byte_range(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options) const;

  /**
   * @brief Filter the input row groups using column chunk statistics
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Vector of vectors of filtered row group indices, one per source
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_stats(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options,
    cuda::stream_ref stream) const;

  /**
   * @brief Get byte ranges of bloom filters for row group pruning
   *
   * @note Device buffers for bloom filter byte ranges must be allocated using a 32 byte
   *       aligned memory resource
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param options Parquet reader options
   * @return A pair of vectors containing bloom filter byte ranges and corresponding source indices
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<size_type>>
  bloom_filters_byte_ranges(std::span<std::vector<size_type> const> row_group_indices,
                            parquet_reader_options const& options) const;

  /**
   * @brief Filter the row groups using column chunk bloom filters
   *
   * @param bloom_filter_data Device spans of header-stripped bloom filter bitsets of column
   *                           chunks with an equality predicate, ordered to match the bloom filter
   *                           byte ranges returned by `bloom_filters_byte_ranges`
   * @param row_group_indices Input row group indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Vectors of filtered per-source row group indices, one per source
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_bloom_filters(
    std::span<cudf::device_span<uint8_t const> const> bloom_filter_data,
    std::span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options,
    cuda::stream_ref stream) const;

  /**
   * @brief Get byte ranges of column chunk dictionary pages for row group pruning
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param options Parquet reader options
   * @return Pair of flattened byte ranges to column chunk dictionary pages subject to the filter
   *         predicate and their corresponding source indices
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<size_type>>
  dictionary_pages_byte_ranges(cudf::host_span<std::vector<size_type> const> row_group_indices,
                               parquet_reader_options const& options) const;

  /**
   * @brief Filter the row groups using column chunk dictionary pages
   *
   * @param dictionary_page_data Device spans of dictionary page data of column chunks with an
   * (in)equality predicate, in the same order as the byte ranges returned by
   * `dictionary_pages_byte_ranges` including empty spans against empty byte ranges
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Vector of vectors of filtered row group indices, one per source
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_dictionary_pages(
    cudf::host_span<cudf::device_span<uint8_t const> const> dictionary_page_data,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options,
    cuda::stream_ref stream) const;

  /**
   * @brief Builds a boolean survival column of size equal to the total number of rows in the row
   * groups containing all `true` values
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return An all-true boolean (survival) column spanning all selected rows across all sources
   */
  [[nodiscard]] std::unique_ptr<cudf::column> build_all_true_row_mask(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Builds a boolean column indicating surviving rows using page-level statistics in the
   * page index
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return A boolean column spanning all selected rows across all sources and indicating which
   * filter column rows survive the statistics in the page index
   */
  [[nodiscard]] std::unique_ptr<cudf::column> build_row_mask_with_page_index_stats(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Get byte ranges of column chunks of filter columns
   *
   * Byte ranges are flattened in source order. Within each source, byte ranges follow the selected
   * row group and column chunk order used by `row_group_indices` and `options`. The returned source
   * map has one source index per byte range and can be used to regroup byte ranges by datasource
   * before fetching.
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param options Parquet reader options
   * @return Pair of flattened byte ranges to column chunks of filter columns and their
   *         corresponding source indices
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<size_type>>
  filter_column_chunks_byte_ranges(cudf::host_span<std::vector<size_type> const> row_group_indices,
                                   parquet_reader_options const& options) const;

  /**
   * @brief Materializes filter columns and updates the input row mask to only the rows that exist
   * in the output table
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param column_chunk_data Flattened device spans of filter column chunk data returned in the
   * same order as `filter_column_chunks_byte_ranges`
   * @param[in,out] row_mask Mutable boolean column spanning all selected rows across all sources
   * and indicating surviving rows from page pruning
   * @param mask_data_pages Whether to build and use a data page mask using the row mask
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the device memory for the output table
   * @return Table of materialized filter columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_filter_columns(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    cudf::mutable_column_view& row_mask,
    use_data_page_mask mask_data_pages,
    parquet_reader_options const& options,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Get byte ranges of column chunks of payload columns
   *
   * Byte ranges are flattened in source order. Within each source, byte ranges follow the selected
   * row group and column chunk order used by `row_group_indices` and `options`. The returned source
   * map has one source index per byte range and can be used to regroup byte ranges by datasource
   * before fetching.
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param options Parquet reader options
   * @return Pair of flattened byte ranges to column chunks of payload columns and their
   *         corresponding source indices
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<size_type>>
  payload_column_chunks_byte_ranges(cudf::host_span<std::vector<size_type> const> row_group_indices,
                                    parquet_reader_options const& options) const;

  /**
   * @brief Materialize payload columns and applies the row mask to the output table
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param column_chunk_data Flattened device spans of payload column chunk data returned in the
   *        same order as `payload_column_chunks_byte_ranges`
   * @param row_mask Boolean column spanning all selected rows across all sources and indicating
   *        which rows need to be read
   * @param mask_data_pages Whether to build and use a data page mask using the row mask
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the device memory for the output table
   * @return Table of materialized payload columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_payload_columns(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    cudf::column_view const& row_mask,
    use_data_page_mask mask_data_pages,
    parquet_reader_options const& options,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Get byte ranges of column chunks of all (or selected) columns
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param options Parquet reader options
   * @return Pair of flattened byte ranges to column chunks of all (or selected) columns and their
   *         corresponding source indices
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<size_type>>
  all_column_chunks_byte_ranges(cudf::host_span<std::vector<size_type> const> row_group_indices,
                                parquet_reader_options const& options) const;

  /**
   * @brief Materializes all (or selected) columns and returns the final output table
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param column_chunk_data Flattened device spans of column chunk data returned in the same order
   *        as `all_column_chunks_byte_ranges`
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the device memory for the output table
   * @return Table of all materialized columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_all_columns(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Setup chunking information for filter columns and preprocess the input data pages
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per table chunk. `0` if
   * there is no limit
   * @param pass_read_limit Limit on the memory used for reading and decompressing data. `0` if
   * there is no limit
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param row_mask Boolean column spanning all selected rows across all sources and indicating
   * which rows need to be read
   * @param mask_data_pages Whether to build and use a data page mask using the row mask
   * @param column_chunk_data Flattened device spans of filter column chunk data returned in the
   * same order as `filter_column_chunks_byte_ranges`
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the device memory for the output table chunks
   */
  void setup_chunking_for_filter_columns(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::column_view const& row_mask,
    use_data_page_mask mask_data_pages,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Materializes a chunk of filter columns and updates the corresponding range of input row
   * mask to only the rows that exist in the output table
   *
   * @param[in,out] row_mask Mutable boolean column spanning all selected rows across all sources
   * and indicating surviving rows from page pruning. The row mask size must equal the total
   * number of rows in the input row groups, and is empty only when there are no such rows
   * (yielding an empty output table)
   *
   * @return Table chunk of materialized filter columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_filter_columns_chunk(
    cudf::mutable_column_view& row_mask) const;

  /**
   * @brief Setup chunking information for payload columns and preprocess the input data pages
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per table chunk. `0` if
   * there is no limit
   * @param pass_read_limit Limit on the memory used for reading and decompressing data. `0` if
   * there is no limit
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param row_mask Boolean column spanning all selected rows across all sources and indicating
   * which rows need to be read
   * @param mask_data_pages Whether to build and use a data page mask using the row mask
   * @param column_chunk_data Flattened device spans of payload column chunk data returned in the
   * same order as `payload_column_chunks_byte_ranges`
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the device memory for the output table chunks
   */
  void setup_chunking_for_payload_columns(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::column_view const& row_mask,
    use_data_page_mask mask_data_pages,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Materializes a chunk of payload columns and applies the corresponding range of input row
   * mask to the output table chunk
   *
   * @param row_mask Boolean column spanning all selected rows across all sources and indicating
   * which rows need to be read
   *
   * @return Table chunk of materialized payload columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_payload_columns_chunk(
    cudf::column_view const& row_mask) const;

  /**
   * @brief Setup chunking information for all (or selected) columns and preprocess the input data
   * pages
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per table chunk. `0` if
   * there is no limit
   * @param pass_read_limit Limit on the memory used for reading and decompressing data. `0` if
   * there is no limit
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param column_chunk_data Flattened device spans of column chunk data returned in the same order
   * as `all_column_chunks_byte_ranges`
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the device memory for the output table chunks
   */
  void setup_chunking_for_all_columns(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Materializes a chunk of all (or selected) columns and returns the output table chunk
   *
   * @return Table chunk of materialized all (or selected) columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_all_columns_chunk() const;

  /**
   * @brief Partition row groups into passes such that the amount of GPU memory required to read,
   * decompress and decode a pass is bounded by the specified limit
   *
   * Note that the `pass_read_limit` is a hint, not an absolute limit - if a single row group
   * cannot fit within the limit given, it will still constitute a pass. The compressed row group
   * size is estimated over all columns in each row group (not just the columns selected for
   * reading), for conservative estimates.
   *
   * @throws std::invalid_argument if no row group indices in the input
   *
   * @param row_group_indices Span of vectors of input row group indices, one per source
   * @param pass_read_limit Memory limit to read and decompress row group data, `0` if there is
   * no limit (single pass)
   *
   * @return Vector of per-source row group indices, one per constructed pass
   */
  [[nodiscard]] std::vector<std::vector<std::vector<size_type>>> construct_row_group_passes(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    std::size_t pass_read_limit) const;

  /**
   * @brief Check if there is any parquet data left to read for the current chunked setup
   *
   * @return Boolean indicating if there is any data left to read
   */
  [[nodiscard]] bool has_next_table_chunk() const;

 private:
  std::unique_ptr<detail::hybrid_scan_reader_impl> _impl;
};

/** @} */  // end of group

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
