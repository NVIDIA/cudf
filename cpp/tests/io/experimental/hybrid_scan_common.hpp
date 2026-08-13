/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

/**
 * @brief Helper to construct a random list<str> column
 *
 * @param gen Random engine
 * @param is_str_nullable Whether the string column should be nullable
 * @param is_list_nullable Whether the list column should be nullable
 *
 * @return Unique pointer to the constructed list<str> column
 */
[[nodiscard]] std::unique_ptr<cudf::column> make_list_str_column(std::mt19937& gen,
                                                                 bool is_str_nullable,
                                                                 bool is_list_nullable);

/**
 * @brief Struct to hold multifile datasources and footer buffers along with their byte spans
 */
struct multifile_inputs {
  /**
   * @brief Construct datasources, datasource refs, and footer byte spans from source info
   */
  explicit multifile_inputs(cudf::io::source_info const& source_info);

  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  std::vector<std::reference_wrapper<cudf::io::datasource>> datasource_refs;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footer_buffers;
  std::vector<cudf::host_span<uint8_t const>> footer_byte_spans;
};

/**
 * @brief Device buffers and spans from multiple input sources
 */
struct multisource_device_data {
  std::vector<rmm::device_buffer> buffers;
  std::vector<std::vector<cudf::device_span<uint8_t const>>> per_source_spans;
  std::vector<cudf::device_span<uint8_t const>> flat_spans;
};

/**
 * @brief Construct source info from host buffers
 */
[[nodiscard]] cudf::io::source_info build_source_info(
  std::vector<std::vector<char>> const& file_buffers);

/**
 * @brief Fetch and set up page indexes for all sources in a multifile reader
 */
void setup_page_indexes(cudf::io::parquet::experimental::hybrid_scan_multifile const& reader,
                        multifile_inputs const& inputs);

/**
 * @brief Groups a flat byte range list by source using the specified source map
 */
[[nodiscard]] std::vector<std::vector<cudf::io::text::byte_range_info>> group_byte_ranges_by_source(
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  std::size_t num_sources);

/**
 * @brief Fetches byte ranges from multiple sources and returns per-source and flattened spans
 */
[[nodiscard]] multisource_device_data fetch_multisource_device_data(
  multifile_inputs const& inputs,
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Concatenate a vector of tables and return the resultant table
 *
 * @param tables Vector of tables to concatenate
 * @param stream CUDA stream to use
 * @param mr Device memory resource used to allocate the returned table's device memory
 *
 * @return Unique pointer to the resultant concatenated table
 */
[[nodiscard]] std::unique_ptr<cudf::table> concatenate_tables(
  std::vector<std::unique_ptr<cudf::table>>&& tables,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Creates a table and writes it to Parquet host buffer with column level statistics
 *
 * This function creates a table with three columns:
 * - col0: ascending T values
 * - col1: descending T values (reduced cardinality for timestamps and durations)
 * - col2: constant cudf::string_view values
 *
 * The function creates a table by concatenating the same set of columns NumTableConcats times.
 * It then writes this table to a Parquet host buffer with column level statistics.
 *
 * @tparam T Data type for columns 0 and 1
 * @tparam NumTableConcats Number of times to concatenate the base table (must be >= 1)
 * @tparam IsConstantStrings Whether to use constant strings for column 2
 * @tparam IsNullable Whether to create nullable columns
 *
 * @param str_col_value Value for the constant string column used when IsConstantStrings is true
 * @param compression Compression type
 * @param column_names Top-level column names assigned in `column_order` order (default
 *        {"col0", "col1", "col2"})
 * @param column_order Physical emit order of the base [col0, col1, col2] columns (default
 *        {0, 1, 2}). Reordering emits the same logical columns at different schema positions, which
 *        is used to build mismatched per-source schemas for the row-group filtering tests.
 * @param stream CUDA stream
 *
 * @note `column_order` must be a permutation of {0, 1, 2}; passing a non-permutation (e.g.
 *       {0, 0, 1}) is undefined behavior.
 *
 * @return Tuple of table and Parquet host buffer
 */
template <typename T,
          size_t NumTableConcats,
          bool IsConstantStrings = true,
          bool IsNullable        = false>
[[nodiscard]] std::pair<std::unique_ptr<cudf::table>, std::vector<char>> create_parquet_with_stats(
  cudf::size_type str_col_value             = 100,
  cudf::io::compression_type compression    = cudf::io::compression_type::AUTO,
  std::vector<std::string> column_names     = {"col0", "col1", "col2"},
  std::vector<cudf::size_type> column_order = {0, 1, 2},
  rmm::cuda_stream_view stream              = cudf::get_default_stream());

/**
 * @brief Prune row groups using column chunk dictionaries via the single-file hybrid scan reader
 *
 * @return Dictionary-filtered row group indices
 */
[[nodiscard]] std::vector<cudf::size_type> filter_row_groups_with_dictionaries(
  cudf::io::datasource& datasource,
  cudf::io::parquet::experimental::hybrid_scan_reader const& reader,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Prune row groups using column chunk dictionaries via the multi-file hybrid scan reader
 *
 * @return Per-source dictionary-filtered row group indices
 */
[[nodiscard]] std::vector<std::vector<cudf::size_type>> filter_row_groups_with_dictionaries(
  multifile_inputs const& inputs,
  cudf::io::parquet::experimental::hybrid_scan_multifile const& reader,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
