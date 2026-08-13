/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/parquet/expression_transform_helpers.hpp"
#include "io/parquet/parquet_gpu.hpp"
#include "io/parquet/reader_impl_helpers.hpp"

#include <cudf/io/detail/utils.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

using aggregate_reader_metadata_base = parquet::detail::aggregate_reader_metadata;
using metadata_base                  = parquet::detail::metadata;

using io::detail::inline_column_buffer;
using parquet::detail::equality_literals_collector;
using parquet::detail::input_column_info;
using parquet::detail::row_group_info;

/**
 * @brief Class for parsing dataset metadata
 */
struct metadata : public metadata_base {
  explicit metadata(cudf::host_span<uint8_t const> footer_bytes);
  explicit metadata(FileMetaData const& other) { static_cast<FileMetaData&>(*this) = other; }
  metadata(metadata const& other)            = delete;
  metadata(metadata&& other)                 = default;
  metadata& operator=(metadata const& other) = delete;
  metadata& operator=(metadata&& other)      = default;

  ~metadata() = default;
};

class aggregate_reader_metadata : public aggregate_reader_metadata_base {
 private:
  /**
   * @brief Check whether selected columns have column and offset indexes
   *
   * Schema indices are mapped to each source before locating the column chunks.
   *
   * @param row_group_indices Row group indices, one vector per source
   * @param schema_indices Schema indices from the first source
   * @return A pair indicating column-index and offset-index presence, respectively
   */
  [[nodiscard]] std::pair<bool, bool> page_index_presence(
    std::span<std::vector<size_type> const> row_group_indices,
    std::span<size_type const> schema_indices) const;

  /**
   * @brief Filters the row groups using dictionary pages
   *
   * @param chunks Host device span of column chunk descriptors, one per column chunk with
   *               dictionary page and (in)equality predicate
   * @param pages Host device span of decoded page headers, one per column chunk with dictionary
   *              page and (in)equality predicate
   * @param input_row_group_indices Lists of input row groups, one per source
   * @param literals Lists of literals, one per column with (in)equality predicate
   * @param operators Lists of operators, one per column with (in)equality predicate
   * @param total_row_groups Total number of row groups in `input_row_group_indices`
   * @param output_dtypes Output data types for the filtered columns
   * @param dictionary_col_schemas Schema indices of columns with (in)equality predicate
   * @param filter AST expression to filter row groups based on dictionary pages
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return A pair of filtered row group indices if any is filtered.
   */
  [[nodiscard]] std::optional<std::vector<std::vector<cudf::size_type>>> apply_dictionary_filter(
    cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks,
    cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages,
    std::span<std::vector<size_type> const> input_row_group_indices,
    std::span<std::vector<ast::literal*> const> literals,
    std::span<std::vector<ast::ast_operator> const> operators,
    size_t total_row_groups,
    std::span<data_type const> output_dtypes,
    std::span<int const> dictionary_col_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;

 public:
  /**
   * @brief Constructor for aggregate_reader_metadata
   *
   * @param footer_bytes Host span of Parquet file footer buffer bytes, one per source
   * @param use_arrow_schema Whether to use Arrow schema
   * @param has_cols_from_mismatched_srcs Whether to have columns from mismatched sources
   */
  aggregate_reader_metadata(cudf::host_span<cudf::host_span<uint8_t const> const> footer_bytes,
                            bool use_arrow_schema,
                            bool has_cols_from_mismatched_srcs);

  /**
   * @brief Constructor for aggregate_reader_metadata
   *
   * @param parquet_metadatas Host span of pre-populated Parquet file metadata, one per source
   * @param use_arrow_schema Whether to use Arrow schema
   * @param has_cols_from_mismatched_srcs Whether to have columns from mismatched sources
   */
  aggregate_reader_metadata(cudf::host_span<FileMetaData const> parquet_metadatas,
                            bool use_arrow_schema,
                            bool has_cols_from_mismatched_srcs);

  aggregate_reader_metadata(aggregate_reader_metadata const&)            = delete;
  aggregate_reader_metadata& operator=(aggregate_reader_metadata const&) = delete;
  aggregate_reader_metadata(aggregate_reader_metadata&&)                 = default;
  aggregate_reader_metadata& operator=(aggregate_reader_metadata&&)      = default;

  /**
   * @brief Initialize the internal variables
   */
  void initialize_internals(bool use_arrow_schema, bool has_cols_from_mismatched_srcs);

  /**
   * @brief Fetch the byte range of the page index in each Parquet file
   *
   * @return Vector of byte ranges of the page index, one per source
   */
  [[nodiscard]] std::vector<text::byte_range_info> page_index_byte_ranges() const;

  /**
   * @brief Get the Parquet file metadata for every source
   *
   * @return Vector of file metadata, one per source
   */
  [[nodiscard]] std::vector<FileMetaData> parquet_metadatas() const;

  /**
   * @brief Setup and populate the page index structs in every source's `FileMetaData`
   *
   * @param page_index_bytes Host span of Parquet page index buffer bytes, one per source
   */
  void setup_page_indexes(cudf::host_span<cudf::host_span<uint8_t const> const> page_index_bytes);

  /**
   * @brief Get all available row group indices, one inner vector per source
   *
   * If `options.get_row_groups()` is non-empty, validates that its size equals the number of
   * sources and returns it as-is. Otherwise returns `[0 .. per_source_num_row_groups[i])` for
   * each source.
   *
   * @param options Parquet reader options
   * @return Vector of row group indices, one inner vector per source
   */
  [[nodiscard]] std::vector<std::vector<size_type>> all_row_groups(
    parquet_reader_options const& options) const;

  /**
   * @brief Get the total number of top-level rows in the row groups
   *
   * @param row_group_indices Input row groups indices
   * @return Total number of top-level rows in the row groups
   */
  [[nodiscard]] std::size_t total_rows_in_row_groups(
    std::span<std::vector<size_type> const> row_group_indices) const;

  /**
   * @brief Filters and reduces down to the selection of payload columns
   *
   * @param payload_column_names List of paths of select payload column names, if any
   * @param filter_column_names List of paths of column names present only in filter, if any
   * @param selection_options Bundled column selection options
   *
   * @return input column information, output column buffers, list of output column schema
   * indices
   */
  [[nodiscard]] std::
    tuple<std::vector<input_column_info>, std::vector<inline_column_buffer>, std::vector<size_type>>
    select_payload_columns(std::optional<std::vector<std::string>> const& payload_column_names,
                           std::optional<std::vector<std::string>> const& filter_column_names,
                           parquet::detail::column_selection_options const& selection_options);

  /**
   * @brief Filters row groups such that only the row groups that start within the byte range
   * specified by [`bytes_to_skip`, `bytes_to_skip + bytes_to_read`) are selected
   *
   * @note The last selected row group may end beyond the byte range.
   *
   * @param row_group_indices Input row groups indices
   * @param bytes_to_skip Bytes to skip before selecting row groups
   * @param bytes_to_read Optional bytes to select row groups from after skipping. All row groups
   * until the end of the file are selected if not provided
   *
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<std::vector<cudf::size_type>> filter_row_groups_with_byte_range(
    std::span<std::vector<size_type> const> row_group_indices,
    std::size_t bytes_to_skip,
    std::optional<std::size_t> const& bytes_to_read) const;

  /**
   * @brief Filter the row groups with statistics based on predicate filter
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on Column chunk statistics
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Filtered row group indices, if any are filtered
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_stats(
    std::span<std::vector<size_type> const> row_group_indices,
    std::span<data_type const> output_dtypes,
    std::span<cudf::size_type const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Get the bloom filter byte ranges, one per column chunk with equality predicate
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on bloom filters
   *
   * @return Byte ranges of bloom filters, one per column chunk with equality predicate
   */
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> get_bloom_filter_bytes(
    std::span<std::vector<size_type> const> row_group_indices,
    std::span<data_type const> output_dtypes,
    std::span<cudf::size_type const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter);

  /**
   * @brief Get the dictionary page byte ranges, one per column chunk with (in)equality predicate
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on dictionary pages
   *
   * @return A pair of vectors containing dictionary page byte ranges and corresponding source
   *         indices
   */
  [[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                          std::vector<cudf::size_type>>
  dictionary_pages_byte_ranges(std::span<std::vector<size_type> const> row_group_indices,
                               std::span<data_type const> output_dtypes,
                               std::span<cudf::size_type const> output_column_schemas,
                               std::reference_wrapper<ast::expression const> filter);

  /**
   * @brief Filter the row groups using dictionaries based on predicate filter
   *
   * @param chunks Host device span of column chunk descriptors, one per column chunk with
   *               dictionary page and (in)equality predicate
   * @param pages Host device span of decoded page headers, one per column chunk with dictionary
   *              page and (in)equality predicate
   * @param row_group_indices Input row groups indices
   * @param literals Lists of literals, one per input column
   * @param operators Lists of operators, one per input column
   * @param output_dtypes Datatypes of output columns
   * @param dictionary_col_schemas Schema indices of output columns with (in)equality predicate
   * @param filter AST expression to filter row groups based on dictionary pages
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Filtered row group indices, if any are filtered
   */
  [[nodiscard]] std::vector<std::vector<cudf::size_type>> filter_row_groups_with_dictionary_pages(
    cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks,
    cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages,
    std::span<std::vector<cudf::size_type> const> row_group_indices,
    std::span<std::vector<ast::literal*> const> literals,
    std::span<std::vector<ast::ast_operator> const> operators,
    std::span<data_type const> output_dtypes,
    std::span<cudf::size_type const> dictionary_col_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filter the row groups using bloom filters based on predicate filter
   *
   * @param bloom_filter_data Device spans of bloom filters, one per input column chunk
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on bloom filters
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Filtered row group indices, if any are filtered
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_bloom_filters(
    std::span<cudf::device_span<uint8_t const> const> bloom_filter_data,
    std::span<std::vector<size_type> const> row_group_indices,
    std::span<data_type const> output_dtypes,
    std::span<cudf::size_type const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Builds a row mask with all rows set to true
   *
   * @param row_group_indices Input row groups indices
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   *
   * @return A boolean column representing a mask of rows with all rows set to true
   */
  [[nodiscard]] std::unique_ptr<cudf::column> build_all_true_row_mask(
    std::span<std::vector<size_type> const> row_group_indices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Builds a row mask based on the data pages that survive page-level statistics based on
   * predicate filter
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter data pages based on page index statistics
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   *
   * @return A boolean column representing a mask of rows surviving the predicate filter at
   *         page-level
   */
  [[nodiscard]] std::unique_ptr<cudf::column> build_row_mask_with_page_index_stats(
    std::span<std::vector<size_type> const> row_group_indices,
    std::span<cudf::data_type const> output_dtypes,
    std::span<cudf::size_type const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Computes which data pages need decoding to construct input columns based on the row mask
   *
   * Compute a vector of boolean vectors indicating which data pages need to be decoded to
   * construct each input column based on the row mask, one vector per column
   *
   * @tparam ColumnView Type of the row mask column view - cudf::mutable_column_view for filter
   * columns and cudf::column_view for payload columns
   *
   * @param row_mask Boolean column indicating which rows need to be read after page-pruning
   * @param row_group_indices Input row groups indices
   * @param input_columns Input column information
   * @param row_mask_offset Offset into the row mask column for the current pass
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Boolean vector indicating which data pages need to be decoded to produce
   *         the output table based on the input row mask across all input columns
   */
  template <typename ColumnView>
  [[nodiscard]] thrust::host_vector<bool> compute_data_page_mask(
    ColumnView const& row_mask,
    std::span<std::vector<size_type> const> row_group_indices,
    std::span<input_column_info const> input_columns,
    cudf::size_type row_mask_offset,
    rmm::cuda_stream_view stream) const;
};

/**
 * @brief Collects lists of equal and not-equal predicate literals and operators in the AST
 * expression, one per input table column. This is used in row group filtering based on dictionary
 * pages
 */
class dictionary_literals_collector : public equality_literals_collector {
 public:
  dictionary_literals_collector() = default;

  dictionary_literals_collector(ast::expression const& expr,
                                std::span<cudf::data_type const> output_dtypes);

  // Bring all overloads of `visit` from equality_literals_collector into scope
  using equality_literals_collector::visit;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Returns vectors of collected literals and (in)equality operators in the AST expression,
   * one per input table column
   *
   * @return A pair of vectors of collected literals and (in)equality operators, one per input table
   * column
   */
  [[nodiscard]] std::pair<std::vector<std::vector<ast::literal*>>,
                          std::vector<std::vector<ast::ast_operator>>>
  get_literals_and_operators() &&;

 private:
  std::vector<std::vector<ast::ast_operator>> _operators;
};

/**
 * @brief Converts named columns to index reference columns
 */
class named_to_reference_converter : public parquet::detail::named_to_reference_converter {
 public:
  named_to_reference_converter() = default;

  named_to_reference_converter(std::optional<std::reference_wrapper<ast::expression const>> expr,
                               table_metadata const& metadata,
                               std::vector<SchemaElement> const& schema_tree,
                               cudf::io::parquet_reader_options const& options,
                               bool case_sensitive_names);

  using parquet::detail::named_to_reference_converter::visit;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

 private:
  std::unordered_map<int32_t, std::string> _column_indices_to_names;
};

}  // namespace cudf::io::parquet::experimental::detail
