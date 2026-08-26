/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file reader_impl.hpp
 * @brief cuDF-IO Parquet reader class implementation header
 */

#pragma once

#include "expression_transform_helpers.hpp"
#include "parquet_gpu.hpp"
#include "reader_impl_chunking.hpp"
#include "reader_impl_helpers.hpp"

#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>
#include <optional>
#include <vector>

namespace cudf::io::parquet::detail {

/**
 * @brief Implementation for Parquet reader
 */
class reader_impl {
 public:
  /**
   * @brief Default constructor for subclassing
   */
  reader_impl();

  /**
   * @brief Constructor from an array of dataset sources with reader options
   *
   * By using this constructor, each call to `read()` or `read_chunk()` will perform reading the
   * entire given file.
   *
   * @param sources Dataset sources
   * @param parquet_metadatas Pre-materialized Parquet file metadata(s). Read from sources if empty
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader_impl(std::vector<std::unique_ptr<datasource>>&& sources,
                       std::vector<FileMetaData>&& parquet_metadatas,
                       parquet_reader_options const& options,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr);

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @return The set of columns along with metadata
   */
  table_with_metadata read();

  /**
   * @brief Constructor from a chunk read limit and an array of dataset sources with reader options.
   *
   * By using this constructor, the reader will support iterative (chunked) reading through
   * `has_next() ` and `read_chunk()`. For example:
   * ```
   *  do {
   *    auto const chunk = reader.read_chunk();
   *    // Process chunk
   *  } while (reader.has_next());
   *
   * // Alternatively
   *
   *  while (reader.has_next()) {
   *    auto const chunk = reader.read_chunk();
   *    // Process chunk
   *  }
   *
   * ```
   *
   * Reading the whole given file at once through `read()` function is still supported if
   * `chunk_read_limit == 0` (i.e., no reading limit) and `pass_read_limit == 0` (no temporary
   * memory limit) In such case, `read_chunk()` will also return rows of the entire file.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   *        or `0` if there is no limit
   * @param pass_read_limit Limit on memory usage for the purposes of decompression and processing
   * of input, or `0` if there is no limit.
   * @param sources Dataset sources
   * @param parquet_metadatas Pre-materialized Parquet file metadata(s). Read from sources if empty
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader_impl(std::size_t chunk_read_limit,
                       std::size_t pass_read_limit,
                       std::vector<std::unique_ptr<datasource>>&& sources,
                       std::vector<FileMetaData>&& parquet_metadatas,
                       parquet_reader_options const& options,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr);

  reader_impl(reader_impl const&)            = delete;
  reader_impl& operator=(reader_impl const&) = delete;
  reader_impl(reader_impl&&)                 = delete;
  reader_impl& operator=(reader_impl&&)      = delete;

  /**
   * @copydoc cudf::io::chunked_parquet_reader::has_next
   */
  bool has_next();

  /**
   * @copydoc cudf::io::chunked_parquet_reader::read_chunk
   */
  table_with_metadata read_chunk();

  // top level functions involved with ratcheting through the passes, subpasses
  // and output chunks of the read process
 protected:
  /**
   * @brief The enum indicating whether the data sources are read all at once or chunk by chunk.
   */
  enum class read_mode { READ_ALL, CHUNKED_READ };

  /**
   * @brief Perform the necessary data preprocessing for parsing file later on.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void prepare_data(read_mode mode);

  /**
   * @brief Preprocess step for the entire file.
   *
   * Only ever called once. This function reads in rowgroup and associated chunk
   * information and computes the schedule of top level passes (see `pass_intermediate_data`).
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void preprocess_file(read_mode mode);

  /**
   * @brief Ratchet the pass/subpass/chunk process forward.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void handle_chunking(read_mode mode);

  /**
   * @brief Setup step for the next input read pass.
   *
   * A 'pass' is defined as a subset of row groups read out of the globally
   * requested set of all row groups.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void setup_next_pass(read_mode mode);

  /**
   * @brief Setup step for the next decompression subpass.
   *
   * A 'subpass' is defined as a subset of pages within a pass that are
   * decompressed and decoded as a batch. Subpasses may be further subdivided
   * into output chunks.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   *
   */
  void setup_next_subpass(read_mode mode);

  /**
   * @brief Preprocess string length and bounds information for the subpass.
   *
   * At the end of this process, the `str_bytes` field of the the PageInfo struct
   * will be populated, and if applicable, the delta_temp_buf in the subpass struct will
   * be allocated and the pages in the subpass will point into it properly.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param read_info The range of rows to be read in the subpass
   */
  void preprocess_chunk_strings(read_mode mode, row_range const& read_info);

  /**
   * @brief Detect per-column eligibility for direct Parquet-dict → DICTIONARY32 transcode, and
   * apply the required host-side mutations to `_output_buffers` and `subpass.pages`.
   *
   * Must be called after `prepare_data()`. Populates `_dict_transcode_eligible` with a bool per
   * input column indicating whether the column will be assembled as a DICTIONARY32 output later in
   * `assemble_dict_transcoded_columns`. That member is the sole signal of whether the fast path is
   * active: `assemble_dict_transcoded_columns` no-ops when no column is eligible.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void prepare_dict_transcode(read_mode mode);

  /**
   * @brief Assemble DICTIONARY32 output columns for input columns that were marked eligible by
   * `prepare_dict_transcode`.
   *
   * @param out_columns The output columns vector to transcode in place.
   */
  void assemble_dict_transcoded_columns(std::vector<std::unique_ptr<column>>& out_columns);

  /**
   * @brief Copies over the relevant page mask information for the subpass
   */
  void set_subpass_page_mask();

  /**
   * @brief Read a chunk of data and return an output table.
   *
   * This function is called internally and expects all preprocessing steps have already been done.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @return The output table along with columns' metadata
   */
  table_with_metadata read_chunk_internal(read_mode mode);

  // utility functions
 protected:
  /**
   * @brief Read the set of column chunks to be processed for this pass.
   *
   * Does not decompress the chunk data.
   *
   * @return pair of boolean indicating if compressed chunks were found and a future for
   * read completion
   */
  std::pair<bool, std::future<void>> read_column_chunks();

  /**
   * @brief Build the `column_selection_options` bundle for `select_columns()`.
   *
   * @param options Reader options
   * @return Column selection options
   */
  [[nodiscard]] column_selection_options make_column_selection_options(
    parquet_reader_options const& options) const;

  /**
   * @brief Read compressed data and page information for the current pass.
   */
  void read_compressed_data();

  /**
   * @brief Build string dictionary indices for a pass.
   *
   */
  void build_string_dict_indices();

  /**
   * @brief For list columns, generate estimated row counts for pages in the current pass.
   *
   * The row counts in the pages that come out of the file only reflect the number of values in
   * all of the rows in the page, not the number of rows themselves. In order to do subpass reading
   * more accurately, we would like to have a more accurate guess of the real number of rows per
   * page.
   */
  enum class is_estimate_row_counts : bool { NO = false, YES = true };
  void generate_list_column_row_counts(is_estimate_row_counts is_estimate_row_counts);

  /**
   * @brief Perform some preprocessing for subpass page data and also compute the split locations
   * {skip_rows, num_rows} for chunked reading.
   *
   * There are several pieces of information we can't compute directly from row counts in
   * the parquet headers when dealing with nested schemas:
   * - The total sizes of all output columns at all nesting levels
   * - The starting output buffer offset for each page, for each nesting level
   *
   * For flat schemas, these values are computed during header decoding (see decode_page_headers).
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   *        or `0` if there is no limit
   */
  void preprocess_subpass_pages(read_mode mode, size_t chunk_read_limit);

  /**
   * @brief Set page string offset indices for non-dictionary, non-FLBA string columns.
   *
   * This function calculates the string offset index for each page of non-dictionary, non-FLBA
   * string columns and populates the subpass.page_string_offset_indices member variable.
   * The indices are used by decode kernels to access pre-computed string offsets.
   *
   * @param skip_rows The number of rows to skip in this subpass
   * @param num_rows The number of rows to read in this subpass
   */
  void compute_page_string_offset_indices(size_t skip_rows, size_t num_rows);

  /**
   * @brief Allocate nesting information storage for all pages and set pointers to it.
   *
   * One large contiguous buffer of PageNestingInfo structs is allocated and
   * distributed among the PageInfo structs.
   *
   * Note that this gets called even in the flat schema case so that we have a
   * consistent place to store common information such as value counts, etc.
   */
  void allocate_nesting_info();

  /**
   * @brief Allocate space for use when decoding definition/repetition levels.
   *
   * One large contiguous buffer of data allocated and
   * distributed among the PageInfo structs.
   */
  void allocate_level_decode_space();

  /**
   * @brief Populate the output table metadata from the parquet file metadata.
   *
   * @param out_metadata The output table metadata to add to
   */
  void populate_metadata(table_metadata& out_metadata);

  /**
   * @brief Finalize the output table by adding empty columns for the non-selected columns in
   * schema.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param out_metadata The output table metadata
   * @param out_columns The columns for building the output table
   * @return The output table along with columns' metadata
   */
  table_with_metadata finalize_output(read_mode mode,
                                      table_metadata& out_metadata,
                                      std::vector<std::unique_ptr<column>>& out_columns);

  /**
   * @brief Allocate data buffers for the output columns.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param skip_rows Crop all rows below skip_rows
   * @param num_rows Number of rows to read
   */
  void allocate_columns(read_mode mode, size_t skip_rows, size_t num_rows);

  /**
   * @brief Calculate per-page offsets for string data
   *
   * @return Vector of total string data sizes for each column
   */
  cudf::detail::host_vector<size_t> calculate_page_string_offsets();

  /**
   * @brief Converts the page data and outputs to columns.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to decode
   */
  void decode_page_data(read_mode mode, size_t skip_rows, size_t num_rows);

  /**
   * @brief Invalidate output buffer nullmask for rows spanned by the pruned pages
   *
   * @param page_mask Boolean vector indicating if a page needs to be decoded or is pruned
   * @param skip_rows Offset of the first row in the table chunk
   * @param num_rows Number of rows in the table chunk
   */
  void update_output_nullmasks_for_pruned_pages(cudf::host_span<bool const> page_mask,
                                                size_t skip_rows,
                                                size_t num_rows);

  /**
   * @brief Fill in string and list offsets for rows covered by pruned data pages.
   *
   * @param skip_rows Offset of the first row in the table chunk
   * @param num_rows Number of rows in the table chunk
   * @param initial_str_offsets Initial offsets used to construct large nested strings
   */
  void fill_pruned_offsets(size_t skip_rows,
                           size_t num_rows,
                           cudf::device_span<size_t> initial_str_offsets);

  /**
   * @brief Creates file-wide parquet chunk information.
   *
   * Creates information about all chunks in the file, storing it in
   * the file-wide _file_itm_data structure.
   */
  void create_global_chunk_info();

  /**
   * @brief Computes all of the passes we will perform over the file
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void compute_input_passes(read_mode mode);

  /**
   * @brief Given a set of pages that have had their sizes computed by nesting level and
   * a limit on total read size, generate a set of {skip_rows, num_rows} pairs representing
   * a set of reads that will generate output columns of total size <= `chunk_read_limit` bytes.
   */
  void compute_output_chunks_for_subpass();

  /**
   * @brief Check if there is more work to be done
   */
  [[nodiscard]] bool has_more_work() const
  {
    return _file_itm_data.num_passes() > 0 &&
           _file_itm_data._current_input_pass < _file_itm_data.num_passes();
  }

  /**
   * @brief Check if the user has specified columns from mismatched sources
   *
   * @param options Reader options
   * @return True if the user has specified columns from mismatched sources
   */
  [[nodiscard]] bool has_cols_from_mismatched_sources(parquet_reader_options const& options) const
  {
    return (options.get_column_names().has_value() or
            options.get_column_field_ids().has_value()) and
           options.is_enabled_allow_mismatched_pq_schemas();
  }

  /**
   * @brief Effective `ignore_missing_columns` policy for column selection
   *
   * This flag would be disabled when multiple sources use mismatched-schema column selection.
   *
   * @param options Reader options
   * @return Effective `ignore_missing_columns` value
   */
  [[nodiscard]] bool ignore_missing_columns_policy(parquet_reader_options const& options) const
  {
    return options.is_enabled_ignore_missing_columns() and
           not(has_cols_from_mismatched_sources(options) and _metadata->get_num_sources() > 1);
  }

 protected:
  /**
   * @brief Check if the user has specified custom row bounds
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   * @return True if the user has specified custom row bounds
   */
  [[nodiscard]] bool uses_custom_row_bounds(read_mode mode) const
  {
    // TODO: `read_mode` is hardcoded to `true` when `read_mode::CHUNKED_READ` to enforce
    // `compute_page_sizes()` computation for all remaining chunks.
    return (mode == read_mode::READ_ALL)
             ? (_options.num_rows.has_value() or _options.skip_rows != 0)
             : true;
  }

  /**
   * @brief Check if this is the first output chunk
   *
   * @return True if this is the first output chunk
   */
  [[nodiscard]] bool is_first_output_chunk() const
  {
    return _file_itm_data._output_chunk_count == 0;
  }

  [[nodiscard]] cudf::detail::hostdevice_span<bool> subpass_page_mask_span() const
  {
    return _subpass_page_mask ? *_subpass_page_mask : cudf::detail::hostdevice_span<bool>{};
  }

  /**
   * @brief Offset the column references in `_expr_conv` by the number of columns prepended to
   * the output
   *
   * @return Offsetted expression converter
   */
  [[nodiscard]] inline offset_column_references compute_offset_filter() const
  {
    auto const num_prepended_cols = static_cast<size_type>(_options.prepend_source_index_column) +
                                    static_cast<size_type>(_options.prepend_row_index_column);
    return offset_column_references(_expr_conv.get_converted_expr(), num_prepended_cols);
  }

  /**
   * @brief Calculate the number of rows read from each source in the output chunk
   *
   * @param chunk_start_row The offset of the first row in the output chunk
   * @param chunk_num_rows The number of rows in the the output chunk
   * @return Vector of number of rows from each respective data source in the output chunk
   */
  [[nodiscard]] std::vector<size_t> calculate_output_num_rows_per_source(size_t chunk_start_row,
                                                                         size_t chunk_num_rows);

  /**
   * @brief Computes the names of columns to be read from the file, if specified.
   *
   * @param options The reader options
   * @return Names of columns to be read from the file if specified, `nullopt` otherwise
   */
  [[nodiscard]] std::optional<std::vector<std::string>> get_column_projection(
    parquet_reader_options const& options) const;

  /**
   * @brief Cast any fixed-point output columns to the decimal width specified in options.
   *
   * @param out_columns Output columns to cast
   */
  void apply_decimal_width_cast(std::vector<std::unique_ptr<cudf::column>>& out_columns);

  cuda::stream_ref _stream;
  rmm::device_async_resource_ref _mr{cudf::get_current_device_resource_ref()};

  // Reader configs.
  struct {
    // timestamp_type
    data_type timestamp_type;
    // decimal_width
    type_id decimal_width;
    // Use specified row selection
    int64_t const skip_rows;
    std::optional<int64_t> num_rows;
    // Use specified bytes selection
    size_t skip_bytes;
    std::optional<size_t> num_bytes;
    // User specified row group selection
    std::vector<std::vector<size_type>> row_group_indices;
    // Whether to use JIT for filtering
    bool use_jit_filter = false;
    // Whether to use case-sensitive matching for column names
    bool case_sensitive_names = true;
    // Whether to prepend the source file index column to the output
    bool prepend_source_index_column = false;
    // Whether to prepend the file-local row index column to the output
    bool prepend_row_index_column = false;
    // Whether to try outputting DICTIONARY32 columns for fully dict-encoded string columns
    bool output_dict_columns = false;
  } _options;

  // Converts the input filter to AST output filter.
  parquet_filter_normalizer _expr_conv{std::nullopt, table_metadata{}, true};

  std::vector<std::unique_ptr<datasource>> _sources;
  std::shared_ptr<aggregate_reader_metadata> _metadata;

  // Number of sources
  size_t _num_sources{0};

  // input columns to be processed
  std::vector<input_column_info> _input_columns;

  // Buffers for generating output columns
  std::vector<cudf::io::detail::inline_column_buffer> _output_buffers;

  // Buffers copied from `_output_buffers` after construction for reuse
  std::vector<cudf::io::detail::inline_column_buffer> _output_buffers_template;

  // _output_buffers associated schema indices
  std::vector<int> _output_column_schemas;

  // Page mask for filtering out pass data pages (Not copied to the device)
  thrust::host_vector<bool> _pass_page_mask;

  // Page mask for filtering out subpass data pages (Copied to the device)
  std::unique_ptr<cudf::detail::hostdevice_vector<bool>> _subpass_page_mask;

  // _output_buffers associated metadata
  std::unique_ptr<table_metadata> _output_metadata;

  // number of extra filter columns
  std::size_t _num_filter_only_columns{0};

  bool _strings_to_categorical = false;

  // are offset indexes available for selected row groups
  bool _has_offset_index = false;

  // whether sparse page I/O is enabled
  bool _sparse_page_io = false;

  std::optional<std::vector<reader_column_schema>> _reader_column_schema;

  // chunked reading happens in 2 parts:
  //
  // At the top level, the entire file is divided up into "passes" on which we try and limit the
  // total amount of temporary memory (compressed data, decompressed data) in use
  // via _input_pass_read_limit.
  //
  // Within a pass, we produce one or more chunks of output, whose maximum total
  // byte size is controlled by _output_chunk_read_limit.

  file_intermediate_data _file_itm_data;
  bool _file_preprocessed{false};

  std::unique_ptr<pass_intermediate_data> _pass_itm_data;

  std::size_t _output_chunk_read_limit{0};  // output chunk size limit in bytes
  std::size_t _input_pass_read_limit{0};    // input pass memory usage limit in bytes

  // Per-input-column flag indicating whether that column was selected for direct
  // Parquet-dict → DICTIONARY32 transcode.
  std::vector<bool> _dict_transcode_eligible;
};

}  // namespace cudf::io::parquet::detail
