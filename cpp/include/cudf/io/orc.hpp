/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/detail/orc.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @file
 * @brief APIs for reading and writing ORC files.
 */

namespace CUDF_EXPORT cudf {
namespace io {
/**
 * @addtogroup io_readers
 * @{
 */

constexpr size_t default_stripe_size_bytes   = 64 * 1024 * 1024;  ///< 64MB default orc stripe size
constexpr size_type default_stripe_size_rows = 1000000;  ///< 1M rows default orc stripe rows
constexpr size_type default_row_index_stride = 10000;    ///< 10K rows default orc row index stride

/**
 * @brief Check if the compression type is supported for reading ORC files.
 *
 * @note This is a runtime check. Some compression types may not be supported because of the current
 * system configuration.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported
 */
[[nodiscard]] bool is_supported_read_orc(compression_type compression);

/**
 * @brief Check if the compression type is supported for writing ORC files.
 *
 * @note This is a runtime check. Some compression types may not be supported because of the current
 * system configuration.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported
 */
[[nodiscard]] bool is_supported_write_orc(compression_type compression);

/**
 * @brief Builds settings to use for `read_orc()`.
 */
class orc_reader_options_builder;

/**
 * @brief Settings to use for `read_orc()`.
 */
class orc_reader_options {
  source_info _source;

  // Names of column to read; `nullopt` is all
  std::optional<std::vector<std::string>> _columns;

  // List of individual stripes to read (ignored if empty)
  std::vector<std::vector<size_type>> _stripes;
  // Rows to skip from the start
  int64_t _skip_rows = 0;
  // Rows to read; `nullopt` is all
  std::optional<int64_t> _num_rows;

  // Whether to use row index to speed-up reading
  bool _use_index = true;

  // Whether to use numpy-compatible dtypes
  bool _use_np_dtypes = true;
  // Cast timestamp columns to a specific type
  data_type _timestamp_type{type_id::EMPTY};

  // Columns that should be read as Decimal128
  std::vector<std::string> _decimal128_columns;

  // Ignore writer timezone in the stripe footer, read as UTC timezone
  bool _ignore_timezone_in_stripe_footer = false;

  friend orc_reader_options_builder;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read orc file
   */
  explicit orc_reader_options(source_info src) : _source{std::move(src)} {}

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  orc_reader_options() = default;

  /**
   * @brief Creates `orc_reader_options_builder` which will build `orc_reader_options`.
   *
   * @param src Source information to read orc file
   * @return Builder to build reader options
   */
  static orc_reader_options_builder builder(source_info src);

  /**
   * @brief Returns source info.
   *
   * @return Source info
   */
  [[nodiscard]] source_info const& get_source() const { return _source; }

  /**
   * @brief Returns names of the columns to read, if set.
   *
   * @return Names of the columns to read; `nullopt` if the option is not set
   */
  [[nodiscard]] auto const& get_columns() const { return _columns; }

  /**
   * @brief Returns vector of vectors, stripes to read for each input source
   *
   * @return Vector of vectors, stripes to read for each input source
   */
  [[nodiscard]] auto const& get_stripes() const { return _stripes; }

  /**
   * @brief Returns number of rows to skip from the start.
   *
   * @return Number of rows to skip from the start
   */
  [[nodiscard]] int64_t get_skip_rows() const { return _skip_rows; }

  /**
   * @brief Returns number of row to read.
   *
   * @return Number of rows to read; `nullopt` if the option hasn't been set (in which case the file
   * is read until the end)
   */
  [[nodiscard]] std::optional<int64_t> const& get_num_rows() const { return _num_rows; }

  /**
   * @brief Whether to use row index to speed-up reading.
   *
   * @return `true` if row index is used to speed-up reading
   */
  [[nodiscard]] bool is_enabled_use_index() const { return _use_index; }

  /**
   * @brief Whether to use numpy-compatible dtypes.
   *
   * @return `true` if numpy-compatible dtypes are used
   */
  [[nodiscard]] bool is_enabled_use_np_dtypes() const { return _use_np_dtypes; }

  /**
   * @brief Returns timestamp type to which timestamp column will be cast.
   *
   * @return Timestamp type to which timestamp column will be cast
   */
  [[nodiscard]] data_type get_timestamp_type() const { return _timestamp_type; }

  /**
   * @brief Returns fully qualified names of columns that should be read as 128-bit Decimal.
   *
   * @return Fully qualified names of columns that should be read as 128-bit Decimal
   */
  [[nodiscard]] std::vector<std::string> const& get_decimal128_columns() const
  {
    return _decimal128_columns;
  }

  /**
   * @brief Returns whether to ignore writer timezone in the stripe footer.
   *
   * @return `true` if the writer timezone in the stripe footer is ignored.
   */
  [[nodiscard]] bool get_ignore_timezone_in_stripe_footer() const
  {
    return _ignore_timezone_in_stripe_footer;
  }

  // Setters

  /**
   * @brief Sets source info.
   *
   * @param src The source info.
   */
  void set_source(source_info src) { _source = std::move(src); }

  /**
   * @brief Sets names of the column to read.
   *
   * @param col_names Vector of column names
   */
  void set_columns(std::vector<std::string> col_names) { _columns = std::move(col_names); }

  /**
   * @brief Sets list of stripes to read for each input source
   *
   * @param stripes Vector of vectors, mapping stripes to read to input sources
   *
   * @throw cudf::logic_error if a non-empty vector is passed, and `skip_rows` has been previously
   * set
   * @throw cudf::logic_error if a non-empty vector is passed, and `num_rows` has been previously
   * set
   */
  void set_stripes(std::vector<std::vector<size_type>> stripes)
  {
    CUDF_EXPECTS(stripes.empty() or (_skip_rows == 0), "Can't set stripes along with skip_rows");
    CUDF_EXPECTS(stripes.empty() or not _num_rows.has_value(),
                 "Can't set stripes along with num_rows");
    _stripes = std::move(stripes);
  }

  /**
   * @brief Sets number of rows to skip from the start.
   *
   * @param rows Number of rows
   *
   * @throw cudf::logic_error if a negative value is passed
   * @throw cudf::logic_error if stripes have been previously set
   */
  void set_skip_rows(int64_t rows)
  {
    CUDF_EXPECTS(rows >= 0, "skip_rows cannot be negative");
    CUDF_EXPECTS(rows == 0 or _stripes.empty(), "Can't set both skip_rows along with stripes");
    _skip_rows = rows;
  }

  /**
   * @brief Sets number of row to read.
   *
   * @param nrows Number of rows
   *
   * @throw cudf::logic_error if a negative value is passed
   * @throw cudf::logic_error if stripes have been previously set
   */
  void set_num_rows(int64_t nrows)
  {
    CUDF_EXPECTS(nrows >= 0, "num_rows cannot be negative");
    CUDF_EXPECTS(_stripes.empty(), "Can't set both num_rows and stripes");
    _num_rows = nrows;
  }

  /**
   * @brief Enable/Disable use of row index to speed-up reading.
   *
   * @param use Boolean value to enable/disable row index use
   */
  void enable_use_index(bool use) { _use_index = use; }

  /**
   * @brief Enable/Disable use of numpy-compatible dtypes
   *
   * @param use Boolean value to enable/disable
   */
  void enable_use_np_dtypes(bool use) { _use_np_dtypes = use; }

  /**
   * @brief Sets timestamp type to which timestamp column will be cast.
   *
   * @param type Type of timestamp
   */
  void set_timestamp_type(data_type type) { _timestamp_type = type; }

  /**
   * @brief Set columns that should be read as 128-bit Decimal
   *
   * @param val Vector of fully qualified column names
   */
  void set_decimal128_columns(std::vector<std::string> val)
  {
    _decimal128_columns = std::move(val);
  }

  /**
   * @brief Sets whether to ignore writer timezone in the stripe footer.
   *
   * @param val Boolean value to enable/disable ignoring writer timezone
   */
  void enable_ignore_timezone_in_stripe_footer(bool val)
  {
    _ignore_timezone_in_stripe_footer = val;
  }
};

/**
 * @brief Builds settings to use for `read_orc()`.
 */
class orc_reader_options_builder {
  orc_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit orc_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read orc file
   */
  explicit orc_reader_options_builder(source_info src) : options{std::move(src)} {}

  /**
   * @brief Sets names of the column to read.
   *
   * @param col_names Vector of column names
   * @return this for chaining
   */
  orc_reader_options_builder& columns(std::vector<std::string> col_names)
  {
    options.set_columns(std::move(col_names));
    return *this;
  }

  /**
   * @brief Sets list of individual stripes to read per source
   *
   * @param stripes Vector of vectors, mapping stripes to read to input sources
   * @return this for chaining
   */
  orc_reader_options_builder& stripes(std::vector<std::vector<size_type>> stripes)
  {
    options.set_stripes(std::move(stripes));
    return *this;
  }

  /**
   * @brief Sets number of rows to skip from the start.
   *
   * @param rows Number of rows
   * @return this for chaining
   */
  orc_reader_options_builder& skip_rows(int64_t rows)
  {
    options.set_skip_rows(rows);
    return *this;
  }

  /**
   * @brief Sets number of row to read.
   *
   * @param nrows Number of rows
   * @return this for chaining
   */
  orc_reader_options_builder& num_rows(int64_t nrows)
  {
    options.set_num_rows(nrows);
    return *this;
  }

  /**
   * @brief Enable/Disable use of row index to speed-up reading.
   *
   * @param use Boolean value to enable/disable row index use
   * @return this for chaining
   */
  orc_reader_options_builder& use_index(bool use)
  {
    options.enable_use_index(use);
    return *this;
  }

  /**
   * @brief Enable/Disable use of numpy-compatible dtypes.
   *
   * @param use Boolean value to enable/disable
   * @return this for chaining
   */
  orc_reader_options_builder& use_np_dtypes(bool use)
  {
    options.enable_use_np_dtypes(use);
    return *this;
  }

  /**
   * @brief Sets timestamp type to which timestamp column will be cast.
   *
   * @param type Type of timestamp
   * @return this for chaining
   */
  orc_reader_options_builder& timestamp_type(data_type type)
  {
    options.set_timestamp_type(type);
    return *this;
  }

  /**
   * @brief Columns that should be read as 128-bit Decimal
   *
   * @param val Vector of column names
   * @return this for chaining
   */
  orc_reader_options_builder& decimal128_columns(std::vector<std::string> val)
  {
    options.set_decimal128_columns(std::move(val));
    return *this;
  }

  /**
   * @brief Set whether to ignore writer timezone in the stripe footer.
   *
   * @param ignore Boolean value to enable/disable ignoring writer timezone
   * @return this for chaining
   */
  orc_reader_options_builder& ignore_timezone_in_stripe_footer(bool ignore)
  {
    options.enable_ignore_timezone_in_stripe_footer(ignore);
    return *this;
  }

  /**
   * @brief move orc_reader_options member once it's built.
   */
  operator orc_reader_options&&() { return std::move(options); }

  /**
   * @brief move orc_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `orc_reader_options` object's r-value reference
   */
  orc_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads an ORC dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  auto source  = cudf::io::source_info("dataset.orc");
 *  auto options = cudf::io::orc_reader_options::builder(source);
 *  auto result  = cudf::io::read_orc(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata.
 *
 * @return The set of columns
 */
table_with_metadata read_orc(
  orc_reader_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief The chunked orc reader class to read an ORC file iteratively into a series of
 * tables, chunk by chunk.
 *
 * This class is designed to address the reading issue when reading very large ORC files such
 * that sizes of their columns exceed the limit that can be stored in cudf columns. By reading the
 * file content by chunks using this class, each chunk is guaranteed to have its size stay within
 * the given limit.
 */
class chunked_orc_reader {
 public:
  /**
   * @brief Default constructor, this should never be used.
   *
   * This is added just to satisfy cython.
   */
  chunked_orc_reader();

  /**
   * @brief Construct the reader from input/output size limits, output row granularity, along with
   * other ORC reader options.
   *
   * The typical usage should be similar to this:
   * ```
   *  do {
   *    auto const chunk = reader.read_chunk();
   *    // Process chunk
   *  } while (reader.has_next());
   *
   * ```
   *
   * If `chunk_read_limit == 0` (i.e., no output limit) and `pass_read_limit == 0` (no temporary
   * memory size limit), a call to `read_chunk()` will read the whole data source and return a table
   * containing all rows.
   *
   * The `chunk_read_limit` parameter controls the size of the output table to be returned per
   * `read_chunk()` call. If the user specifies a 100 MB limit, the reader will attempt to return
   * tables that have a total bytes size (over all columns) of 100 MB or less.
   * This is a soft limit and the code will not fail if it cannot satisfy the limit.
   *
   * The `pass_read_limit` parameter controls how much temporary memory is used in the entire
   * process of loading, decompressing and decoding of data. Again, this is also a soft limit and
   * the reader will try to make the best effort.
   *
   * Finally, the parameter `output_row_granularity` controls the changes in row number of the
   * output chunk. For each call to `read_chunk()`, with respect to the given `pass_read_limit`, a
   * subset of stripes may be loaded, decompressed and decoded into an intermediate table. The
   * reader will then subdivide that table into smaller tables for final output using
   * `output_row_granularity` as the subdivision step.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per `read_chunk()` call,
   *        or `0` if there is no limit
   * @param pass_read_limit Limit on temporary memory usage for reading the data sources,
   *        or `0` if there is no limit
   * @param output_row_granularity The granularity parameter used for subdividing the decoded
   *        table for final output
   * @param options Settings for controlling reading behaviors
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   *
   * @throw cudf::logic_error if `output_row_granularity` is non-positive
   */
  explicit chunked_orc_reader(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    size_type output_row_granularity,
    orc_reader_options const& options,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Construct the reader from input/output size limits along with other ORC reader options.
   *
   * This constructor implicitly call the other constructor with `output_row_granularity` set to
   * `DEFAULT_OUTPUT_ROW_GRANULARITY` rows.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per `read_chunk()` call,
   *        or `0` if there is no limit
   * @param pass_read_limit Limit on temporary memory usage for reading the data sources,
   *        or `0` if there is no limit
   * @param options Settings for controlling reading behaviors
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit chunked_orc_reader(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    orc_reader_options const& options,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Construct the reader from output size limits along with other ORC reader options.
   *
   * This constructor implicitly call the other constructor with `pass_read_limit` set to `0` and
   * `output_row_granularity` set to `DEFAULT_OUTPUT_ROW_GRANULARITY` rows.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per `read_chunk()` call,
   *        or `0` if there is no limit
   * @param options Settings for controlling reading behaviors
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit chunked_orc_reader(
    std::size_t chunk_read_limit,
    orc_reader_options const& options,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Destructor, destroying the internal reader instance.
   */
  ~chunked_orc_reader();

  /**
   * @brief Check if there is any data in the given data sources has not yet read.
   *
   * @return A boolean value indicating if there is any data left to read
   */
  [[nodiscard]] bool has_next() const;

  /**
   * @brief Read a chunk of rows in the given data sources.
   *
   * The sequence of returned tables, if concatenated by their order, guarantees to form a complete
   * dataset as reading the entire given data sources at once.
   *
   * An empty table will be returned if the given sources are empty, or all the data has
   * been read and returned by the previous calls.
   *
   * @return An output `cudf::table` along with its metadata
   */
  [[nodiscard]] table_with_metadata read_chunk() const;

 private:
  std::unique_ptr<cudf::io::orc::detail::chunked_reader> reader;
};

/** @} */  // end of group
/**
 * @addtogroup io_writers
 * @{
 */

/**
 * @brief Builds settings to use for `write_orc()`.
 */
class orc_writer_options_builder;

/**
 * @brief Constants to disambiguate statistics terminology for ORC.
 *
 * ORC refers to its finest granularity of row-grouping as "row group",
 * which corresponds to Parquet "pages".
 * Similarly, ORC's "stripe" corresponds to a Parquet "row group".
 * The following constants disambiguate the terminology for the statistics
 * collected at each level.
 */
static constexpr statistics_freq ORC_STATISTICS_STRIPE    = statistics_freq::STATISTICS_ROWGROUP;
static constexpr statistics_freq ORC_STATISTICS_ROW_GROUP = statistics_freq::STATISTICS_PAGE;

/**
 * @brief Settings to use for `write_orc()`.
 */
class orc_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::SNAPPY;
  // Specify frequency of statistics collection
  statistics_freq _stats_freq = ORC_STATISTICS_ROW_GROUP;
  // Maximum size of each stripe (unless smaller than a single row group)
  size_t _stripe_size_bytes = default_stripe_size_bytes;
  // Maximum number of rows in stripe (unless smaller than a single row group)
  size_type _stripe_size_rows = default_stripe_size_rows;
  // Row index stride (maximum number of rows in each row group)
  size_type _row_index_stride = default_row_index_stride;
  // Set of columns to output
  table_view _table;
  // Optional associated metadata
  std::optional<table_input_metadata> _metadata;
  // Optional footer key_value_metadata
  std::map<std::string, std::string> _user_data;
  // Optional compression statistics
  std::shared_ptr<writer_compression_statistics> _compression_stats;
  // Specify whether string dictionaries should be alphabetically sorted
  bool _enable_dictionary_sort = true;

  friend orc_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit orc_writer_options(sink_info sink, table_view table)
    : _sink(std::move(sink)), _table(std::move(table))
  {
  }

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit orc_writer_options() = default;

  /**
   * @brief Create builder to create `orc_writer_options`.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build `orc_writer_options`
   */
  static orc_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Returns sink info.
   *
   * @return Sink info
   */
  [[nodiscard]] sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression type.
   *
   * @return Compression type
   */
  [[nodiscard]] compression_type get_compression() const { return _compression; }

  /**
   * @brief Whether writing column statistics is enabled/disabled.
   *
   * @return `true` if writing column statistics is enabled
   */
  [[nodiscard]] bool is_enabled_statistics() const
  {
    return _stats_freq != statistics_freq::STATISTICS_NONE;
  }

  /**
   * @brief Returns frequency of statistics collection.
   *
   * @return Frequency of statistics collection
   */
  [[nodiscard]] statistics_freq get_statistics_freq() const { return _stats_freq; }

  /**
   * @brief Returns maximum stripe size, in bytes.
   *
   * @return Maximum stripe size, in bytes
   */
  [[nodiscard]] auto get_stripe_size_bytes() const { return _stripe_size_bytes; }

  /**
   * @brief Returns maximum stripe size, in rows.
   *
   * @return Maximum stripe size, in rows
   */
  [[nodiscard]] auto get_stripe_size_rows() const { return _stripe_size_rows; }

  /**
   * @brief Returns the row index stride.
   *
   * @return Row index stride
   */
  [[nodiscard]] auto get_row_index_stride() const
  {
    auto const unaligned_stride = std::min(_row_index_stride, get_stripe_size_rows());
    return unaligned_stride - unaligned_stride % 8;
  }

  /**
   * @brief Returns table to be written to output.
   *
   * @return Table to be written to output
   */
  [[nodiscard]] table_view get_table() const { return _table; }

  /**
   * @brief Returns associated metadata.
   *
   * @return Associated metadata
   */
  [[nodiscard]] auto const& get_metadata() const { return _metadata; }

  /**
   * @brief Returns Key-Value footer metadata information.
   *
   * @return Key-Value footer metadata information
   */
  [[nodiscard]] std::map<std::string, std::string> const& get_key_value_metadata() const
  {
    return _user_data;
  }

  /**
   * @brief Returns a shared pointer to the user-provided compression statistics.
   *
   * @return Compression statistics
   */
  [[nodiscard]] std::shared_ptr<writer_compression_statistics> get_compression_statistics() const
  {
    return _compression_stats;
  }

  /**
   * @brief Returns whether string dictionaries should be sorted.
   *
   * @return `true` if string dictionaries should be sorted
   */
  [[nodiscard]] bool get_enable_dictionary_sort() const { return _enable_dictionary_sort; }

  // Setters

  /**
   * @brief Sets compression type.
   *
   * @param comp Compression type
   */
  void set_compression(compression_type comp)
  {
    _compression = comp;
    if (comp == compression_type::AUTO) { _compression = compression_type::SNAPPY; }
  }

  /**
   * @brief Choose granularity of statistics collection.
   *
   * The granularity can be set to:
   * - cudf::io::STATISTICS_NONE: No statistics are collected.
   * - cudf::io::ORC_STATISTICS_STRIPE: Statistics are collected for each ORC stripe.
   * - cudf::io::ORC_STATISTICS_ROWGROUP: Statistics are collected for each ORC row group.
   *
   * @param val Frequency of statistics collection
   */
  void enable_statistics(statistics_freq val) { _stats_freq = val; }

  /**
   * @brief Sets the maximum stripe size, in bytes.
   *
   * @param size_bytes Maximum stripe size, in bytes to be set
   *
   * @throw cudf::logic_error if a value below the minimal size is passed
   */
  void set_stripe_size_bytes(size_t size_bytes)
  {
    CUDF_EXPECTS(size_bytes >= 64 << 10, "64KB is the minimum stripe size");
    _stripe_size_bytes = size_bytes;
  }

  /**
   * @brief Sets the maximum stripe size, in rows.
   *
   * If the stripe size is smaller that the row group size, row group size will be reduced to math
   * the stripe size.
   *
   * @param size_rows Maximum stripe size, in rows to be set
   *
   * @throw cudf::logic_error if a value below the minimal number of rows is passed
   */
  void set_stripe_size_rows(size_type size_rows)
  {
    CUDF_EXPECTS(size_rows >= 512, "Maximum stripe size cannot be smaller than 512");
    _stripe_size_rows = size_rows;
  }

  /**
   * @brief Sets the row index stride.
   *
   * Rounded down to a multiple of 8.
   *
   * @param stride Row index stride to be set
   *
   * @throw cudf::logic_error if a value below the minimal row index stride is passed
   */
  void set_row_index_stride(size_type stride)
  {
    CUDF_EXPECTS(stride >= 512, "Row index stride cannot be smaller than 512");
    _row_index_stride = stride;
  }

  /**
   * @brief Sets table to be written to output.
   *
   * @param tbl Table for the output
   */
  void set_table(table_view tbl) { _table = tbl; }

  /**
   * @brief Sets associated metadata
   *
   * @param meta Associated metadata
   */
  void set_metadata(table_input_metadata meta) { _metadata = std::move(meta); }

  /**
   * @brief Sets metadata.
   *
   * @param metadata Key-Value footer metadata
   */
  void set_key_value_metadata(std::map<std::string, std::string> metadata)
  {
    _user_data = std::move(metadata);
  }

  /**
   * @brief Sets the pointer to the output compression statistics.
   *
   * @param comp_stats Pointer to compression statistics to be updated after writing
   */
  void set_compression_statistics(std::shared_ptr<writer_compression_statistics> comp_stats)
  {
    _compression_stats = std::move(comp_stats);
  }

  /**
   * @brief Sets whether string dictionaries should be sorted.
   *
   * @param val Boolean value to enable/disable
   */
  void set_enable_dictionary_sort(bool val) { _enable_dictionary_sort = val; }
};

/**
 * @brief Builds settings to use for `write_orc()`.
 */
class orc_writer_options_builder {
  orc_writer_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  orc_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  orc_writer_options_builder(sink_info const& sink, table_view const& table) : options{sink, table}
  {
  }

  /**
   * @brief Sets compression type.
   *
   * @param comp The compression type to use
   * @return this for chaining
   */
  orc_writer_options_builder& compression(compression_type comp)
  {
    options.set_compression(comp);
    return *this;
  }

  /**
   * @brief Choose granularity of column statistics to be written
   *
   * The granularity can be set to:
   * - cudf::io::STATISTICS_NONE: No statistics are collected.
   * - cudf::io::ORC_STATISTICS_STRIPE: Statistics are collected for each ORC stripe.
   * - cudf::io::ORC_STATISTICS_ROWGROUP: Statistics are collected for each ORC row group.
   *
   * @param val Level of statistics collection
   * @return this for chaining
   */
  orc_writer_options_builder& enable_statistics(statistics_freq val)
  {
    options.enable_statistics(val);
    return *this;
  }

  /**
   * @brief Sets the maximum stripe size, in bytes.
   *
   * @param val maximum stripe size
   * @return this for chaining
   */
  orc_writer_options_builder& stripe_size_bytes(size_t val)
  {
    options.set_stripe_size_bytes(val);
    return *this;
  }

  /**
   * @brief Sets the maximum number of rows in output stripes.
   *
   * @param val maximum number or rows
   * @return this for chaining
   */
  orc_writer_options_builder& stripe_size_rows(size_type val)
  {
    options.set_stripe_size_rows(val);
    return *this;
  }

  /**
   * @brief Sets the row index stride.
   *
   * @param val new row index stride
   * @return this for chaining
   */
  orc_writer_options_builder& row_index_stride(size_type val)
  {
    options.set_row_index_stride(val);
    return *this;
  }

  /**
   * @brief Sets table to be written to output.
   *
   * @param tbl Table for the output
   * @return this for chaining
   */
  orc_writer_options_builder& table(table_view tbl)
  {
    options.set_table(tbl);
    return *this;
  }

  /**
   * @brief Sets associated metadata.
   *
   * @param meta Associated metadata
   * @return this for chaining
   */
  orc_writer_options_builder& metadata(table_input_metadata meta)
  {
    options.set_metadata(std::move(meta));
    return *this;
  }

  /**
   * @brief Sets Key-Value footer metadata.
   *
   * @param metadata Key-Value footer metadata
   * @return this for chaining
   */
  orc_writer_options_builder& key_value_metadata(std::map<std::string, std::string> metadata)
  {
    options.set_key_value_metadata(std::move(metadata));
    return *this;
  }

  /**
   * @brief Sets the pointer to the output compression statistics.
   *
   * @param comp_stats Pointer to compression statistics to be filled once writer is done
   * @return this for chaining
   */
  orc_writer_options_builder& compression_statistics(
    std::shared_ptr<writer_compression_statistics> const& comp_stats)
  {
    options.set_compression_statistics(comp_stats);
    return *this;
  }

  /**
   * @brief Sets whether string dictionaries should be sorted.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  orc_writer_options_builder& enable_dictionary_sort(bool val)
  {
    options.set_enable_dictionary_sort(val);
    return *this;
  }

  /**
   * @brief move orc_writer_options member once it's built.
   */
  operator orc_writer_options&&() { return std::move(options); }

  /**
   * @brief move orc_writer_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `orc_writer_options` object's r-value reference
   */
  orc_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Writes a set of columns to ORC format.
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  auto destination = cudf::io::sink_info("dataset.orc");
 *  auto options     = cudf::io::orc_writer_options::builder(destination, table->view());
 *  cudf::io::write_orc(options);
 * @endcode
 *
 * @note If an exception is thrown during encoding or compression, no data is written to the sink.
 *
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void write_orc(orc_writer_options const& options,
               rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Builds settings to use for `write_orc_chunked()`.
 */
class chunked_orc_writer_options_builder;

/**
 * @brief Settings to use for `write_orc_chunked()`.
 */
class chunked_orc_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::SNAPPY;
  // Specify granularity of statistics collection
  statistics_freq _stats_freq = ORC_STATISTICS_ROW_GROUP;
  // Maximum size of each stripe (unless smaller than a single row group)
  size_t _stripe_size_bytes = default_stripe_size_bytes;
  // Maximum number of rows in stripe (unless smaller than a single row group)
  size_type _stripe_size_rows = default_stripe_size_rows;
  // Row index stride (maximum number of rows in each row group)
  size_type _row_index_stride = default_row_index_stride;
  // Optional associated metadata
  std::optional<table_input_metadata> _metadata;
  // Optional footer key_value_metadata
  std::map<std::string, std::string> _user_data;
  // Optional compression statistics
  std::shared_ptr<writer_compression_statistics> _compression_stats;
  // Specify whether string dictionaries should be alphabetically sorted
  bool _enable_dictionary_sort = true;

  friend chunked_orc_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   */
  chunked_orc_writer_options(sink_info sink) : _sink(std::move(sink)) {}

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit chunked_orc_writer_options() = default;

  /**
   * @brief Create builder to create `chunked_orc_writer_options`.
   *
   * @param sink The sink used for writer output
   *
   * @return Builder to build chunked_orc_writer_options
   */
  static chunked_orc_writer_options_builder builder(sink_info const& sink);

  /**
   * @brief Returns sink info.
   *
   * @return Sink info
   */
  [[nodiscard]] sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression type.
   *
   * @return Compression type
   */
  [[nodiscard]] compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns granularity of statistics collection.
   *
   * @return Granularity of statistics collection
   */
  [[nodiscard]] statistics_freq get_statistics_freq() const { return _stats_freq; }

  /**
   * @brief Returns maximum stripe size, in bytes.
   *
   * @return Maximum stripe size, in bytes
   */
  [[nodiscard]] auto get_stripe_size_bytes() const { return _stripe_size_bytes; }

  /**
   * @brief Returns maximum stripe size, in rows.
   *
   * @return Maximum stripe size, in rows
   */
  [[nodiscard]] auto get_stripe_size_rows() const { return _stripe_size_rows; }

  /**
   * @brief Returns the row index stride.
   *
   * @return Row index stride
   */
  [[nodiscard]] auto get_row_index_stride() const
  {
    auto const unaligned_stride = std::min(_row_index_stride, get_stripe_size_rows());
    return unaligned_stride - unaligned_stride % 8;
  }

  /**
   * @brief Returns associated metadata.
   *
   * @return Associated metadata
   */
  [[nodiscard]] auto const& get_metadata() const { return _metadata; }

  /**
   * @brief Returns Key-Value footer metadata information.
   *
   * @return Key-Value footer metadata information
   */
  [[nodiscard]] std::map<std::string, std::string> const& get_key_value_metadata() const
  {
    return _user_data;
  }

  /**
   * @brief Returns a shared pointer to the user-provided compression statistics.
   *
   * @return Compression statistics
   */
  [[nodiscard]] std::shared_ptr<writer_compression_statistics> get_compression_statistics() const
  {
    return _compression_stats;
  }

  /**
   * @brief Returns whether string dictionaries should be sorted.
   *
   * @return `true` if string dictionaries should be sorted
   */
  [[nodiscard]] bool get_enable_dictionary_sort() const { return _enable_dictionary_sort; }

  // Setters

  /**
   * @brief Sets compression type.
   *
   * @param comp The compression type to use
   */
  void set_compression(compression_type comp)
  {
    _compression = comp;
    if (comp == compression_type::AUTO) { _compression = compression_type::SNAPPY; }
  }

  /**
   * @brief Choose granularity of statistics collection
   *
   * The granularity can be set to:
   * - cudf::io::STATISTICS_NONE: No statistics are collected.
   * - cudf::io::ORC_STATISTICS_STRIPE: Statistics are collected for each ORC stripe.
   * - cudf::io::ORC_STATISTICS_ROWGROUP: Statistics are collected for each ORC row group.
   *
   * @param val Frequency of statistics collection
   */
  void enable_statistics(statistics_freq val) { _stats_freq = val; }

  /**
   * @brief Sets the maximum stripe size, in bytes.
   *
   * @param size_bytes Maximum stripe size, in bytes to be set
   *
   * @throw cudf::logic_error if a value below the minimal stripe size is passed
   */
  void set_stripe_size_bytes(size_t size_bytes)
  {
    CUDF_EXPECTS(size_bytes >= 64 << 10, "64KB is the minimum stripe size");
    _stripe_size_bytes = size_bytes;
  }

  /**
   * @brief Sets the maximum stripe size, in rows.
   *
   * If the stripe size is smaller that the row group size, row group size will be reduced to math
   * the stripe size.
   *
   * @param size_rows Maximum stripe size, in rows to be set
   *
   * @throw cudf::logic_error if a value below the minimal number of rows in a stripe is passed
   */
  void set_stripe_size_rows(size_type size_rows)
  {
    CUDF_EXPECTS(size_rows >= 512, "maximum stripe size cannot be smaller than 512");
    _stripe_size_rows = size_rows;
  }

  /**
   * @brief Sets the row index stride.
   *
   * Rounded down to a multiple of 8.
   *
   * @param stride Row index stride to be set
   *
   * @throw cudf::logic_error if a value below the minimal number of rows in a row group is passed
   */
  void set_row_index_stride(size_type stride)
  {
    CUDF_EXPECTS(stride >= 512, "Row index stride cannot be smaller than 512");
    _row_index_stride = stride;
  }

  /**
   * @brief Sets associated metadata.
   *
   * @param meta Associated metadata
   */
  void metadata(table_input_metadata meta) { _metadata = std::move(meta); }

  /**
   * @brief Sets Key-Value footer metadata.
   *
   * @param metadata Key-Value footer metadata
   */
  void set_key_value_metadata(std::map<std::string, std::string> metadata)
  {
    _user_data = std::move(metadata);
  }

  /**
   * @brief Sets the pointer to the output compression statistics.
   *
   * @param comp_stats Pointer to compression statistics to be updated after writing
   */
  void set_compression_statistics(std::shared_ptr<writer_compression_statistics> comp_stats)
  {
    _compression_stats = std::move(comp_stats);
  }

  /**
   * @brief Sets whether string dictionaries should be sorted.
   *
   * @param val Boolean value to enable/disable
   */
  void set_enable_dictionary_sort(bool val) { _enable_dictionary_sort = val; }
};

/**
 * @brief Builds settings to use for `write_orc_chunked()`.
 */
class chunked_orc_writer_options_builder {
  chunked_orc_writer_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  chunked_orc_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   */
  explicit chunked_orc_writer_options_builder(sink_info const& sink) : options{sink} {}

  /**
   * @brief Sets compression type.
   *
   * @param comp The compression type to use
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& compression(compression_type comp)
  {
    options.set_compression(comp);
    return *this;
  }

  /**
   * @brief Choose granularity of statistics collection
   *
   * The granularity can be set to:
   * - cudf::io::STATISTICS_NONE: No statistics are collected.
   * - cudf::io::ORC_STATISTICS_STRIPE: Statistics are collected for each ORC stripe.
   * - cudf::io::ORC_STATISTICS_ROWGROUP: Statistics are collected for each ORC row group.
   *
   * @param val Frequency of statistics collection
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& enable_statistics(statistics_freq val)
  {
    options.enable_statistics(val);
    return *this;
  }

  /**
   * @brief Sets the maximum stripe size, in bytes.
   *
   * @param val maximum stripe size
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& stripe_size_bytes(size_t val)
  {
    options.set_stripe_size_bytes(val);
    return *this;
  }

  /**
   * @brief Sets the maximum number of rows in output stripes.
   *
   * @param val maximum number or rows
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& stripe_size_rows(size_type val)
  {
    options.set_stripe_size_rows(val);
    return *this;
  }

  /**
   * @brief Sets the row index stride.
   *
   * @param val new row index stride
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& row_index_stride(size_type val)
  {
    options.set_row_index_stride(val);
    return *this;
  }

  /**
   * @brief Sets associated metadata.
   *
   * @param meta Associated metadata
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& metadata(table_input_metadata meta)
  {
    options.metadata(std::move(meta));
    return *this;
  }

  /**
   * @brief Sets Key-Value footer metadata.
   *
   * @param metadata Key-Value footer metadata
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& key_value_metadata(
    std::map<std::string, std::string> metadata)
  {
    options.set_key_value_metadata(std::move(metadata));
    return *this;
  }

  /**
   * @brief Sets the pointer to the output compression statistics.
   *
   * @param comp_stats Pointer to compression statistics to be filled once writer is done
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& compression_statistics(
    std::shared_ptr<writer_compression_statistics> const& comp_stats)
  {
    options.set_compression_statistics(comp_stats);
    return *this;
  }

  /**
   * @brief Sets whether string dictionaries should be sorted.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  chunked_orc_writer_options_builder& enable_dictionary_sort(bool val)
  {
    options.set_enable_dictionary_sort(val);
    return *this;
  }

  /**
   * @brief move chunked_orc_writer_options member once it's built.
   */
  operator chunked_orc_writer_options&&() { return std::move(options); }

  /**
   * @brief move chunked_orc_writer_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `chunked_orc_writer_options` object's r-value reference
   */
  chunked_orc_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Chunked orc writer class writes an ORC file in a chunked/stream form.
 *
 * The intent of the write_orc_chunked_ path is to allow writing of an
 * arbitrarily large / arbitrary number of rows to an ORC file in multiple passes.
 *
 * The following code snippet demonstrates how to write a single ORC file containing
 * one logical table by writing a series of individual cudf::tables.
 * @code
 *  ...
 *  std::string filepath = "dataset.orc";
 *  cudf::io::chunked_orc_writer_options options = cudf::io::chunked_orc_writer_options
 * options::builder(cudf::sink_info(filepath));
 *  ...
 *  orc_chunked_writer writer(options)
 *  writer.write(table0)
 *  writer.write(table1)
 *    ...
 *  writer.close();
 * @endcode
 */
class orc_chunked_writer {
 public:
  /**
   * @brief Default constructor, this should never be used.
   *        This is added just to satisfy cython.
   */
  orc_chunked_writer();

  /**
   * @brief virtual destructor, Added so we don't leak detail types.
   */
  ~orc_chunked_writer();

  /**
   * @brief Constructor with chunked writer options
   *
   * @param[in] options options used to write table
   * @param[in] stream CUDA stream used for device memory operations and kernel launches
   */
  orc_chunked_writer(chunked_orc_writer_options const& options,
                     rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Writes table to output.
   *
   * @note If an exception is thrown during encoding or compression, the data from the failing call
   * is not written to the sink. Data from previous successful calls is unaffected.
   *
   * @param[in] table Table that needs to be written
   * @return returns reference of the class object
   */
  orc_chunked_writer& write(table_view const& table);

  /**
   * @brief Finishes the chunked/streamed write process.
   */
  void close();

  /// Unique pointer to impl writer class
  std::unique_ptr<orc::detail::writer> writer;
};

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
