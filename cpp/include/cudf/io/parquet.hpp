/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/**
 * @file
 * @brief APIs for reading and writing Parquet files.
 */

namespace CUDF_EXPORT cudf {
namespace io {
/**
 * @addtogroup io_readers
 * @{
 */

constexpr size_t default_row_group_size_bytes =
  std::numeric_limits<size_t>::max();                          ///< Infinite bytes per row group
constexpr size_type default_row_group_size_rows = 1'000'000;   ///< 1 million rows per row group
constexpr size_t default_max_page_size_bytes    = 512 * 1024;  ///< 512KB per page
constexpr size_type default_max_page_size_rows  = 20000;       ///< 20k rows per page
constexpr int32_t default_column_index_truncate_length = 64;   ///< truncate to 64 bytes
constexpr size_t default_max_dictionary_size           = 1024 * 1024;  ///< 1MB dictionary size
constexpr size_type default_max_page_fragment_size     = 5000;  ///< 5000 rows per page fragment

/**
 * @brief Check if the compression type is supported for reading Parquet files.
 *
 * @note This is a runtime check. Some compression types may not be supported because of the current
 * system configuration.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported
 */
[[nodiscard]] bool is_supported_read_parquet(compression_type compression);

/**
 * @brief Check if the compression type is supported for writing Parquet files.
 *
 * @note This is a runtime check. Some compression types may not be supported because of the current
 * system configuration.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported
 */
[[nodiscard]] bool is_supported_write_parquet(compression_type compression);

class parquet_reader_options_builder;

/**
 * @brief Settings for `read_parquet()`.
 */
class parquet_reader_options {
  source_info _source;

  // Column selection options. At most one of these may be set at a time.

  // Path in schema of column names to read; `nullopt` is all
  std::optional<std::vector<std::string>> _column_names;
  // Indices of top-level columns to read; `nullopt` is all
  std::optional<std::vector<cudf::size_type>> _column_indices;
  // Parquet field IDs of columns/fields to read; `nullopt` is all
  std::optional<std::vector<int32_t>> _column_field_ids;

  // List of individual row groups to read (ignored if empty)
  std::vector<std::vector<size_type>> _row_groups;
  // Number of rows to skip from the start; Parquet stores the number of rows as int64_t
  int64_t _skip_rows = 0;
  // Number of rows to read; `nullopt` is all
  std::optional<int64_t> _num_rows;

  // Read row groups that start at or after this byte offset into the source
  size_t _skip_bytes = 0;
  // Read row groups that start before _num_bytes bytes after _skip_bytes into the source
  std::optional<size_t> _num_bytes;

  // Predicate filter as AST to filter output rows.
  std::optional<std::reference_wrapper<ast::expression const>> _filter;

  // Whether to store string data as categorical type
  bool _convert_strings_to_categories = false;
  // Whether to use PANDAS metadata to load columns
  bool _use_pandas_metadata = true;
  // Whether to read and use ARROW schema
  bool _use_arrow_schema = true;
  // Whether to allow reading matching select columns from mismatched Parquet files.
  bool _allow_mismatched_pq_schemas = false;
  // Whether to ignore non-existent projected columns
  bool _ignore_missing_columns = true;
  // Cast timestamp columns to a specific type
  data_type _timestamp_type{type_id::EMPTY};
  // Cast decimal columns to a specific width
  type_id _decimal_width{type_id::EMPTY};
  // Whether to use JIT compilation for filtering
  bool _use_jit_filter = false;
  // Whether to output flat string columns as DICT32 encoded columns
  bool _output_dict_columns = false;
  // Whether column name matching is case sensitive. In case of multiple
  // case-insensitive matches, the first matched column is selected
  bool _case_sensitive_names = true;
  // Whether to prepend a source file index column to the output
  bool _prepend_source_index_column = false;
  // Whether to prepend a file-local row index column to the output
  bool _prepend_row_index_column = false;

  std::optional<std::vector<reader_column_schema>> _reader_column_schema;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read parquet file
   */
  explicit parquet_reader_options(source_info src) : _source{std::move(src)} {}

  friend parquet_reader_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   * The `hybrid_scan_reader` also uses this to create `parquet_reader_options` without a source.
   */
  explicit parquet_reader_options() = default;

  /**
   * @brief Creates a `parquet_reader_options_builder` to build `parquet_reader_options`.
   *        By default, build with empty data source info.
   *
   * @param src Source information to read parquet file
   * @return Builder to build reader options
   */
  static parquet_reader_options_builder builder(source_info src = source_info{});

  /**
   * @brief Returns source info.
   *
   * @return Source info
   */
  [[nodiscard]] source_info const& get_source() const { return _source; }

  /**
   * @brief Returns boolean depending on whether strings should be converted to categories.
   *
   * @return `true` if strings should be converted to categories
   */
  [[nodiscard]] bool is_enabled_convert_strings_to_categories() const
  {
    return _convert_strings_to_categories;
  }

  /**
   * @brief Returns boolean depending on whether to use pandas metadata while reading.
   *
   * @return `true` if pandas metadata is used while reading
   */
  [[nodiscard]] bool is_enabled_use_pandas_metadata() const { return _use_pandas_metadata; }

  /**
   * @brief Returns boolean depending on whether to use arrow schema while reading.
   *
   * @return `true` if arrow schema is used while reading
   */
  [[nodiscard]] bool is_enabled_use_arrow_schema() const { return _use_arrow_schema; }

  /**
   * @brief Returns boolean depending on whether to read matching projected and filter columns
   * from mismatched Parquet sources.
   *
   * @return `true` if mismatched projected and filter columns will be read from mismatched Parquet
   * sources.
   */
  [[nodiscard]] bool is_enabled_allow_mismatched_pq_schemas() const
  {
    return _allow_mismatched_pq_schemas;
  }

  /**
   * @brief Returns boolean depending on whether to ignore non-existent projected columns while
   * reading.
   *
   * @return `true` if non-existent projected columns will be ignored while reading.
   *
   */
  [[nodiscard]] bool is_enabled_ignore_missing_columns() const { return _ignore_missing_columns; }

  /**
   * @brief Returns optional tree of metadata.
   *
   * @return vector of reader_column_schema objects.
   */
  [[nodiscard]] std::optional<std::vector<reader_column_schema>> get_column_schema() const
  {
    return _reader_column_schema;
  }

  /**
   * @brief Returns number of rows to skip from the start.
   *
   * @return Number of rows to skip from the start
   */
  [[nodiscard]] int64_t get_skip_rows() const { return _skip_rows; }

  /**
   * @brief Returns number of rows to read.
   *
   * @return Number of rows to read; `nullopt` if the option hasn't been set (in which case the file
   * is read until the end)
   */
  [[nodiscard]] std::optional<int64_t> const& get_num_rows() const { return _num_rows; }

  /**
   * @brief Returns bytes to skip before starting reading row groups
   *
   * @return Bytes to skip before starting reading row groups; only valid for single parquet source
   * case
   */
  [[nodiscard]] size_t get_skip_bytes() const { return _skip_bytes; }

  /**
   * @brief Returns number of bytes after skipping to end reading row groups at
   *
   * @return Number of bytes after skipping to end reading row groups at; only valid for single
   * parquet source case
   */
  [[nodiscard]] std::optional<size_t> const& get_num_bytes() const { return _num_bytes; }

  /**
   * @brief Returns names of column to be read, if set.
   *
   * @return Names of column to be read; `nullopt` if the option is not set
   */
  [[nodiscard]] [[deprecated("Use `get_column_names` instead.")]] auto const& get_columns() const
  {
    return _column_names;
  }

  /**
   * @brief Returns names of column to be read, if set.
   *
   * @return Names of column to be read; `nullopt` if the option is not set
   */
  [[nodiscard]] auto const& get_column_names() const { return _column_names; }

  /**
   * @brief Returns indices of top-level columns to be read, if set.
   *
   * @return Indices of top-level columns to be read; `nullopt` if the option is not set
   */
  [[nodiscard]] auto const& get_column_indices() const { return _column_indices; }

  /**
   * @brief Returns Parquet field IDs of columns/fields to be read, if set.
   *
   * @return Parquet field IDs of columns/fields to be read; `nullopt` if the option is not set
   */
  [[nodiscard]] auto const& get_column_field_ids() const { return _column_field_ids; }

  /**
   * @brief Returns list of individual row groups to be read.
   *
   * @return List of individual row groups to be read
   */
  [[nodiscard]] auto const& get_row_groups() const { return _row_groups; }

  /**
   * @brief Returns AST based filter for predicate pushdown.
   *
   * @return AST expression to use as filter
   */
  [[nodiscard]] auto const& get_filter() const { return _filter; }

  /**
   * @brief Returns timestamp type used to cast timestamp columns.
   *
   * @return Timestamp type used to cast timestamp columns
   */
  [[nodiscard]] data_type get_timestamp_type() const { return _timestamp_type; }

  /**
   * @brief Returns decimal width used to cast decimal columns.
   *
   * @return Decimal type_id used to cast decimal columns
   */
  [[nodiscard]] type_id get_decimal_width() const { return _decimal_width; }

  /**
   * @brief Returns whether to use JIT compilation for filtering.
   *
   * @return `true` if JIT compilation should be used for filtering
   */
  [[nodiscard]] bool is_enabled_use_jit_filter() const { return _use_jit_filter; }

  /**
   * @brief Returns whether column name matching is case sensitive.
   *
   * @note When disabled, if there are multiple case-insensitive matches, the first
   * matched column is selected from the Parquet schema.
   *
   * @return `true` if column name matching is case sensitive (default)
   */
  [[nodiscard]] bool is_enabled_case_sensitive_names() const { return _case_sensitive_names; }

  /**
   * @brief Returns whether to prepend a source file index column to the output.
   *
   * @return `true` if a source file index column should be prepended
   */
  [[nodiscard]] bool is_enabled_prepend_source_index_column() const
  {
    return _prepend_source_index_column;
  }

  /**
   * @brief Returns whether to prepend a file-local row index column to the output.
   *
   * The row index column contains, for each output row, the row's index within its parquet
   * source file. If the source index column is also enabled, the column order is: source index,
   * row index, data columns.
   *
   * @return `true` if a row index column should be prepended
   */
  [[nodiscard]] bool is_enabled_prepend_row_index_column() const
  {
    return _prepend_row_index_column;
  }

  /**
   * @brief Returns whether the reader returns flat string columns as DICTIONARY32 encoded columns
   *
   * When true, the reader outputs STRING columns as DICTIONARY32 encoded columns. A DICTIONARY32
   * column consists of an INT32 indices child and a STRING keys child.
   *
   * When AST/JIT filters are set, the direct transcode fast path is disabled.
   * String columns are materialized, then operated on by the filter. The filtered results are then
   * encoded as DICTIONARY32 columns.
   *
   * @return `true` if the reader returns flat string columns as DICTIONARY32 encoded columns
   */
  [[nodiscard]] bool is_enabled_output_dict_columns() const { return _output_dict_columns; }

  /**
   * @brief Set a new source location
   *
   * @param src New `source_info`.
   */
  void set_source(source_info src) { _source = std::move(src); }

  /**
   * @brief Sets the names of columns to be read from all input sources.
   *
   * @deprecated Deprecated in 26.04 and will be removed in 26.06+. Use `set_column_names` instead.
   *
   * Applies the same list of column names across all sources. Unlike `set_row_groups`,
   * which allows per-source configuration, `set_columns` applies globally.
   *
   * Columns that do not exist in the input files will be ignored silently and the output table will
   * only include the columns that are actually found. This behavior can be changed by setting
   * `enable_ignore_missing_columns` to false.
   *
   * To select a nested column (e.g., a struct member), use dot notation.
   *
   * Example:
   * To read only the `bar` and `baz` fields, call:
   *   set_columns({"foo.bar", "foo.baz"});
   *
   * @note This function does not currently support per-source column selection.
   *
   * @param column_names A vector of column names to attempt to read from each input source.
   */
  [[deprecated("Use `set_column_names` instead.")]] void set_columns(
    std::vector<std::string> column_names)
  {
    set_column_names(std::move(column_names));
  }

  /**
   * @brief Sets the names of columns to be read from all input sources.
   *
   * Applies the same list of column names across all sources. Unlike `set_row_groups`,
   * which allows per-source configuration, `set_columns` applies globally.
   *
   * Columns that do not exist in the input files will be ignored silently and the output table will
   * only include the columns that are actually found. This behavior can be changed by setting
   * `enable_ignore_missing_columns` to false.
   *
   * To select a nested column (e.g., a struct member), use dot notation.
   *
   * Example:
   * To read only the `bar` and `baz` fields, call:
   *   set_column_names({"foo.bar", "foo.baz"});
   *
   * @note This function does not currently support per-source column selection.
   *
   * @param column_names A vector of column names to attempt to read from each input source.
   */
  void set_column_names(std::vector<std::string> column_names)
  {
    CUDF_EXPECTS(not _column_indices.has_value(),
                 "Cannot select columns by names and indices simultaneously");
    CUDF_EXPECTS(not _column_field_ids.has_value(),
                 "Cannot select columns by names and field IDs simultaneously");
    _column_names = std::move(column_names);
  }

  /**
   * @brief Sets the indices of top-level columns to be read from all input sources.
   *
   * Applies the same list of top-level column indices across all sources. Unlike `set_row_groups`,
   * which allows per-source configuration, `set_column_indices` applies globally.
   *
   * Note that `set_column_indices` can only be used to select top-level columns. unlike
   * `set_columns` which can also select nested columns.
   *
   * @param col_indices A vector of column indices to attempt to read from each input source.
   */
  void set_column_indices(std::vector<cudf::size_type> col_indices)
  {
    CUDF_EXPECTS(not _column_names.has_value(),
                 "Cannot select columns by indices and names simultaneously");
    CUDF_EXPECTS(not _column_field_ids.has_value(),
                 "Cannot select columns by indices and field IDs simultaneously");
    CUDF_EXPECTS(
      not _allow_mismatched_pq_schemas,
      "Cannot select columns by indices and allow mismatched Parquet schemas simultaneously");
    _column_indices = std::move(col_indices);
  }

  /**
   * @brief Sets the Parquet field IDs of columns/fields to be read from all input sources.
   *
   * @param column_field_ids A vector of Parquet field IDs to attempt to read from each input
   * source.
   */
  void set_column_field_ids(std::vector<int32_t> column_field_ids)
  {
    CUDF_EXPECTS(not _column_names.has_value(),
                 "Cannot select columns by field IDs and names simultaneously");
    CUDF_EXPECTS(not _column_indices.has_value(),
                 "Cannot select columns by field IDs and indices simultaneously");
    _column_field_ids = std::move(column_field_ids);
  }

  /**
   * @brief Specifies which row groups to read from each input source.
   *
   * When reading from multiple sources (e.g., multiple files), this function allows selecting
   * specific row groups for each source individually. The outer vector corresponds to the list
   * of input sources, and each inner vector contains the row group indices to read from the
   * respective source.
   *
   * If no row groups should be read from a given source, its entry should be an empty vector.
   *
   * Example:
   * To read row groups [0, 2] from the first input and [1] from the second input, call:
   *   set_row_groups({{0, 2}, {1}});
   *
   * Output ordering: rows are emitted in input-source order; all rows selected from source 0
   * are emitted before rows selected from source 1, and so on. Within each source, row groups
   * appear in the exact order given by the inner vector; the reader does not sort or deduplicate
   * the indices, and repeated indices are emitted multiple times. An empty inner vector means that
   * source contributes no rows but does not affect the order of the remaining sources. When this
   * setter is not called, all row groups are read in source order, then in on-disk order within
   * each source. Row groups removed by standard `read_parquet` predicate pushdown (statistics or
   * bloom filter pruning) are dropped in place; the remaining row groups keep their relative order.
   *
   * @param row_groups A vector of vectors, one per input source, each specifying the
   *                   row group indices to read from that source.
   */
  void set_row_groups(std::vector<std::vector<size_type>> row_groups);

  /**
   * @brief Sets AST based filter for predicate pushdown.
   *
   * The filter can utilize cudf::ast::column_name_reference to reference a column by its name,
   * even if it's not necessarily present in the requested projected columns.
   * To refer to output column indices, you can use cudf::ast::column_reference.
   *
   * For a parquet with columns ["A", "B", "C", ... "X", "Y", "Z"],
   * Example 1: with/without column projection
   * @code
   * use_columns({"A", "X", "Z"})
   * .filter(operation(ast_operator::LESS, column_name_reference{"C"}, literal{100}));
   * @endcode
   * Column "C" need not be present in output table.
   * Example 2: without column projection
   * @code
   * filter(operation(ast_operator::LESS, column_reference{1}, literal{100}));
   * @endcode
   * Here, `1` will refer to column "B" because output will contain all columns in
   * order ["A", ..., "Z"].
   * Example 3: with column projection
   * @code
   * use_columns({"A", "Z", "X"})
   * .filter(operation(ast_operator::LESS, column_reference{1}, literal{100}));
   * @endcode
   * Here, `1` will refer to column "Z" because output will contain 3 columns in
   * order ["A", "Z", "X"].
   *
   * @param filter AST expression to use as filter
   */
  void set_filter(ast::expression const& filter) { _filter = filter; }

  /**
   * @brief Sets to enable/disable conversion of strings to categories.
   *
   * @param val Boolean value to enable/disable conversion of string columns to categories
   */
  void enable_convert_strings_to_categories(bool val) { _convert_strings_to_categories = val; }

  /**
   * @brief Sets to enable/disable use of pandas metadata to read.
   *
   * @param val Boolean indicating whether to use pandas metadata
   */
  void enable_use_pandas_metadata(bool val) { _use_pandas_metadata = val; }

  /**
   * @brief Sets to enable/disable use of arrow schema to read.
   *
   * @param val Boolean indicating whether to use arrow schema
   */
  void enable_use_arrow_schema(bool val) { _use_arrow_schema = val; }

  /**
   * @brief Sets to enable/disable reading of matching projected and filter columns from mismatched
   * Parquet sources.
   *
   * @param val Boolean indicating whether to read matching projected and filter columns from
   * mismatched Parquet sources.
   */
  void enable_allow_mismatched_pq_schemas(bool val)
  {
    CUDF_EXPECTS(
      not val or not _column_indices.has_value(),
      "Cannot enable reading mismatched Parquet schemas when selecting columns by index");
    _allow_mismatched_pq_schemas = val;
  }

  /**
   * @brief Sets to enable/disable ignoring of non-existent projected columns while reading.
   *
   * @param val Boolean indicating whether to ignore non-existent projected columns while reading.
   *
   */
  void enable_ignore_missing_columns(bool val) { _ignore_missing_columns = val; }

  /**
   * @brief Sets reader column schema.
   *
   * @param val Tree of schema nodes to enable/disable conversion of binary to string columns.
   * Note default is to convert to string columns.
   */
  void set_column_schema(std::vector<reader_column_schema> val)
  {
    _reader_column_schema = std::move(val);
  }

  /**
   * @brief Sets number of rows to skip.
   *
   * @param val Number of rows to skip from start
   */
  void set_skip_rows(int64_t val);

  /**
   * @brief Sets number of rows to read.
   *
   * @note Although this allows one to request more than `size_type::max()` rows, if any
   * single read would produce a table larger than this row limit, an error is thrown.
   *
   * @param val Number of rows to read after skip
   */
  void set_num_rows(int64_t val);

  /**
   * @brief Sets bytes to skip before starting reading row groups.
   *
   * @param val Bytes to skip before starting reading row groups
   */
  void set_skip_bytes(size_t val);

  /**
   * @brief Sets number of bytes after skipping to end reading row groups at.
   *
   * @param val Number of bytes after skipping to end reading row groups at
   */
  void set_num_bytes(size_t val);

  /**
   * @brief Sets timestamp_type used to cast timestamp columns.
   *
   * @param type The timestamp data_type to which all timestamp columns need to be cast
   */
  void set_timestamp_type(data_type type) { _timestamp_type = type; }

  /**
   * @brief Sets decimal width used to cast decimal columns.
   *
   * @param width The decimal type_id (DECIMAL32, DECIMAL64, or DECIMAL128) to which all decimal
   * columns need to be cast. The scale of each column is preserved from the file.
   */
  void set_decimal_width(type_id width) { _decimal_width = width; }

  /**
   * @brief Sets whether to use JIT for filtering.
   *
   * @param val Boolean indicating whether to enable JIT filtering.
   */
  void enable_use_jit_filter(bool val) { _use_jit_filter = val; }

  /**
   * @brief Sets whether column name matching is case sensitive.
   *
   * @note When disabled, if there are multiple case-insensitive matches, the first
   * matched column is selected from the Parquet schema.
   *
   * @param val Boolean indicating whether to enable case-sensitive matching.
   */
  void enable_case_sensitive_names(bool val) { _case_sensitive_names = val; }

  /**
   * @brief Sets whether to prepend a source file index column to the output.
   *
   * @param val Boolean indicating whether to prepend the source file index column.
   */
  void enable_prepend_source_index_column(bool val) { _prepend_source_index_column = val; }

  /**
   * @brief Sets whether to prepend a file-local row index column to the output.
   *
   * @param val Boolean indicating whether to prepend the row index column.
   */
  void enable_prepend_row_index_column(bool val) { _prepend_row_index_column = val; }

  /**
   * @brief Sets to enable/disable trying to output DICTIONARY32 columns for flat string columns.
   *
   * @param val Boolean indicating whether to output DICTIONARY32 columns for flat string columns
   */
  void enable_output_dict_columns(bool val) { _output_dict_columns = val; }
};

/**
 * @brief Builds parquet_reader_options to use for `read_parquet()`.
 */
class parquet_reader_options_builder {
  parquet_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   * The `hybrid_scan_reader` also uses this to construct `parquet_reader_options` without a source.
   */
  parquet_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read parquet file
   */
  explicit parquet_reader_options_builder(source_info src) : options{std::move(src)} {}

  /**
   * @brief Sets names of the columns to be read.
   *
   * @deprecated Deprecated in 26.04 and will be removed in 26.06+. Use `column_names` instead.
   *
   * @param column_names Vector of column names
   * @return this for chaining
   */
  [[deprecated("Use `column_names` instead.")]] parquet_reader_options_builder& columns(
    std::vector<std::string> column_names)
  {
    return this->column_names(std::move(column_names));
  }

  /**
   * @brief Sets names of the columns to be read.
   *
   * @param column_names Vector of column names
   * @return this for chaining
   */
  parquet_reader_options_builder& column_names(std::vector<std::string> column_names)
  {
    options.set_column_names(std::move(column_names));
    return *this;
  }

  /**
   * @brief Sets the indices of top-level columns to be read from all input sources.
   *
   * @param col_indices A vector of column indices to attempt to read from each input source.
   * @return this for chaining
   */
  parquet_reader_options_builder& column_indices(std::vector<cudf::size_type> col_indices)
  {
    options.set_column_indices(std::move(col_indices));
    return *this;
  }

  /**
   * @brief Sets the Parquet field IDs of columns/fields to be read from all input sources.
   *
   * @param column_field_ids A vector of Parquet field IDs to attempt to read from each input
   * source.
   * @return this for chaining
   */
  parquet_reader_options_builder& column_field_ids(std::vector<int32_t> column_field_ids)
  {
    options.set_column_field_ids(std::move(column_field_ids));
    return *this;
  }

  /**
   * @copydoc parquet_reader_options::set_row_groups
   * @return this for chaining
   */
  parquet_reader_options_builder& row_groups(std::vector<std::vector<size_type>> row_groups)
  {
    options.set_row_groups(std::move(row_groups));
    return *this;
  }

  /**
   * @copydoc parquet_reader_options::set_filter
   * @return this for chaining
   */
  parquet_reader_options_builder& filter(ast::expression const& filter)
  {
    options.set_filter(filter);
    return *this;
  }

  /**
   * @brief Sets enable/disable conversion of strings to categories.
   *
   * @param val Boolean value to enable/disable conversion of string columns to categories
   * @return this for chaining
   */
  parquet_reader_options_builder& convert_strings_to_categories(bool val)
  {
    options.enable_convert_strings_to_categories(val);
    return *this;
  }

  /**
   * @brief Sets to enable/disable use of pandas metadata to read.
   *
   * @param val Boolean value whether to use pandas metadata
   * @return this for chaining
   */
  parquet_reader_options_builder& use_pandas_metadata(bool val)
  {
    options.enable_use_pandas_metadata(val);
    return *this;
  }

  /**
   * @brief Sets to enable/disable use of arrow schema to read.
   *
   * @param val Boolean value whether to use arrow schema
   * @return this for chaining
   */
  parquet_reader_options_builder& use_arrow_schema(bool val)
  {
    options.enable_use_arrow_schema(val);
    return *this;
  }

  /**
   * @brief Sets to enable/disable reading of matching projected and filter columns from mismatched
   * Parquet sources.
   *
   * @param val Boolean value whether to read matching projected and filter columns from mismatched
   * Parquet sources.
   *
   * @return this for chaining.
   */
  parquet_reader_options_builder& allow_mismatched_pq_schemas(bool val)
  {
    options.enable_allow_mismatched_pq_schemas(val);
    return *this;
  }

  /**
   * @brief Sets to enable/disable ignoring of non-existent projected columns while reading.
   *
   * @param val Boolean indicating whether to ignore non-existent projected columns while reading.
   *
   * @return this for chaining.
   */
  parquet_reader_options_builder& ignore_missing_columns(bool val)
  {
    options.enable_ignore_missing_columns(val);
    return *this;
  }

  /**
   * @brief Sets reader metadata.
   *
   * @param val Tree of metadata information.
   * @return this for chaining
   */
  parquet_reader_options_builder& set_column_schema(std::vector<reader_column_schema> val)
  {
    options.set_column_schema(std::move(val));
    return *this;
  }

  /**
   * @brief Sets number of rows to skip.
   *
   * @param val Number of rows to skip from start
   * @return this for chaining
   */
  parquet_reader_options_builder& skip_rows(int64_t val)
  {
    options.set_skip_rows(val);
    return *this;
  }

  /**
   * @brief Sets number of rows to read.
   *
   * @note Although this allows one to request more than `size_type::max()` rows, if any
   * single read would produce a table larger than this row limit, an error is thrown.
   *
   * @param val Number of rows to read after skip
   * @return this for chaining
   */
  parquet_reader_options_builder& num_rows(int64_t val)
  {
    options.set_num_rows(val);
    return *this;
  }

  /**
   * @brief Sets bytes to skip before starting reading row groups.
   *
   * @param val Bytes to skip before starting reading row groups
   * @return this for chaining
   */
  parquet_reader_options_builder& skip_bytes(size_t val)
  {
    options.set_skip_bytes(val);
    return *this;
  }

  /**
   * @brief Sets number of bytes after skipping to end reading row groups at.
   *
   * @param val Number of bytes after skipping to end reading row groups at
   * @return this for chaining
   */
  parquet_reader_options_builder& num_bytes(size_t val)
  {
    options.set_num_bytes(val);
    return *this;
  }

  /**
   * @brief timestamp_type used to cast timestamp columns.
   *
   * @param type The timestamp data_type to which all timestamp columns need to be cast
   * @return this for chaining
   */
  parquet_reader_options_builder& timestamp_type(data_type type)
  {
    options.set_timestamp_type(type);
    return *this;
  }

  /**
   * @brief Sets the decimal width used to cast decimal columns.
   *
   * @param width The decimal type_id (DECIMAL32, DECIMAL64, or DECIMAL128) to which all decimal
   * columns need to be cast. The scale of each column is preserved from the file.
   * @return this for chaining
   */
  parquet_reader_options_builder& decimal_width(type_id width)
  {
    options.set_decimal_width(width);
    return *this;
  }

  /**
   * @brief Sets whether to use JIT for filtering.
   *
   * @param val Boolean indicating whether to enable JIT filtering.
   * @return this for chaining
   */
  parquet_reader_options_builder& use_jit_filter(bool val)
  {
    options.enable_use_jit_filter(val);
    return *this;
  }

  /**
   * @brief Sets whether column name matching is case sensitive.
   *
   * @note When disabled, if there are multiple case-insensitive matches, the first
   * matched column is selected from the Parquet schema.
   *
   * @param val Boolean indicating whether to enable case-sensitive matching
   * @return this for chaining
   */
  parquet_reader_options_builder& case_sensitive_names(bool val)
  {
    options.enable_case_sensitive_names(val);
    return *this;
  }

  /**
   * @brief Sets whether to prepend a source file index column to the output.
   *
   * @param val Boolean indicating whether to prepend a source file index column
   * @return this for chaining
   */
  parquet_reader_options_builder& prepend_source_index_column(bool val)
  {
    options.enable_prepend_source_index_column(val);
    return *this;
  }

  /**
   * @brief Sets whether to prepend a file-local row index column to the output.
   *
   * @param val Boolean indicating whether to prepend a row index column
   * @return this for chaining
   */
  parquet_reader_options_builder& prepend_row_index_column(bool val)
  {
    options.enable_prepend_row_index_column(val);
    return *this;
  }

  /**
   * @brief Sets options for enabling/disabling output of DICTIONARY32 columns for flat string
   * columns.
   *
   * @param val Boolean value whether to output flat string columns as DICTIONARY32 encoded columns
   *
   * @note When enabled, the output columns will be of type DICTIONARY32. When disabled, the output
   * columns will be of type STRING.
   *
   * @return this for chaining
   */
  parquet_reader_options_builder& output_dict_columns(bool val)
  {
    options.enable_output_dict_columns(val);
    return *this;
  }

  /**
   * @brief move parquet_reader_options member once it's built.
   */
  operator parquet_reader_options&&() { return std::move(options); }

  /**
   * @brief move parquet_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `parquet_reader_options` object's r-value reference
   */
  parquet_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads a Parquet dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  auto source  = cudf::io::source_info("dataset.parquet");
 *  auto options = cudf::io::parquet_reader_options::builder(source);
 *  auto result  = cudf::io::read_parquet(options);
 * @endcode
 *
 * Row-group selection and output ordering are described in
 * `parquet_reader_options::set_row_groups()`.
 *
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_parquet(
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reads a Parquet dataset into a set of columns using pre-existing Parquet datasources and
 * file metadatas.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  auto sources = cudf::io::make_datasources(cudf::io::source_info("dataset.parquet"));
 *  // Pass read_page_indexes=false when reusing only with read_parquet (page indexes optional).
 *  auto metadatas = cudf::io::read_parquet_footers(sources, false);
 *  auto options = cudf::io::parquet_reader_options::builder();
 *  auto result  = cudf::io::read_parquet(std::move(sources), std::move(metadatas), options);
 * @endcode
 *
 * Row-group selection and output ordering are described in
 * `parquet_reader_options::set_row_groups()`.
 *
 * @param sources Input `datasource` objects to read the dataset from
 * @param parquet_metadatas Pre-materialized Parquet file metadata(s). Read from sources if empty
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_parquet(
  std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
  std::vector<parquet::FileMetaData>&& parquet_metadatas,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief The chunked parquet reader class to read Parquet file iteratively in to a series of
 * tables, chunk by chunk.
 *
 * This class is designed to address the reading issue when reading very large Parquet files such
 * that the sizes of their column exceed the limit that can be stored in cudf column. By reading the
 * file content by chunks using this class, each chunk is guaranteed to have its sizes stay within
 * the given limit.
 */
class chunked_parquet_reader {
 public:
  /**
   * @brief Default constructor, this should never be used.
   *
   * This is added just to satisfy cython.
   * This is added to not leak detail API
   */
  chunked_parquet_reader();

  /**
   * @brief Constructor for chunked reader.
   *
   * This constructor requires the same `parquet_reader_option` parameter as in
   * `cudf::read_parquet()`, and an additional parameter to specify the size byte limit of the
   * output table for each reading.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   *        or `0` if there is no limit
   * @param options The options used to read Parquet file
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  chunked_parquet_reader(
    std::size_t chunk_read_limit,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Constructor for chunked reader using pre-existing Parquet datasources and
   * file metadatas.
   *
   * This constructor requires the same `parquet_reader_option` parameter as in
   * `cudf::read_parquet()`, and an additional parameter to specify the size byte limit of the
   * output table for each reading.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   *        or `0` if there is no limit
   * @param sources Input `datasource` objects to read the dataset from
   * @param parquet_metadatas Pre-materialized Parquet file metadata(s). Read from sources if empty
   * @param options The options used to read Parquet file
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  chunked_parquet_reader(
    std::size_t chunk_read_limit,
    std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
    std::vector<parquet::FileMetaData>&& parquet_metadatas,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Constructor for chunked reader.
   *
   * This constructor requires the same `parquet_reader_option` parameter as in
   * `cudf::read_parquet()`, with additional parameters to specify the size byte limit of the
   * output table for each reading, and a byte limit on the amount of temporary memory to use
   * when reading. pass_read_limit affects how many row groups we can read at a time by limiting
   * the amount of memory dedicated to decompression space. pass_read_limit is a hint, not an
   * absolute limit - if a single row group cannot fit within the limit given, it will still be
   * loaded.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   * or `0` if there is no limit
   * @param pass_read_limit Limit on the amount of memory used for reading and decompressing data or
   * `0` if there is no limit
   * @param options The options used to read Parquet file
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  chunked_parquet_reader(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Constructor for chunked reader using pre-existing Parquet datasources and
   * file metadatas.
   *
   * This constructor requires the same `parquet_reader_option` parameter as in
   * `cudf::read_parquet()`, with additional parameters to specify the size byte limit of the
   * output table for each reading, and a byte limit on the amount of temporary memory to use
   * when reading. pass_read_limit affects how many row groups we can read at a time by limiting
   * the amount of memory dedicated to decompression space. pass_read_limit is a hint, not an
   * absolute limit - if a single row group cannot fit within the limit given, it will still be
   * loaded.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   * or `0` if there is no limit
   * @param pass_read_limit Limit on the amount of memory used for reading and decompressing data or
   * `0` if there is no limit
   * @param sources Input `datasource` objects to read the dataset from
   * @param parquet_metadatas Pre-materialized Parquet file metadata(s). Read from sources if empty
   * @param options The options used to read Parquet file
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  chunked_parquet_reader(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
    std::vector<parquet::FileMetaData>&& parquet_metadatas,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Destructor, destroying the internal reader instance.
   *
   * Since the declaration of the internal `reader` object does not exist in this header, this
   * destructor needs to be defined in a separate source file which can access to that object's
   * declaration.
   */
  ~chunked_parquet_reader();

  /**
   * @brief Check if there is any data in the given file has not yet read.
   *
   * @return A boolean value indicating if there is any data left to read
   */
  [[nodiscard]] bool has_next() const;

  /**
   * @brief Read a chunk of rows in the given Parquet file.
   *
   * The sequence of returned tables, if concatenated by their order, guarantees to form a complete
   * dataset as reading the entire given file at once.
   *
   * An empty table will be returned if the given file is empty, or all the data in the file has
   * been read and returned by the previous calls.
   *
   * @return An output `cudf::table` along with its metadata
   */
  [[nodiscard]] table_with_metadata read_chunk() const;

 private:
  std::unique_ptr<cudf::io::parquet::detail::chunked_reader> reader;
};

/** @} */  // end of group
/**
 * @addtogroup io_writers
 * @{
 */

/**
 * @brief Struct used to describe column sorting metadata
 */
struct sorting_column {
  int column_idx{};           //!< leaf column index within the row group
  bool is_descending{false};  //!< true if sort order is descending
  bool is_nulls_first{true};  //!< true if nulls come before non-null values
};

/**
 * @brief Base settings for `write_parquet()` and `chunked_parquet_writer`.
 */
class parquet_writer_options_base {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format to use
  compression_type _compression = compression_type::SNAPPY;
  // Specify the level of statistics in the output file
  statistics_freq _stats_level = statistics_freq::STATISTICS_ROWGROUP;
  // Optional associated metadata
  std::optional<table_input_metadata> _metadata;
  // Optional footer key_value_metadata
  std::vector<std::map<std::string, std::string>> _user_data;
  // Parquet writer can write INT96 or TIMESTAMP_MICROS. Defaults to TIMESTAMP_MICROS.
  // If true then overrides any per-column setting in _metadata.
  bool _write_timestamps_as_int96 = false;
  // Parquet writer can write timestamps as UTC
  // Defaults to true because libcudf timestamps are implicitly UTC
  bool _write_timestamps_as_UTC = true;
  // Whether to write ARROW schema
  bool _write_arrow_schema = false;
  // Maximum size of each row group (unless smaller than a single page)
  size_t _row_group_size_bytes = default_row_group_size_bytes;
  // Maximum number of rows in row group (unless smaller than a single page)
  size_type _row_group_size_rows = default_row_group_size_rows;
  // Maximum size of each page (uncompressed)
  size_t _max_page_size_bytes = default_max_page_size_bytes;
  // Maximum number of rows in a page
  size_type _max_page_size_rows = default_max_page_size_rows;
  // Maximum size of min or max values in column index
  int32_t _column_index_truncate_length = default_column_index_truncate_length;
  // When to use dictionary encoding for data
  dictionary_policy _dictionary_policy = dictionary_policy::ADAPTIVE;
  // Maximum size of column chunk dictionary (in bytes)
  size_t _max_dictionary_size = default_max_dictionary_size;
  // Maximum number of rows in a page fragment
  std::optional<size_type> _max_page_fragment_size;
  // Optional compression statistics
  std::shared_ptr<writer_compression_statistics> _compression_stats;
  // write V2 page headers?
  bool _v2_page_headers = false;
  // enable per-page compression decision for V2?
  bool _page_level_compression = false;
  // Which columns in _table are used for sorting
  std::optional<std::vector<sorting_column>> _sorting_columns;

 protected:
  /**
   * @brief Constructor from sink.
   *
   * @param sink The sink used for writer output
   */
  explicit parquet_writer_options_base(sink_info sink) : _sink(std::move(sink)) {}

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  parquet_writer_options_base() = default;

  /**
   * @brief Returns sink info.
   *
   * @return Sink info
   */
  [[nodiscard]] sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns compression format used.
   *
   * @return Compression format
   */
  [[nodiscard]] compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns level of statistics requested in output file.
   *
   * @return level of statistics requested in output file
   */
  [[nodiscard]] statistics_freq get_stats_level() const { return _stats_level; }

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
  [[nodiscard]] std::vector<std::map<std::string, std::string>> const& get_key_value_metadata()
    const
  {
    return _user_data;
  }

  /**
   * @brief Returns `true` if timestamps will be written as INT96
   *
   * @return `true` if timestamps will be written as INT96
   */
  [[nodiscard]] bool is_enabled_int96_timestamps() const { return _write_timestamps_as_int96; }

  /**
   * @brief Returns `true` if timestamps will be written as UTC
   *
   * @return `true` if timestamps will be written as UTC
   */
  [[nodiscard]] auto is_enabled_utc_timestamps() const { return _write_timestamps_as_UTC; }

  /**
   * @brief Returns `true` if arrow schema will be written
   *
   * @return `true` if arrow schema will be written
   */
  [[nodiscard]] auto is_enabled_write_arrow_schema() const { return _write_arrow_schema; }

  /**
   * @brief Returns maximum row group size, in bytes.
   *
   * @return Maximum row group size, in bytes
   */
  [[nodiscard]] auto get_row_group_size_bytes() const { return _row_group_size_bytes; }

  /**
   * @brief Returns maximum row group size, in rows.
   *
   * @return Maximum row group size, in rows
   */
  [[nodiscard]] auto get_row_group_size_rows() const { return _row_group_size_rows; }

  /**
   * @brief Returns the maximum uncompressed page size, in bytes.
   *
   * If set larger than the row group size, then this will return the row group size.
   *
   * @return Maximum uncompressed page size, in bytes
   */
  [[nodiscard]] auto get_max_page_size_bytes() const
  {
    return std::min(_max_page_size_bytes, get_row_group_size_bytes());
  }

  /**
   * @brief Returns maximum page size, in rows.
   *
   * If set larger than the row group size, then this will return the row group size.
   *
   * @return Maximum page size, in rows
   */
  [[nodiscard]] auto get_max_page_size_rows() const
  {
    return std::min(_max_page_size_rows, get_row_group_size_rows());
  }

  /**
   * @brief Returns maximum length of min or max values in column index, in bytes.
   *
   * @return length min/max will be truncated to
   */
  [[nodiscard]] auto get_column_index_truncate_length() const
  {
    return _column_index_truncate_length;
  }

  /**
   * @brief Returns policy for dictionary use.
   *
   * @return policy for dictionary use
   */
  [[nodiscard]] dictionary_policy get_dictionary_policy() const { return _dictionary_policy; }

  /**
   * @brief Returns maximum dictionary size, in bytes.
   *
   * @return Maximum dictionary size, in bytes.
   */
  [[nodiscard]] auto get_max_dictionary_size() const { return _max_dictionary_size; }

  /**
   * @brief Returns maximum page fragment size, in rows.
   *
   * @return Maximum page fragment size, in rows.
   */
  [[nodiscard]] auto get_max_page_fragment_size() const { return _max_page_fragment_size; }

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
   * @brief Returns `true` if V2 page headers should be written.
   *
   * @return `true` if V2 page headers should be written.
   */
  [[nodiscard]] auto is_enabled_write_v2_headers() const { return _v2_page_headers; }

  /**
   * @brief Returns `true` if per-page compression is enabled for V2 pages.
   *
   * When enabled, each V2 data page can independently decide whether to store
   * compressed or uncompressed based on compression effectiveness. This may
   * produce files that are not readable by all Parquet implementations.
   *
   * @return `true` if per-page compression decision is enabled.
   */
  [[nodiscard]] auto is_enabled_page_level_compression() const { return _page_level_compression; }

  /**
   * @brief Returns the sorting_columns.
   *
   * @return Column sort order metadata
   */
  [[nodiscard]] auto const& get_sorting_columns() const { return _sorting_columns; }

  /**
   * @brief Sets metadata.
   *
   * @param metadata Associated metadata
   */
  void set_metadata(table_input_metadata metadata);

  /**
   * @brief Sets metadata.
   *
   * @param metadata Key-Value footer metadata
   */
  void set_key_value_metadata(std::vector<std::map<std::string, std::string>> metadata);

  /**
   * @brief Sets the level of statistics.
   *
   * @param sf Level of statistics requested in the output file
   */
  void set_stats_level(statistics_freq sf);
  /**
   * @brief Sets compression type.
   *
   * @param compression The compression type to use
   */
  void set_compression(compression_type compression);

  /**
   * @brief Sets timestamp writing preferences. INT96 timestamps will be written
   * if `true` and TIMESTAMP_MICROS will be written if `false`.
   *
   * @param req Boolean value to enable/disable writing of INT96 timestamps
   */
  void enable_int96_timestamps(bool req);

  /**
   * @brief Sets preference for writing timestamps as UTC. Write timestamps as UTC if set to `true`.
   *
   * @param val Boolean value to enable/disable writing of timestamps as UTC.
   */
  void enable_utc_timestamps(bool val);

  /**
   * @brief Sets preference for writing arrow schema. Write arrow schema if set to `true`.
   *
   * @param val Boolean value to enable/disable writing of arrow schema.
   */
  void enable_write_arrow_schema(bool val);

  /**
   * @brief Sets the maximum row group size, in bytes.
   *
   * @param size_bytes Maximum row group size, in bytes to set
   */
  void set_row_group_size_bytes(size_t size_bytes);

  /**
   * @brief Sets the maximum row group size, in rows.
   *
   * @param size_rows Maximum row group size, in rows to set
   */
  void set_row_group_size_rows(size_type size_rows);

  /**
   * @brief Sets the maximum uncompressed page size, in bytes.
   *
   * @param size_bytes Maximum uncompressed page size, in bytes to set
   */
  void set_max_page_size_bytes(size_t size_bytes);

  /**
   * @brief Sets the maximum page size, in rows.
   *
   * @param size_rows Maximum page size, in rows to set
   */
  void set_max_page_size_rows(size_type size_rows);

  /**
   * @brief Sets the maximum length of min or max values in column index, in bytes.
   *
   * @param size_bytes length min/max will be truncated to
   */
  void set_column_index_truncate_length(int32_t size_bytes);

  /**
   * @brief Sets the policy for dictionary use.
   *
   * @param policy Policy for dictionary use
   */
  void set_dictionary_policy(dictionary_policy policy);

  /**
   * @brief Sets the maximum dictionary size, in bytes.
   *
   * @param size_bytes Maximum dictionary size, in bytes
   */
  void set_max_dictionary_size(size_t size_bytes);

  /**
   * @brief Sets the maximum page fragment size, in rows.
   *
   * @param size_rows Maximum page fragment size, in rows.
   */
  void set_max_page_fragment_size(size_type size_rows);

  /**
   * @brief Sets the pointer to the output compression statistics.
   *
   * @param comp_stats Pointer to compression statistics to be updated after writing
   */
  void set_compression_statistics(std::shared_ptr<writer_compression_statistics> comp_stats);

  /**
   * @brief Sets preference for V2 page headers. Write V2 page headers if set to `true`.
   *
   * @param val Boolean value to enable/disable writing of V2 page headers.
   */
  void enable_write_v2_headers(bool val);

  /**
   * @brief Sets preference for per-page compression decision in V2 pages.
   *
   * When enabled, each V2 data page can independently decide whether to store
   * compressed or uncompressed based on compression effectiveness. This may
   * produce files that are not readable by all Parquet implementations (e.g., PyArrow).
   *
   * @param val Boolean value to enable/disable per-page compression decisions.
   */
  void enable_page_level_compression(bool val);

  /**
   * @brief Sets sorting columns.
   *
   * @param sorting_columns Column sort order metadata
   */
  void set_sorting_columns(std::vector<sorting_column> sorting_columns);
};

/**
 * @brief Base class for Parquet options builders.
 */
template <class BuilderT, class OptionsT>
class parquet_writer_options_builder_base {
  OptionsT _options;

 protected:
  /**
   * @brief Return reference to the options object being built
   *
   * @return the options object
   */
  inline OptionsT& get_options() { return _options; }

  /**
   * @brief Constructor from options.
   *
   * @param options Options object to build
   */
  explicit parquet_writer_options_builder_base(OptionsT options);

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_writer_options_builder_base() = default;

  /**
   * @brief Sets metadata.
   *
   * @param metadata Associated metadata
   * @return this for chaining
   */
  BuilderT& metadata(table_input_metadata metadata);

  /**
   * @brief Sets Key-Value footer metadata.
   *
   * @param metadata Key-Value footer metadata
   * @return this for chaining
   */
  BuilderT& key_value_metadata(std::vector<std::map<std::string, std::string>> metadata);

  /**
   * @brief Sets the level of statistics.
   *
   * @param sf Level of statistics requested in the output file
   * @return this for chaining
   */
  BuilderT& stats_level(statistics_freq sf);

  /**
   * @brief Sets compression type.
   *
   * @param compression The compression type to use
   * @return this for chaining
   */
  BuilderT& compression(compression_type compression);

  /**
   * @brief Sets the maximum row group size, in bytes.
   *
   * @param val maximum row group size
   * @return this for chaining
   */
  BuilderT& row_group_size_bytes(size_t val);

  /**
   * @brief Sets the maximum number of rows in output row groups.
   *
   * @param val maximum number or rows
   * @return this for chaining
   */
  BuilderT& row_group_size_rows(size_type val);

  /**
   * @brief Sets the maximum uncompressed page size, in bytes.
   *
   * Serves as a hint to the writer, and can be exceeded under certain circumstances.
   * Cannot be larger than the row group size in bytes, and will be adjusted to
   * match if it is.
   *
   * @param val maximum page size
   * @return this for chaining
   */
  BuilderT& max_page_size_bytes(size_t val);

  /**
   * @brief Sets the maximum page size, in rows. Counts only top-level rows, ignoring any nesting.
   * Cannot be larger than the row group size in rows, and will be adjusted to match if it is.
   *
   * @param val maximum rows per page
   * @return this for chaining
   */
  BuilderT& max_page_size_rows(size_type val);

  /**
   * @brief Sets the desired maximum size in bytes for min and max values in the column index.
   *
   * Values exceeding this limit will be truncated, but modified such that they will still
   * be valid lower and upper bounds. This only applies to variable length types, such as string.
   * Maximum values will not be truncated if there is no suitable truncation that results in
   * a valid upper bound.
   *
   * Default value is 64.
   *
   * @param val length min/max will be truncated to, with 0 indicating no truncation
   * @return this for chaining
   */
  BuilderT& column_index_truncate_length(int32_t val);

  /**
   * @brief Sets the policy for dictionary use.
   *
   * Certain compression algorithms (e.g Zstandard) have limits on how large of a buffer can
   * be compressed. In some circumstances, the dictionary can grow beyond this limit, which
   * will prevent the column from being compressed. This setting controls how the writer
   * should act in these circumstances. A setting of dictionary_policy::ADAPTIVE will disable
   * dictionary encoding for columns where the dictionary exceeds the limit. A setting of
   * dictionary_policy::NEVER will disable the use of dictionary encoding globally. A setting of
   * dictionary_policy::ALWAYS will allow the use of dictionary encoding even if it will result in
   * the disabling of compression for columns that would otherwise be compressed.
   *
   * The default value is dictionary_policy::ADAPTIVE.
   *
   * @param val policy for dictionary use
   * @return this for chaining
   */
  BuilderT& dictionary_policy(enum dictionary_policy val);

  /**
   * @brief Sets the maximum dictionary size, in bytes.
   *
   * Disables dictionary encoding for any column chunk where the dictionary will
   * exceed this limit.  Only used when the dictionary_policy is set to 'ADAPTIVE'.
   *
   * Default value is 1048576 (1MiB).
   *
   * @param val maximum dictionary size
   * @return this for chaining
   */
  BuilderT& max_dictionary_size(size_t val);

  /**
   * @brief Sets the maximum page fragment size, in rows.
   *
   * Files with nested schemas or very long strings may need a page fragment size
   * smaller than the default value of 5000 to ensure a single fragment will not
   * exceed the desired maximum page size in bytes.
   *
   * @param val maximum page fragment size
   * @return this for chaining
   */
  BuilderT& max_page_fragment_size(size_type val);

  /**
   * @brief Sets the pointer to the output compression statistics.
   *
   * @param comp_stats Pointer to compression statistics to be filled once writer is done
   * @return this for chaining
   */
  BuilderT& compression_statistics(
    std::shared_ptr<writer_compression_statistics> const& comp_stats);

  /**
   * @brief Sets whether int96 timestamps are written or not.
   *
   * @param enabled Boolean value to enable/disable int96 timestamps
   * @return this for chaining
   */
  BuilderT& int96_timestamps(bool enabled);

  /**
   * @brief Set to true if timestamps are to be written as UTC.
   *
   * @param enabled Boolean value to enable/disable writing of timestamps as UTC.
   * @return this for chaining
   */
  BuilderT& utc_timestamps(bool enabled);

  /**
   * @brief Set to true if arrow schema is to be written
   *
   * @param enabled Boolean value to enable/disable writing of arrow schema
   * @return this for chaining
   */
  BuilderT& write_arrow_schema(bool enabled);

  /**
   * @brief Set to true if V2 page headers are to be written.
   *
   * @param enabled Boolean value to enable/disable writing of V2 page headers.
   * @return this for chaining
   */
  BuilderT& write_v2_headers(bool enabled);

  /**
   * @brief Set to true to enable per-page compression decisions for V2 pages.
   *
   * When enabled, each V2 data page can independently decide whether to store
   * compressed or uncompressed based on compression effectiveness. This may
   * produce files that are not readable by all Parquet implementations.
   *
   * @param enabled Boolean value to enable/disable per-page compression decisions.
   * @return this for chaining
   */
  BuilderT& page_level_compression(bool enabled);

  /**
   * @brief Sets column sorting metadata.
   *
   * @param sorting_columns Column sort order metadata
   * @return this for chaining
   */
  BuilderT& sorting_columns(std::vector<sorting_column> sorting_columns);

  /**
   * @brief move options member once it's built.
   */
  operator OptionsT&&();

  /**
   * @brief move options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `parquet_writer_options` object's r-value reference
   */
  OptionsT&& build();
};

class parquet_writer_options_builder;

/**
 * @brief Settings for `write_parquet()`.
 */
class parquet_writer_options : public parquet_writer_options_base {
  // Sets of columns to output
  table_view _table;
  // Partitions described as {start_row, num_rows} pairs
  std::vector<partition_info> _partitions;
  // Column chunks file paths to be set in the raw output metadata. One per output file
  std::vector<std::string> _column_chunks_file_paths;

  friend parquet_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit parquet_writer_options(sink_info const& sink, table_view table);

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  parquet_writer_options() = default;

  /**
   * @brief Create builder to create `parquet_writer_options`.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build parquet_writer_options
   */
  static parquet_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Create builder to create `parquet_writer_options`.
   *
   * @return parquet_writer_options_builder
   */
  static parquet_writer_options_builder builder();

  /**
   * @brief Returns table_view.
   *
   * @return Table view
   */
  [[nodiscard]] table_view get_table() const { return _table; }

  /**
   * @brief Returns partitions.
   *
   * @return Partitions
   */
  [[nodiscard]] std::vector<partition_info> const& get_partitions() const { return _partitions; }

  /**
   * @brief Returns Column chunks file paths to be set in the raw output metadata.
   *
   * @return Column chunks file paths to be set in the raw output metadata
   */
  [[nodiscard]] std::vector<std::string> const& get_column_chunks_file_paths() const
  {
    return _column_chunks_file_paths;
  }

  /**
   * @brief Sets partitions.
   *
   * @param partitions Partitions of input table in {start_row, num_rows} pairs. If specified, must
   * be same size as number of sinks in sink_info
   */
  void set_partitions(std::vector<partition_info> partitions);

  /**
   * @brief Sets column chunks file path to be set in the raw output metadata.
   *
   * @param file_paths Vector of Strings which indicates file path. Must be same size as number of
   * data sinks in sink info
   */
  void set_column_chunks_file_paths(std::vector<std::string> file_paths);
};

/**
 * @brief Class to build `parquet_writer_options`.
 */
class parquet_writer_options_builder
  : public parquet_writer_options_builder_base<parquet_writer_options_builder,
                                               parquet_writer_options> {
 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit parquet_writer_options_builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Sets partitions in parquet_writer_options.
   *
   * @param partitions Partitions of input table in {start_row, num_rows} pairs. If specified, must
   * be same size as number of sinks in sink_info
   * @return this for chaining
   */
  parquet_writer_options_builder& partitions(std::vector<partition_info> partitions);

  /**
   * @brief Sets column chunks file path to be set in the raw output metadata.
   *
   * @param file_paths Vector of Strings which indicates file path. Must be same size as number of
   * data sinks
   * @return this for chaining
   */
  parquet_writer_options_builder& column_chunks_file_paths(std::vector<std::string> file_paths);
};

/**
 * @brief Writes a set of columns to parquet format.
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  auto destination = cudf::io::sink_info("dataset.parquet");
 *  auto options     = cudf::io::parquet_writer_options::builder(destination, table->view());
 *  cudf::io::write_parquet(options);
 * @endcode
 *
 * @note If an exception is thrown during encoding or compression, no data is written to the sink.
 *
 * @param options Settings for controlling writing behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A parquet-compatible blob that contains the file metadata (parquet FileMetadata thrift
 * message).
 */

std::unique_ptr<std::vector<uint8_t>> write_parquet(
  parquet_writer_options const& options, rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Merges multiple raw metadata blobs that were previously created by write_parquet
 * into a single metadata blob.
 *
 * @ingroup io_writers
 *
 * @param[in] metadata_list List of input file metadata
 * @return A parquet-compatible blob that contains the data for all row groups in the list
 */
std::unique_ptr<std::vector<uint8_t>> merge_row_group_metadata(
  std::vector<std::unique_ptr<std::vector<uint8_t>>> const& metadata_list);

class chunked_parquet_writer_options_builder;

/**
 * @brief Settings for `chunked_parquet_writer`.
 */
class chunked_parquet_writer_options : public parquet_writer_options_base {
  /**
   * @brief Constructor from sink.
   *
   * @param sink Sink used for writer output
   */
  explicit chunked_parquet_writer_options(sink_info sink);

  friend chunked_parquet_writer_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  chunked_parquet_writer_options() = default;

  /**
   * @brief creates builder to build chunked_parquet_writer_options.
   *
   * @param sink sink to use for writer output
   *
   * @return Builder to build `chunked_parquet_writer_options`
   */
  static chunked_parquet_writer_options_builder builder(sink_info const& sink);
};

/**
 * @brief Class to build `chunked_parquet_writer_options`.
 */
class chunked_parquet_writer_options_builder
  : public parquet_writer_options_builder_base<chunked_parquet_writer_options_builder,
                                               chunked_parquet_writer_options> {
 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  chunked_parquet_writer_options_builder() = default;

  /**
   * @brief Constructor from sink.
   *
   * @param sink The sink used for writer output
   */
  chunked_parquet_writer_options_builder(sink_info const& sink);
};

/**
 * @brief chunked parquet writer class to handle options and write tables in chunks.
 *
 * The intent of the chunked_parquet_writer is to allow writing of an
 * arbitrarily large / arbitrary number of rows to a parquet file in multiple passes.
 *
 * The following code snippet demonstrates how to write a single parquet file containing
 * one logical table by writing a series of individual cudf::tables.
 *
 * @code
 *  auto destination = cudf::io::sink_info("dataset.parquet");
 *  auto options = cudf::io::chunked_parquet_writer_options::builder(destination, table->view());
 *  auto writer  = cudf::io::chunked_parquet_writer(options);
 *
 *  writer.write(table0)
 *  writer.write(table1)
 *  writer.close()
 *  @endcode
 */
class chunked_parquet_writer {
 public:
  /**
   * @brief Default constructor, this should never be used.
   *        This is added just to satisfy cython.
   *        This is added to not leak detail API
   */
  chunked_parquet_writer();

  /**
   * @brief Constructor with chunked writer options
   *
   * @param[in] options options used to write table
   * @param[in] stream CUDA stream used for device memory operations and kernel launches
   */
  chunked_parquet_writer(chunked_parquet_writer_options const& options,
                         rmm::cuda_stream_view stream = cudf::get_default_stream());
  /**
   * @brief Default destructor.
   *        This is added to not leak detail API
   */
  ~chunked_parquet_writer();

  /**
   * @brief Writes table to output.
   *
   * @note If an exception is thrown during encoding or compression, the data from the failing call
   * is not written to the sink. Data from previous successful calls is unaffected.
   *
   * @param[in] table Table that needs to be written
   * @param[in] partitions Optional partitions to divide the table into. If specified, must be same
   * size as number of sinks.
   *
   * @throws cudf::logic_error If the number of partitions is not the same as number of sinks
   * @throws rmm::bad_alloc if there is insufficient space for temporary buffers
   * @return returns reference of the class object
   */
  chunked_parquet_writer& write(table_view const& table,
                                std::vector<partition_info> const& partitions = {});

  /**
   * @brief Finishes the chunked/streamed write process.
   *
   * @param[in] column_chunks_file_path Column chunks file path to be set in the raw output
   * metadata
   * @return A parquet-compatible blob that contains the file header and footer metadata. If
   * `column_chunks_file_path` is non-empty, the output metadata blob will also have row group file
   * paths set.
   */
  std::unique_ptr<std::vector<uint8_t>> close(
    std::vector<std::string> const& column_chunks_file_path = {});

  /// Unique pointer to impl writer class
  std::unique_ptr<parquet::detail::writer> writer;
};

/** @} */  // end of group

}  // namespace io
}  // namespace CUDF_EXPORT cudf
