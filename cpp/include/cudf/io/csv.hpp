/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/detail/utils.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

/**
 * @file
 * @brief APIs for reading and writing CSV files.
 */

namespace CUDF_EXPORT cudf {
namespace io {

/**
 * @addtogroup io_readers
 * @{
 */

/**
 *@brief Builder to build options for `read_csv()`.
 */
class csv_reader_options_builder;

/**
 * @brief Settings to use for `read_csv()`.
 */

class csv_reader_options {
  source_info _source;

  // Read settings

  // Specify the compression format of the source or infer from file extension
  compression_type _compression = compression_type::AUTO;
  // Bytes to skip from the source start
  std::size_t _byte_range_offset = 0;
  // Bytes to read; always reads complete rows
  std::size_t _byte_range_size = 0;
  // Names of all the columns; if empty then names are auto-generated
  std::vector<std::string> _names;
  // If there is no header or names, prepend this to the column ID as the name
  std::string _prefix;
  // Whether to rename duplicate column names
  bool _mangle_dupe_cols = true;

  // Filter settings

  // Names of columns to read; empty is all columns
  std::vector<std::string> _use_cols_names;
  // Indexes of columns to read; empty is all columns
  std::vector<int> _use_cols_indexes;
  // Rows to read; -1 is all
  size_type _nrows = -1;
  // Rows to skip from the start
  size_type _skiprows = 0;
  // Rows to skip from the end
  size_type _skipfooter = 0;
  // Header row index
  size_type _header = 0;

  // Parsing settings

  // Line terminator
  char _lineterminator = '\n';
  // Field delimiter
  char _delimiter = ',';
  // Numeric data thousands separator; cannot match delimiter
  char _thousands = '\0';
  // Decimal point character; cannot match delimiter
  char _decimal = '.';
  // Comment line start character
  char _comment                = '\0';
  bool _windowslinetermination = false;
  // Treat whitespace as field delimiter; overrides character delimiter
  bool _delim_whitespace = false;
  // Skip whitespace after the delimiter
  bool _skipinitialspace = false;
  // Ignore empty lines or parse line values as invalid
  bool _skip_blank_lines = true;
  // Treatment of quoting behavior
  quote_style _quoting = quote_style::MINIMAL;
  // Quoting character (if `quoting` is true)
  char _quotechar = '"';
  // Whether a quote inside a value is double-quoted
  bool _doublequote = true;
  // Whether to detect quotes surrounded by spaces e.g. `   "data"   `. This flag has no effect when
  // _doublequote is true
  bool _detect_whitespace_around_quotes = false;
  // Names of columns to read as datetime
  std::vector<std::string> _parse_dates_names;
  // Indexes of columns to read as datetime
  std::vector<int> _parse_dates_indexes;
  // Names of columns to parse as hexadecimal
  std::vector<std::string> _parse_hex_names;
  // Indexes of columns to parse as hexadecimal
  std::vector<int> _parse_hex_indexes;

  // Conversion settings

  // Per-column types; disables type inference on those columns
  std::variant<std::vector<data_type>, std::map<std::string, data_type>> _dtypes;
  // Additional values to recognize as boolean true values
  std::vector<std::string> _true_values{"True", "TRUE", "true"};
  // Additional values to recognize as boolean false values
  std::vector<std::string> _false_values{"False", "FALSE", "false"};
  // Additional values to recognize as null values
  std::vector<std::string> _na_values;
  // Whether to keep the built-in default NA values
  bool _keep_default_na = true;
  // Whether to disable null filter; disabling can improve performance
  bool _na_filter = true;
  // Whether to parse dates as DD/MM versus MM/DD
  bool _dayfirst = false;
  // Cast timestamp columns to a specific type
  data_type _timestamp_type{type_id::EMPTY};

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read csv file
   */
  explicit csv_reader_options(source_info src) : _source{std::move(src)} {}

  friend csv_reader_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  csv_reader_options() = default;

  /**
   * @brief Creates a `csv_reader_options_builder` which will build `csv_reader_options`.
   *
   * @param src Source information to read csv file
   * @return Builder to build reader options
   */
  static csv_reader_options_builder builder(source_info src);

  /**
   * @brief Returns source info.
   *
   * @return Source info
   */
  [[nodiscard]] source_info const& get_source() const { return _source; }

  /**
   * @brief Returns compression format of the source.
   *
   * @return Compression format of the source
   */
  [[nodiscard]] compression_type get_compression() const { return _compression; }

  /**
   * @brief Returns number of bytes to skip from source start.
   *
   * @return Number of bytes to skip from source start
   */
  [[nodiscard]] std::size_t get_byte_range_offset() const { return _byte_range_offset; }

  /**
   * @brief Returns number of bytes to read.
   *
   * @return Number of bytes to read
   */
  [[nodiscard]] std::size_t get_byte_range_size() const { return _byte_range_size; }

  /**
   * @brief Returns number of bytes to read with padding.
   *
   * @return Number of bytes to read with padding
   */
  [[nodiscard]] std::size_t get_byte_range_size_with_padding() const
  {
    if (_byte_range_size == 0) {
      return 0;
    } else {
      return _byte_range_size + get_byte_range_padding();
    }
  }

  /**
   * @brief Returns number of bytes to pad when reading.
   *
   * @return Number of bytes to pad when reading
   */
  [[nodiscard]] std::size_t get_byte_range_padding() const
  {
    auto const num_names   = _names.size();
    auto const num_dtypes  = std::visit([](auto const& dtypes) { return dtypes.size(); }, _dtypes);
    auto const num_columns = std::max(num_dtypes, num_names);

    auto const max_row_bytes = 16 * 1024;  // 16KB
    auto const column_bytes  = 64;
    auto const base_padding  = 1024;  // 1KB

    if (num_columns == 0) {
      // Use flat size if the number of columns is not known
      return max_row_bytes;
    }

    // Expand the size based on the number of columns, if available
    return base_padding + static_cast<size_t>(num_columns) * column_bytes;
  }

  /**
   * @brief Returns names of the columns.
   *
   * @return Names of the columns
   */
  [[nodiscard]] std::vector<std::string> const& get_names() const { return _names; }

  /**
   * @brief Returns prefix to be used for column ID.
   *
   * @return Prefix to be used for column ID
   */
  [[nodiscard]] std::string get_prefix() const { return _prefix; }

  /**
   * @brief Whether to rename duplicate column names.
   *
   * @return `true` if  duplicate column names are renamed
   */
  [[nodiscard]] bool is_enabled_mangle_dupe_cols() const { return _mangle_dupe_cols; }

  /**
   * @brief Returns names of the columns to be read.
   *
   * @return Names of the columns to be read
   */
  [[nodiscard]] std::vector<std::string> const& get_use_cols_names() const
  {
    return _use_cols_names;
  }

  /**
   * @brief Returns indexes of columns to read.
   *
   * @return Indexes of columns to read
   */
  [[nodiscard]] std::vector<int> const& get_use_cols_indexes() const { return _use_cols_indexes; }

  /**
   * @brief Returns number of rows to read.
   *
   * @return Number of rows to read
   */
  [[nodiscard]] size_type get_nrows() const { return _nrows; }

  /**
   * @brief Returns number of rows to skip from start.
   *
   * @return Number of rows to skip from start
   */
  [[nodiscard]] size_type get_skiprows() const { return _skiprows; }

  /**
   * @brief Returns number of rows to skip from end.
   *
   * @return Number of rows to skip from end
   */
  [[nodiscard]] size_type get_skipfooter() const { return _skipfooter; }

  /**
   * @brief Returns header row index.
   *
   * @return Header row index
   */
  [[nodiscard]] size_type get_header() const { return _header; }

  /**
   * @brief Returns line terminator.
   *
   * @return Line terminator
   */
  [[nodiscard]] char get_lineterminator() const { return _lineterminator; }

  /**
   * @brief Returns field delimiter.
   *
   * @return Field delimiter
   */
  [[nodiscard]] char get_delimiter() const { return _delimiter; }

  /**
   * @brief Returns numeric data thousands separator.
   *
   * @return Numeric data thousands separator
   */
  [[nodiscard]] char get_thousands() const { return _thousands; }

  /**
   * @brief Returns decimal point character.
   *
   * @return Decimal point character
   */
  [[nodiscard]] char get_decimal() const { return _decimal; }

  /**
   * @brief Returns comment line start character.
   *
   * @return Comment line start character
   */
  [[nodiscard]] char get_comment() const { return _comment; }

  /**
   * @brief Whether to treat `\r\n` as line terminator.
   *
   * @return `true` if `\r\n` is treated as line terminator
   */
  [[nodiscard]] bool is_enabled_windowslinetermination() const { return _windowslinetermination; }

  /**
   * @brief Whether to treat whitespace as field delimiter.
   *
   * @return `true` if whitespace is treated as field delimiter
   */
  [[nodiscard]] bool is_enabled_delim_whitespace() const { return _delim_whitespace; }

  /**
   * @brief Whether to skip whitespace after the delimiter.
   *
   * @return `true` if whitespace is skipped after the delimiter
   */
  [[nodiscard]] bool is_enabled_skipinitialspace() const { return _skipinitialspace; }

  /**
   * @brief Whether to ignore empty lines or parse line values as invalid.
   *
   * @return `true` if empty lines or parse line values are ignored as invalid
   */
  [[nodiscard]] bool is_enabled_skip_blank_lines() const { return _skip_blank_lines; }

  /**
   * @brief Returns quoting style.
   *
   * @return Quoting style
   */
  [[nodiscard]] quote_style get_quoting() const { return _quoting; }

  /**
   * @brief Returns quoting character.
   *
   * @return Quoting character
   */
  [[nodiscard]] char get_quotechar() const { return _quotechar; }

  /**
   * @brief Whether a quote inside a value is double-quoted.
   *
   * @return `true` if a quote inside a value is double-quoted
   */
  [[nodiscard]] bool is_enabled_doublequote() const { return _doublequote; }

  /**
   * @brief Whether to detect quotes surrounded by spaces e.g. `   "data"   `. This flag has no
   * effect when _doublequote is true
   *
   * @return `true` if detect_whitespace_around_quotes is enabled
   */
  [[nodiscard]] bool is_enabled_detect_whitespace_around_quotes() const
  {
    return _detect_whitespace_around_quotes;
  }

  /**
   * @brief Returns names of columns to read as datetime.
   *
   * @return Names of columns to read as datetime
   */
  [[nodiscard]] std::vector<std::string> const& get_parse_dates_names() const
  {
    return _parse_dates_names;
  }

  /**
   * @brief Returns indexes of columns to read as datetime.
   *
   * @return Indexes of columns to read as datetime
   */
  [[nodiscard]] std::vector<int> const& get_parse_dates_indexes() const
  {
    return _parse_dates_indexes;
  }

  /**
   * @brief Returns names of columns to read as hexadecimal.
   *
   * @return Names of columns to read as hexadecimal
   */
  [[nodiscard]] std::vector<std::string> const& get_parse_hex_names() const
  {
    return _parse_hex_names;
  }

  /**
   * @brief Returns indexes of columns to read as hexadecimal.
   *
   * @return Indexes of columns to read as hexadecimal
   */
  [[nodiscard]] std::vector<int> const& get_parse_hex_indexes() const { return _parse_hex_indexes; }

  /**
   * @brief Returns per-column types.
   *
   * @return Per-column types
   */
  [[nodiscard]] std::variant<std::vector<data_type>, std::map<std::string, data_type>> const&
  get_dtypes() const
  {
    return _dtypes;
  }

  /**
   * @brief Returns additional values to recognize as boolean true values.
   *
   * @return Additional values to recognize as boolean true values
   */
  [[nodiscard]] std::vector<std::string> const& get_true_values() const { return _true_values; }

  /**
   * @brief Returns additional values to recognize as boolean false values.
   *
   * @return Additional values to recognize as boolean false values
   */
  [[nodiscard]] std::vector<std::string> const& get_false_values() const { return _false_values; }

  /**
   * @brief Returns additional values to recognize as null values.
   *
   * @return Additional values to recognize as null values
   */
  [[nodiscard]] std::vector<std::string> const& get_na_values() const { return _na_values; }

  /**
   * @brief Whether to keep the built-in default NA values.
   *
   * @return `true` if the built-in default NA values are kept
   */
  [[nodiscard]] bool is_enabled_keep_default_na() const { return _keep_default_na; }

  /**
   * @brief Whether to disable null filter.
   *
   * @return `true` if null filter is enabled
   */
  [[nodiscard]] bool is_enabled_na_filter() const { return _na_filter; }

  /**
   * @brief Whether to parse dates as DD/MM versus MM/DD.
   *
   * @return True if dates are parsed as DD/MM, false if MM/DD
   */
  [[nodiscard]] bool is_enabled_dayfirst() const { return _dayfirst; }

  /**
   * @brief Returns timestamp_type to which all timestamp columns will be cast.
   *
   * @return timestamp_type to which all timestamp columns will be cast
   */
  [[nodiscard]] data_type get_timestamp_type() const { return _timestamp_type; }

  /**
   * @brief Sets source info.
   *
   * @param src The source info.
   */
  void set_source(source_info src) { _source = std::move(src); }

  /**
   * @brief Sets compression format of the source.
   *
   * @param comp Compression type
   */
  void set_compression(compression_type comp) { _compression = comp; }

  /**
   * @brief Sets number of bytes to skip from source start.
   *
   * @param offset Number of bytes of offset
   */
  void set_byte_range_offset(std::size_t offset)
  {
    if ((offset != 0) and ((_skiprows != 0) or (_skipfooter != 0) or (_nrows != -1))) {
      CUDF_FAIL(
        "When there is valid value in skiprows or skipfooter or nrows, offset can't have non-zero "
        "value");
    }
    _byte_range_offset = offset;
  }

  /**
   * @brief Sets number of bytes to read.
   *
   * @param size Number of bytes to read
   */
  void set_byte_range_size(std::size_t size)
  {
    if ((size != 0) and ((_skiprows != 0) or (_skipfooter != 0) or (_nrows != -1))) {
      CUDF_FAIL(
        "If the value of any of skiprows, skipfooter or nrows is valid, range size cannot be "
        "non-zero.");
    }
    _byte_range_size = size;
  }

  /**
   * @brief Sets names of the column.
   *
   * @param col_names Vector of column names
   */
  void set_names(std::vector<std::string> col_names) { _names = std::move(col_names); }

  /**
   * @brief Sets prefix to be used for column ID.
   *
   * @param pfx String used as prefix in for each column name
   */
  void set_prefix(std::string pfx) { _prefix = std::move(pfx); }

  /**
   * @brief Sets whether to rename duplicate column names.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_mangle_dupe_cols(bool val) { _mangle_dupe_cols = val; }

  /**
   * @brief Sets names of the columns to be read.
   *
   * @param col_names Vector of column names that are needed
   */
  void set_use_cols_names(std::vector<std::string> col_names)
  {
    _use_cols_names = std::move(col_names);
  }

  /**
   * @brief Sets indexes of columns to read.
   *
   * @param col_indices Vector of column indices that are needed
   */
  void set_use_cols_indexes(std::vector<int> col_indices)
  {
    _use_cols_indexes = std::move(col_indices);
  }

  /**
   * @brief Sets number of rows to read.
   *
   * @param nrows Number of rows to read
   */
  void set_nrows(size_type nrows)
  {
    CUDF_EXPECTS((nrows == 0) or (_skipfooter == 0), "Cannot use both `nrows` and `skipfooter`");
    if ((nrows != -1) and ((_byte_range_offset != 0) or (_byte_range_size != 0))) {
      CUDF_FAIL(
        "nrows can't be a non negative value if range offset and/or range size has been set");
    }

    _nrows = nrows;
  }

  /**
   * @brief Sets number of rows to skip from start.
   *
   * @param skiprows Number of rows to skip
   */
  void set_skiprows(size_type skiprows)
  {
    if ((skiprows != 0) and ((_byte_range_offset != 0) or (_byte_range_size != 0))) {
      CUDF_FAIL("skiprows must be zero if range offset or range size has been set",
                std::invalid_argument);
    }
    _skiprows = skiprows;
  }

  /**
   * @brief Sets number of rows to skip from end.
   *
   * @param skipfooter Number of rows to skip
   */
  void set_skipfooter(size_type skipfooter)
  {
    CUDF_EXPECTS((skipfooter == 0) or (_nrows == -1),
                 "Cannot use both `nrows` and `skipfooter`",
                 std::invalid_argument);
    if ((skipfooter != 0) and ((_byte_range_offset != 0) or (_byte_range_size != 0))) {
      CUDF_FAIL("skipfooter must be zero if range offset or range size has been set",
                std::invalid_argument);
    }

    _skipfooter = skipfooter;
  }

  /**
   * @brief Sets header row index.
   *
   * @param hdr Index where header row is located
   */
  void set_header(size_type hdr) { _header = hdr; }

  /**
   * @brief Sets line terminator
   *
   * @param term A character to indicate line termination
   */
  void set_lineterminator(char term) { _lineterminator = term; }

  /**
   * @brief Sets field delimiter.
   *
   * @param delim A character to indicate delimiter
   */
  void set_delimiter(char delim) { _delimiter = delim; }

  /**
   * @brief Sets numeric data thousands separator.
   *
   * @param val A character that separates thousands
   */
  void set_thousands(char val) { _thousands = val; }

  /**
   * @brief Sets decimal point character.
   *
   * @param val A character that indicates decimal values
   */
  void set_decimal(char val) { _decimal = val; }

  /**
   * @brief Sets comment line start character.
   *
   * @param val A character that indicates comment
   */
  void set_comment(char val) { _comment = val; }

  /**
   * @brief Sets whether to treat `\r\n` as line terminator.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_windowslinetermination(bool val) { _windowslinetermination = val; }

  /**
   * @brief Sets whether to treat whitespace as field delimiter.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_delim_whitespace(bool val) { _delim_whitespace = val; }

  /**
   * @brief Sets whether to skip whitespace after the delimiter.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_skipinitialspace(bool val) { _skipinitialspace = val; }

  /**
   * @brief Sets whether to ignore empty lines or parse line values as invalid.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_skip_blank_lines(bool val) { _skip_blank_lines = val; }

  /**
   * @brief Sets the expected quoting style used in the input CSV data.
   *
   * The reader accepts all defined quoting styles. `NONE` disables quotation parsing; all other
   * styles enable it.
   *
   * @param quoting Quoting style used
   */
  void set_quoting(quote_style quoting)
  {
    CUDF_EXPECTS(quoting == quote_style::MINIMAL || quoting == quote_style::ALL ||
                   quoting == quote_style::NONNUMERIC || quoting == quote_style::NONE,
                 "Unsupported quoting style.");
    _quoting = quoting;
  }

  /**
   * @brief Sets quoting character.
   *
   * @param ch A character to indicate quoting
   */
  void set_quotechar(char ch) { _quotechar = ch; }

  /**
   * @brief Sets a quote inside a value is double-quoted.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_doublequote(bool val) { _doublequote = val; }

  /**
   * @brief Sets whether to detect quotes surrounded by spaces e.g. `   "data"   `. This flag has no
   * effect when _doublequote is true
   *
   * @param val Boolean value to enable/disable
   */
  void enable_detect_whitespace_around_quotes(bool val) { _detect_whitespace_around_quotes = val; }

  /**
   * @brief Sets names of columns to read as datetime.
   *
   * @param col_names Vector of column names to infer as datetime
   */
  void set_parse_dates(std::vector<std::string> col_names)
  {
    _parse_dates_names = std::move(col_names);
  }

  /**
   * @brief Sets indexes of columns to read as datetime.
   *
   * @param col_indices Vector of column indices to infer as datetime
   */
  void set_parse_dates(std::vector<int> col_indices)
  {
    _parse_dates_indexes = std::move(col_indices);
  }

  /**
   * @brief Sets names of columns to parse as hexadecimal
   *
   * @param col_names Vector of column names to parse as hexadecimal
   */
  void set_parse_hex(std::vector<std::string> col_names)
  {
    _parse_hex_names = std::move(col_names);
  }

  /**
   * @brief Sets indexes of columns to parse as hexadecimal
   *
   * @param col_indices Vector of column indices to parse as hexadecimal
   */
  void set_parse_hex(std::vector<int> col_indices) { _parse_hex_indexes = std::move(col_indices); }

  /**
   * @brief Sets per-column types
   *
   * @param types Column name -> data type map specifying the columns' target data types
   */
  void set_dtypes(std::map<std::string, data_type> types) { _dtypes = std::move(types); }

  /**
   * @brief Sets per-column types
   *
   * @param types Vector specifying the columns' target data types
   */
  void set_dtypes(std::vector<data_type> types) { _dtypes = std::move(types); }

  /**
   * @brief Sets additional values to recognize as boolean true values.
   *
   * @param vals Vector of values to be considered to be `true`
   */
  void set_true_values(std::vector<std::string> vals)
  {
    _true_values.insert(_true_values.end(), vals.begin(), vals.end());
  }

  /**
   * @brief Sets additional values to recognize as boolean false values.
   *
   * @param vals Vector of values to be considered to be `false`
   */
  void set_false_values(std::vector<std::string> vals)
  {
    _false_values.insert(_false_values.end(), vals.begin(), vals.end());
  }

  /**
   * @brief Sets additional values to recognize as null values.
   *
   * @param vals Vector of values to be considered to be null
   */
  void set_na_values(std::vector<std::string> vals)
  {
    CUDF_EXPECTS(vals.empty() or _na_filter, "Can't set na_values when na_filtering is disabled");

    _na_values = std::move(vals);
  }

  /**
   * @brief Sets whether to keep the built-in default NA values.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_keep_default_na(bool val) { _keep_default_na = val; }

  /**
   * @brief Sets whether to disable null filter.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_na_filter(bool val)
  {
    if (!val) { _na_values.clear(); }
    _na_filter = val;
  }

  /**
   * @brief Sets whether to parse dates as DD/MM versus MM/DD.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_dayfirst(bool val) { _dayfirst = val; }

  /**
   * @brief Sets timestamp_type to which all timestamp columns will be cast.
   *
   * @param type Dtype to which all timestamp column will be cast
   */
  void set_timestamp_type(data_type type) { _timestamp_type = type; }
};

/**
 * @brief Builder to build options for `read_csv()`.
 *
 */
class csv_reader_options_builder {
  csv_reader_options options;  ///< Options to be built.

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  csv_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read csv file
   */
  csv_reader_options_builder(source_info src) : options{std::move(src)} {}

  /**
   * @brief Sets compression format of the source.
   *
   * @param comp Compression type
   * @return this for chaining
   */
  csv_reader_options_builder& compression(compression_type comp)
  {
    options.set_compression(comp);
    return *this;
  }

  /**
   * @brief Sets number of bytes to skip from source start.
   *
   * @param offset Number of bytes of offset
   * @return this for chaining
   */
  csv_reader_options_builder& byte_range_offset(std::size_t offset)
  {
    options.set_byte_range_offset(offset);
    return *this;
  }

  /**
   * @brief Sets number of bytes to read.
   *
   * @param size Number of bytes to read
   * @return this for chaining
   */
  csv_reader_options_builder& byte_range_size(std::size_t size)
  {
    options.set_byte_range_size(size);
    return *this;
  }

  /**
   * @brief Sets names of the column.
   *
   * @param col_names Vector of column names
   * @return this for chaining
   */
  csv_reader_options_builder& names(std::vector<std::string> col_names)
  {
    options.set_names(std::move(col_names));
    return *this;
  }

  /**
   * @brief Sets prefix to be used for column ID.
   *
   * @param pfx String used as prefix in for each column name
   * @return this for chaining
   */
  csv_reader_options_builder& prefix(std::string pfx)
  {
    options.set_prefix(std::move(pfx));
    return *this;
  }

  /**
   * @brief Sets whether to rename duplicate column names.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_reader_options_builder& mangle_dupe_cols(bool val)
  {
    options.enable_mangle_dupe_cols(val);
    return *this;
  }

  /**
   * @brief Sets names of the columns to be read.
   *
   * @param col_names Vector of column names that are needed
   * @return this for chaining
   */
  csv_reader_options_builder& use_cols_names(std::vector<std::string> col_names)
  {
    options.set_use_cols_names(std::move(col_names));
    return *this;
  }

  /**
   * @brief Sets indexes of columns to read.
   *
   * @param col_indices Vector of column indices that are needed
   * @return this for chaining
   */
  csv_reader_options_builder& use_cols_indexes(std::vector<int> col_indices)
  {
    options.set_use_cols_indexes(std::move(col_indices));
    return *this;
  }

  /**
   * @brief Sets number of rows to read.
   *
   * @param rows Number of rows to read
   * @return this for chaining
   */
  csv_reader_options_builder& nrows(size_type rows)
  {
    options.set_nrows(rows);
    return *this;
  }

  /**
   * @brief Sets number of rows to skip from start.
   *
   * @param skip Number of rows to skip
   * @return this for chaining
   */
  csv_reader_options_builder& skiprows(size_type skip)
  {
    options.set_skiprows(skip);
    return *this;
  }

  /**
   * @brief Sets number of rows to skip from end.
   *
   * @param skip Number of rows to skip
   * @return this for chaining
   */
  csv_reader_options_builder& skipfooter(size_type skip)
  {
    options.set_skipfooter(skip);
    return *this;
  }

  /**
   * @brief Sets header row index.
   *
   * @param hdr Index where header row is located
   * @return this for chaining
   */
  csv_reader_options_builder& header(size_type hdr)
  {
    options.set_header(hdr);
    return *this;
  }

  /**
   * @brief Sets line terminator.
   *
   * @param term A character to indicate line termination
   * @return this for chaining
   */
  csv_reader_options_builder& lineterminator(char term)
  {
    options.set_lineterminator(term);
    return *this;
  }

  /**
   * @brief Sets field delimiter
   *
   * @param delim A character to indicate delimiter
   * @return this for chaining
   */
  csv_reader_options_builder& delimiter(char delim)
  {
    options.set_delimiter(delim);
    return *this;
  }

  /**
   * @brief Sets numeric data thousands separator.
   *
   * @param val A character that separates thousands
   * @return this for chaining
   */
  csv_reader_options_builder& thousands(char val)
  {
    options.set_thousands(val);
    return *this;
  }

  /**
   * @brief Sets decimal point character.
   *
   * @param val A character that indicates decimal values
   * @return this for chaining
   */
  csv_reader_options_builder& decimal(char val)
  {
    options.set_decimal(val);
    return *this;
  }

  /**
   * @brief Sets comment line start character.
   *
   * @param val A character that indicates comment
   * @return this for chaining
   */
  csv_reader_options_builder& comment(char val)
  {
    options.set_comment(val);
    return *this;
  }

  /**
   * @brief Sets whether to treat `\r\n` as line terminator.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_reader_options_builder& windowslinetermination(bool val)
  {
    options.enable_windowslinetermination(val);
    return *this;
  }

  /**
   * @brief Sets whether to treat whitespace as field delimiter.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_reader_options_builder& delim_whitespace(bool val)
  {
    options.enable_delim_whitespace(val);
    return *this;
  }

  /**
   * @brief Sets whether to skip whitespace after the delimiter.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_reader_options_builder& skipinitialspace(bool val)
  {
    options.enable_skipinitialspace(val);
    return *this;
  }

  /**
   * @brief Sets whether to ignore empty lines or parse line values as invalid.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_reader_options_builder& skip_blank_lines(bool val)
  {
    options.enable_skip_blank_lines(val);
    return *this;
  }

  /**
   * @brief Sets quoting style.
   *
   * @param style Quoting style used
   * @return this for chaining
   */
  csv_reader_options_builder& quoting(quote_style style)
  {
    options.set_quoting(style);
    return *this;
  }

  /**
   * @brief Sets quoting character.
   *
   * @param ch A character to indicate quoting
   * @return this for chaining
   */
  csv_reader_options_builder& quotechar(char ch)
  {
    options.set_quotechar(ch);
    return *this;
  }

  /**
   * @brief Sets a quote inside a value is double-quoted.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_reader_options_builder& doublequote(bool val)
  {
    options.enable_doublequote(val);
    return *this;
  }

  /**
   * @brief Sets whether to detect quotes surrounded by spaces e.g. `   "data"   `. This flag has no
   * effect when _doublequote is true
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_reader_options_builder& detect_whitespace_around_quotes(bool val)
  {
    options.enable_detect_whitespace_around_quotes(val);
    return *this;
  }

  /**
   * @brief Sets names of columns to read as datetime.
   *
   * @param col_names Vector of column names to read as datetime
   * @return this for chaining
   */
  csv_reader_options_builder& parse_dates(std::vector<std::string> col_names)
  {
    options.set_parse_dates(std::move(col_names));
    return *this;
  }

  /**
   * @brief Sets indexes of columns to read as datetime.
   *
   * @param col_indices Vector of column indices to read as datetime
   * @return this for chaining
   */
  csv_reader_options_builder& parse_dates(std::vector<int> col_indices)
  {
    options.set_parse_dates(std::move(col_indices));
    return *this;
  }

  /**
   * @brief Sets names of columns to parse as hexadecimal.
   *
   * @param col_names Vector of column names to parse as hexadecimal
   * @return this for chaining
   */
  csv_reader_options_builder& parse_hex(std::vector<std::string> col_names)
  {
    options.set_parse_hex(std::move(col_names));
    return *this;
  }

  /**
   * @brief Sets indexes of columns to parse as hexadecimal.
   *
   * @param col_indices Vector of column indices to parse as hexadecimal
   * @return this for chaining
   */
  csv_reader_options_builder& parse_hex(std::vector<int> col_indices)
  {
    options.set_parse_hex(std::move(col_indices));
    return *this;
  }

  /**
   * @brief Sets per-column types.
   *
   * @param types Column name -> data type map specifying the columns' target data types
   * @return this for chaining
   */
  csv_reader_options_builder& dtypes(std::map<std::string, data_type> types)
  {
    options.set_dtypes(std::move(types));
    return *this;
  }

  /**
   * @brief Sets per-column types.
   *
   * @param types Vector of data types in which the column needs to be read
   * @return this for chaining
   */
  csv_reader_options_builder& dtypes(std::vector<data_type> types)
  {
    options.set_dtypes(std::move(types));
    return *this;
  }

  /**
   * @brief Sets additional values to recognize as boolean true values.
   *
   * @param vals Vector of values to be considered to be `true`
   * @return this for chaining
   */
  csv_reader_options_builder& true_values(std::vector<std::string> vals)
  {
    options.set_true_values(std::move(vals));
    return *this;
  }

  /**
   * @brief Sets additional values to recognize as boolean false values.
   *
   * @param vals Vector of values to be considered to be `false`
   * @return this for chaining
   */
  csv_reader_options_builder& false_values(std::vector<std::string> vals)
  {
    options.set_false_values(std::move(vals));
    return *this;
  }

  /**
   * @brief Sets additional values to recognize as null values.
   *
   * @param vals Vector of values to be considered to be null
   * @return this for chaining
   */
  csv_reader_options_builder& na_values(std::vector<std::string> vals)
  {
    options.set_na_values(std::move(vals));
    return *this;
  }

  /**
   * @brief Sets whether to keep the built-in default NA values.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_reader_options_builder& keep_default_na(bool val)
  {
    options.enable_keep_default_na(val);
    return *this;
  }

  /**
   * @brief Sets whether to disable null filter.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_reader_options_builder& na_filter(bool val)
  {
    options.enable_na_filter(val);
    return *this;
  }

  /**
   * @brief Sets whether to parse dates as DD/MM versus MM/DD.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_reader_options_builder& dayfirst(bool val)
  {
    options.enable_dayfirst(val);
    return *this;
  }

  /**
   * @brief Sets timestamp_type to which all timestamp columns will be cast.
   *
   * @param type Dtype to which all timestamp column will be cast
   * @return this for chaining
   */
  csv_reader_options_builder& timestamp_type(data_type type)
  {
    options.set_timestamp_type(type);
    return *this;
  }

  /**
   * @brief move csv_reader_options member once it's built.
   */
  operator csv_reader_options&&() { return std::move(options); }

  /**
   * @brief move csv_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `csv_reader_options` object's r-value reference
   */
  csv_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads a CSV dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  auto source  = cudf::io::source_info("dataset.csv");
 *  auto options = cudf::io::csv_reader_options::builder(source);
 *  auto result  = cudf::io::read_csv(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_csv(
  csv_reader_options options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
/**
 * @addtogroup io_writers
 * @{
 */

/**
 *@brief Builder to build options for `write_csv()`.
 */
class csv_writer_options_builder;

/**
 * @brief Settings to use for `write_csv()`.
 */
class csv_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Set of columns to output
  table_view _table;
  // string to use for null entries
  std::string _na_rep = "";
  // Indicates whether to write headers to csv
  bool _include_header = true;
  // maximum number of rows to write in each chunk (limits memory use)
  size_type _rows_per_chunk = std::numeric_limits<size_type>::max();
  // character to use for separating lines (default "\n")
  std::string _line_terminator = "\n";
  // character to use for separating column values (default ",")
  char _inter_column_delimiter = ',';
  // string to use for values != 0 in INT8 types (default 'true')
  std::string _true_value = std::string{"true"};
  // string to use for values == 0 in INT8 types (default 'false')
  std::string _false_value = std::string{"false"};
  // Names of all columns; if empty, writer will generate column names
  std::vector<std::string> _names;
  // Quote style. Currently only MINIMAL and NONE are supported.
  quote_style _quoting = quote_style::MINIMAL;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit csv_writer_options(sink_info sink, table_view const& table)
    : _sink(std::move(sink)), _table(table), _rows_per_chunk(table.num_rows())
  {
  }

  friend csv_writer_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit csv_writer_options() = default;

  /**
   * @brief Create builder to create `csv_writer_options`.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build csv_writer_options
   */
  static csv_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Returns sink used for writer output.
   *
   * @return sink used for writer output
   */
  [[nodiscard]] sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns table that would be written to output.
   *
   * @return Table that would be written to output
   */
  [[nodiscard]] table_view const& get_table() const { return _table; }

  /**
   * @brief Returns names of the columns.
   *
   * @return Names of the columns in the output file
   */
  [[nodiscard]] std::vector<std::string> const& get_names() const { return _names; }

  /**
   * @brief Returns string to used for null entries.
   *
   * @return string to used for null entries
   */
  [[nodiscard]] std::string const& get_na_rep() const { return _na_rep; }

  /**
   * @brief Whether to write headers to csv.
   *
   * @return `true` if writing headers to csv
   */
  [[nodiscard]] bool is_enabled_include_header() const { return _include_header; }

  /**
   * @brief Returns maximum number of rows to process for each file write.
   *
   * @return Maximum number of rows to process for each file write
   */
  [[nodiscard]] size_type get_rows_per_chunk() const { return _rows_per_chunk; }

  /**
   * @brief Returns character used for separating lines.
   *
   * @return Character used for separating lines
   */
  [[nodiscard]] std::string const& get_line_terminator() const { return _line_terminator; }

  /**
   * @brief Returns character used for separating column values.
   *
   * @return Character used for separating column values.
   */
  [[nodiscard]] char get_inter_column_delimiter() const { return _inter_column_delimiter; }

  /**
   * @brief Returns string used for values != 0 in INT8 types.
   *
   * @return string used for values != 0 in INT8 types
   */
  [[nodiscard]] std::string const& get_true_value() const { return _true_value; }

  /**
   * @brief Returns string used for values == 0 in INT8 types.
   *
   * @return string used for values == 0 in INT8 types
   */
  [[nodiscard]] std::string const& get_false_value() const { return _false_value; }

  /**
   * @brief Returns the quote style for the writer.
   *
   * Note: Only MINIMAL and NONE are supported.
   *   1. MINIMAL: String columns containing special characters like row-delimiters
   *               field-delimiter/quotes will be quoted.
   *   2. NONE: No quoting is done for any columns.
   *
   * @return quote_style The quote style for the writer
   */
  [[nodiscard]] quote_style get_quoting() const { return _quoting; }

  // Setter
  /**
   * @brief Sets optional associated column names.
   *
   @param names Associated column names
   */
  void set_names(std::vector<std::string> names) { _names = std::move(names); }

  /**
   * @brief Sets string to used for null entries.
   *
   * @param val String to represent null value
   */
  void set_na_rep(std::string val) { _na_rep = std::move(val); }

  /**
   * @brief Enables/Disables headers being written to csv.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_include_header(bool val) { _include_header = val; }

  /**
   * @brief Sets maximum number of rows to process for each file write.
   *
   * @param val Number of rows per chunk
   */
  void set_rows_per_chunk(size_type val) { _rows_per_chunk = val; }

  /**
   * @brief Sets character used for separating lines.
   *
   * @param term Character to represent line termination
   */
  void set_line_terminator(std::string term) { _line_terminator = std::move(term); }

  /**
   * @brief Sets character used for separating column values.
   *
   * @param delim Character to delimit column values
   */
  void set_inter_column_delimiter(char delim) { _inter_column_delimiter = delim; }

  /**
   * @brief Sets string used for values != 0 in INT8 types.
   *
   * @param val String to represent values != 0 in INT8 types
   */
  void set_true_value(std::string val) { _true_value = std::move(val); }

  /**
   * @brief Sets string used for values == 0 in INT8 types.
   *
   * @param val String to represent values == 0 in INT8 types
   */
  void set_false_value(std::string val) { _false_value = std::move(val); }

  /**
   * @brief (Re)sets the table being written.
   *
   * @param table Table to be written
   */
  void set_table(table_view const& table) { _table = table; }

  /**
   * @brief Sets the quote style for the writer.
   *
   * Note: Only the following quote styles are supported:
   *   1. MINIMAL: String columns containing special characters like row-delimiters/
   *               field-delimiter/quotes will be quoted.
   *   2. NONE: No quoting is done for any columns.
   *
   * @param quoting The new quote_style for the writer.
   */
  void set_quoting(quote_style quoting)
  {
    CUDF_EXPECTS(quoting == quote_style::MINIMAL || quoting == quote_style::NONE,
                 "Only MINIMAL and NONE are supported for quoting.");
    _quoting = quoting;
  }
};

/**
 * @brief Builder to build options for `writer_csv()`
 */
class csv_writer_options_builder {
  csv_writer_options options;  ///< Options to be built.

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit csv_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit csv_writer_options_builder(sink_info const& sink, table_view const& table)
    : options{sink, table}
  {
  }

  /**
   * @brief Sets optional column names.
   *
   * @param names Column names
   * @return this for chaining
   */
  csv_writer_options_builder& names(std::vector<std::string> names)
  {
    options.set_names(std::move(names));
    return *this;
  }

  /**
   * @brief Sets string to used for null entries.
   *
   * @param val String to represent null value
   * @return this for chaining
   */
  csv_writer_options_builder& na_rep(std::string val)
  {
    options.set_na_rep(std::move(val));
    return *this;
  };

  /**
   * @brief Enables/Disables headers being written to csv.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  csv_writer_options_builder& include_header(bool val)
  {
    options.enable_include_header(val);
    return *this;
  }

  /**
   * @brief Sets maximum number of rows to process for each file write.
   *
   * @param val Number of rows per chunk
   * @return this for chaining
   */
  csv_writer_options_builder& rows_per_chunk(int val)
  {
    options.set_rows_per_chunk(val);
    return *this;
  }

  /**
   * @brief Sets character used for separating lines.
   *
   * @param term Character to represent line termination
   * @return this for chaining
   */
  csv_writer_options_builder& line_terminator(std::string term)
  {
    options.set_line_terminator(std::move(term));
    return *this;
  }

  /**
   * @brief Sets character used for separating column values.
   *
   * @param delim Character to delimit column values
   * @return this for chaining
   */
  csv_writer_options_builder& inter_column_delimiter(char delim)
  {
    options.set_inter_column_delimiter(delim);
    return *this;
  }

  /**
   * @brief Sets string used for values != 0 in INT8 types.
   *
   * @param val String to represent values != 0 in INT8 types
   * @return this for chaining
   */
  csv_writer_options_builder& true_value(std::string val)
  {
    options.set_true_value(std::move(val));
    return *this;
  }

  /**
   * @brief Sets string used for values == 0 in INT8 types.
   *
   * @param val String to represent values == 0 in INT8 types
   * @return this for chaining
   */
  csv_writer_options_builder& false_value(std::string val)
  {
    options.set_false_value(std::move(val));
    return *this;
  }

  /**
   * @brief Sets the quote style for the writer.
   *
   * Only MINIMAL and NONE are supported.
   *
   * @param quoting The new quote style for the writer.
   * @return this for chaining
   */
  csv_writer_options_builder& quoting(quote_style quoting)
  {
    options.set_quoting(quoting);
    return *this;
  }

  /**
   * @brief move `csv_writer_options` member once it's built.
   */
  operator csv_writer_options&&() { return std::move(options); }

  /**
   * @brief move `csv_writer_options` member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `csv_writer_options` object's r-value reference
   */
  csv_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Writes a set of columns to CSV format.
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  auto destination = cudf::io::sink_info("dataset.csv");
 *  auto options     = cudf::io::csv_writer_options(destination, table->view())
 *    .na_rep(na)
 *    .include_header(include_header)
 *    .rows_per_chunk(rows_per_chunk);
 *
 *  cudf::io::write_csv(options);
 * @endcode
 *
 * @param options Settings for controlling writing behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void write_csv(csv_writer_options const& options,
               rmm::cuda_stream_view stream = cudf::get_default_stream());

/// @cond
struct is_supported_csv_write_type_fn {
  template <typename T>
  constexpr bool operator()() const
  {
    return cudf::io::detail::is_convertible_to_string_column<T>();
  }
};
/// @endcond

/**
 * @brief Checks if a cudf::data_type is supported for CSV writing.
 *
 * @param type The data_type to check.
 * @return true if the type is supported for CSV writing, false otherwise.
 */
constexpr bool is_supported_write_csv(data_type type)
{
  return cudf::type_dispatcher(type, is_supported_csv_write_type_fn{});
}

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
