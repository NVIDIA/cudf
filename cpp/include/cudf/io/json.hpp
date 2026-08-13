/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "types.hpp"

#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/io/detail/utils.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

/**
 * @file
 * @brief APIs for reading and writing JSON files.
 */

namespace CUDF_EXPORT cudf {
namespace io {
/**
 * @addtogroup io_readers
 * @{
 */

class json_reader_options_builder;

/**
 * @brief Allows specifying the target types for nested JSON data via json_reader_options'
 * `set_dtypes` method.
 */
struct schema_element {
  /**
   * @brief The type that this column should be converted to
   */
  data_type type;

  /**
   * @brief Allows specifying this column's child columns target type
   */
  std::map<std::string, schema_element> child_types;

  /**
   * @brief Allows specifying the order of the columns
   */
  std::optional<std::vector<std::string>> column_order;
};

/**
 * @brief Control the error recovery behavior of the json parser
 */
enum class json_recovery_mode_t {
  FAIL,              ///< Does not recover from an error when encountering an invalid format
  RECOVER_WITH_NULL  ///< Recovers from an error, replacing invalid records with null
};

/**
 * @brief Input arguments to the `read_json` interface.
 *
 * Available parameters are closely patterned after PANDAS' `read_json` API.
 * Not all parameters are supported. If the matching PANDAS' parameter
 * has a default value of `None`, then a default value of `-1` or `0` may be
 * used as the equivalent.
 *
 * Parameters in PANDAS that are unavailable or in cudf:
 *
 * | Name                 | Description                                      |
 * | -------------------- | ------------------------------------------------ |
 * | `orient`             | currently fixed-format                           |
 * | `typ`                | data is always returned as a cudf::table         |
 * | `convert_axes`       | use column functions for axes operations instead |
 * | `convert_dates`      | dates are detected automatically                 |
 * | `keep_default_dates` | dates are detected automatically                 |
 * | `numpy`              | data is always returned as a cudf::table         |
 * | `precise_float`      | there is only one converter                      |
 * | `date_unit`          | only millisecond units are supported             |
 * | `encoding`           | only ASCII-encoded data is supported             |
 * | `chunksize`          | use `byte_range_xxx` for chunking instead        |
 */
class json_reader_options {
 public:
  using dtype_variant =
    std::variant<std::vector<data_type>,
                 std::map<std::string, data_type>,
                 std::map<std::string, schema_element>,
                 schema_element>;  ///< Variant type holding dtypes information for the columns

 private:
  source_info _source;

  // Data types of the column; empty to infer dtypes
  dtype_variant _dtypes;
  // Specify the compression format of the source or infer from file extension
  compression_type _compression = compression_type::AUTO;

  // Read the file as a json object per line
  bool _lines = false;
  // Parse mixed types as a string column
  bool _mixed_types_as_string = false;
  // Delimiter separating records in JSON lines
  char _delimiter = '\n';
  // Prune columns on read, selected based on the _dtypes option
  bool _prune_columns = false;
  // Experimental features: new column tree construction
  bool _experimental = false;

  // Bytes to skip from the start
  size_t _byte_range_offset = 0;
  // Bytes to read; always reads complete rows
  size_t _byte_range_size = 0;

  // Whether to parse dates as DD/MM versus MM/DD
  bool _dayfirst = false;

  // Whether to keep the quote characters of string values
  bool _keep_quotes = false;

  // Normalize single quotes
  bool _normalize_single_quotes = false;

  // Normalize unquoted spaces and tabs
  bool _normalize_whitespace = false;

  // Whether to recover after an invalid JSON line
  json_recovery_mode_t _recovery_mode = json_recovery_mode_t::FAIL;

  // Validation checks for spark
  // Should the json validation be strict or not
  // Note: strict validation enforces the JSON specification https://www.json.org/json-en.html
  bool _strict_validation = false;
  // Allow leading zeros for numeric values.
  bool _allow_numeric_leading_zeros = true;
  // Allow non-numeric numbers: NaN, +INF, -INF, +Infinity, Infinity, -Infinity
  bool _allow_nonnumeric_numbers = true;
  // Allow unquoted control characters
  bool _allow_unquoted_control_chars = true;
  // Additional values to recognize as null values
  std::vector<std::string> _na_values;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read parquet file
   */
  explicit json_reader_options(source_info src) : _source{std::move(src)} {}

  friend json_reader_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  json_reader_options() = default;

  /**
   * @brief create json_reader_options_builder which will build json_reader_options.
   *
   * @param src source information used to read json file
   * @returns builder to build the options
   */
  static json_reader_options_builder builder(source_info src);

  /**
   * @brief Returns source info.
   *
   * @returns Source info
   */
  [[nodiscard]] source_info const& get_source() const { return _source; }

  /**
   * @brief Returns data types of the columns.
   *
   * @returns Data types of the columns
   */
  [[nodiscard]] dtype_variant const& get_dtypes() const { return _dtypes; }

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
  [[nodiscard]] size_t get_byte_range_offset() const { return _byte_range_offset; }

  /**
   * @brief Returns number of bytes to read.
   *
   * @return Number of bytes to read
   */
  [[nodiscard]] size_t get_byte_range_size() const { return _byte_range_size; }

  /**
   * @brief Returns number of bytes to read with padding.
   *
   * @return Number of bytes to read with padding
   */
  [[nodiscard]] size_t get_byte_range_size_with_padding() const
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
   * @return Number of bytes to pad
   */
  [[nodiscard]] size_t get_byte_range_padding() const
  {
    auto const num_columns =
      std::visit(cudf::detail::visitor_overload{
                   [](auto const& dtypes) { return dtypes.size(); },
                   [](schema_element const& dtypes) { return dtypes.child_types.size(); }},
                 _dtypes);

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
   * @brief Returns delimiter separating records in JSON lines
   *
   * @return Delimiter separating records in JSON lines
   */
  [[nodiscard]] char get_delimiter() const { return _delimiter; }

  /**
   * @brief Whether to read the file as a json object per line.
   *
   * @return `true` if reading the file as a json object per line
   */
  [[nodiscard]] bool is_enabled_lines() const { return _lines; }

  /**
   * @brief Whether to parse mixed types as a string column.
   *
   * @return `true` if mixed types are parsed as a string column
   */
  [[nodiscard]] bool is_enabled_mixed_types_as_string() const { return _mixed_types_as_string; }

  /**
   * @brief Whether to prune columns on read, selected based on the @ref set_dtypes option.
   *
   * When set as true, if the reader options include @ref set_dtypes, then
   * the reader will only return those columns which are mentioned in @ref set_dtypes.
   * If false, then all columns are returned, independent of the @ref set_dtypes
   * setting.
   *
   * @return True if column pruning is enabled
   */
  [[nodiscard]] bool is_enabled_prune_columns() const { return _prune_columns; }

  /**
   * @brief Whether to enable experimental features.
   *
   * When set to true, experimental features, such as the new column tree construction,
   * utf-8 matching of field names will be enabled.
   * @return true if experimental features are enabled
   */
  [[nodiscard]] bool is_enabled_experimental() const { return _experimental; }

  /**
   * @brief Whether to parse dates as DD/MM versus MM/DD.
   *
   * @returns true if dates are parsed as DD/MM, false if MM/DD
   */
  [[nodiscard]] bool is_enabled_dayfirst() const { return _dayfirst; }

  /**
   * @brief Whether the reader should keep quotes of string values.
   *
   * @returns true if the reader should keep quotes, false otherwise
   */
  [[nodiscard]] bool is_enabled_keep_quotes() const { return _keep_quotes; }

  /**
   * @brief Whether the reader should normalize single quotes around strings
   *
   * @returns true if the reader should normalize single quotes, false otherwise
   */
  [[nodiscard]] bool is_enabled_normalize_single_quotes() const { return _normalize_single_quotes; }

  /**
   * @brief Whether the reader should normalize unquoted whitespace characters
   *
   * @returns true if the reader should normalize whitespace, false otherwise
   */
  [[nodiscard]] bool is_enabled_normalize_whitespace() const { return _normalize_whitespace; }

  /**
   * @brief Queries the JSON reader's behavior on invalid JSON lines.
   *
   * @returns An enum that specifies the JSON reader's behavior on invalid JSON lines.
   */
  [[nodiscard]] json_recovery_mode_t recovery_mode() const { return _recovery_mode; }

  /**
   * @brief Whether json validation should be enforced strictly or not.
   *
   * @return true if it should be.
   */
  [[nodiscard]] bool is_strict_validation() const { return _strict_validation; }

  /**
   * @brief Whether leading zeros are allowed in numeric values.
   *
   * @note: This validation is enforced only if strict validation is enabled.
   *
   * @return true if leading zeros are allowed in numeric values
   */
  [[nodiscard]] bool is_allowed_numeric_leading_zeros() const
  {
    return _allow_numeric_leading_zeros;
  }

  /**
   * @brief Whether unquoted number values should be allowed NaN, +INF, -INF, +Infinity, Infinity,
   * and -Infinity.
   *
   * @note: This validation is enforced only if strict validation is enabled.
   *
   * @return true if leading zeros are allowed in numeric values
   */
  [[nodiscard]] bool is_allowed_nonnumeric_numbers() const { return _allow_nonnumeric_numbers; }

  /**
   * @brief Whether in a quoted string should characters greater than or equal to 0 and less than 32
   * be allowed without some form of escaping.
   *
   * @note: This validation is enforced only if strict validation is enabled.
   *
   * @return true if unquoted control chars are allowed.
   */
  [[nodiscard]] bool is_allowed_unquoted_control_chars() const
  {
    return _allow_unquoted_control_chars;
  }

  /**
   * @brief Returns additional values to recognize as null values.
   *
   * @return Additional values to recognize as null values
   */
  [[nodiscard]] std::vector<std::string> const& get_na_values() const { return _na_values; }

  /**
   * @brief Sets source info.
   *
   * @param src The source info.
   */
  void set_source(source_info src) { _source = std::move(src); }

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Vector of dtypes
   */
  void set_dtypes(std::vector<data_type> types) { _dtypes = std::move(types); }

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Vector dtypes in string format
   */
  void set_dtypes(std::map<std::string, data_type> types) { _dtypes = std::move(types); }

  /**
   * @brief Set data types for a potentially nested column hierarchy.
   *
   * @param types Map of column names to schema_element to support arbitrary nesting of data types
   */
  void set_dtypes(std::map<std::string, schema_element> types) { _dtypes = std::move(types); }

  /**
   * @brief Set data types for a potentially nested column hierarchy.
   *
   * @param types schema element with column names and column order to support arbitrary nesting of
   * data types
   */
  void set_dtypes(schema_element types);

  /**
   * @brief Set the compression type.
   *
   * @param comp_type The compression type used
   */
  void set_compression(compression_type comp_type) { _compression = comp_type; }

  /**
   * @brief Set number of bytes to skip from source start.
   *
   * @param offset Number of bytes of offset
   */
  void set_byte_range_offset(size_t offset) { _byte_range_offset = offset; }

  /**
   * @brief Set number of bytes to read.
   *
   * @param size Number of bytes to read
   */
  void set_byte_range_size(size_t size) { _byte_range_size = size; }

  /**
   * @brief Set delimiter separating records in JSON lines
   *
   * @param delimiter Delimiter separating records in JSON lines
   */
  void set_delimiter(char delimiter)
  {
    switch (delimiter) {
      case '{':
      case '[':
      case '}':
      case ']':
      case ',':
      case ':':
      case '"':
      case '\'':
      case '\\':
      case ' ':
      case '\t':
      case '\r': CUDF_FAIL("Unsupported delimiter character.", std::invalid_argument); break;
    }
    _delimiter = delimiter;
  }

  /**
   * @brief Set whether to read the file as a json object per line.
   *
   * @param val Boolean value to enable/disable the option to read each line as a json object
   */
  void enable_lines(bool val) { _lines = val; }

  /**
   * @brief Set whether to parse mixed types as a string column.
   * Also enables forcing to read a struct as string column using schema.
   *
   * @param val Boolean value to enable/disable parsing mixed types as a string column
   */
  void enable_mixed_types_as_string(bool val) { _mixed_types_as_string = val; }

  /**
   * @brief Set whether to prune columns on read, selected based on the @ref set_dtypes option.
   *
   * When set as true, if the reader options include @ref set_dtypes, then
   * the reader will only return those columns which are mentioned in @ref set_dtypes.
   * If false, then all columns are returned, independent of the @ref set_dtypes setting.
   *
   * @param val Boolean value to enable/disable column pruning
   */
  void enable_prune_columns(bool val) { _prune_columns = val; }

  /**
   * @brief Set whether to enable experimental features.
   *
   * When set to true, experimental features, such as the new column tree construction,
   * utf-8 matching of field names will be enabled.
   *
   * @param val Boolean value to enable/disable experimental features
   */
  void enable_experimental(bool val) { _experimental = val; }

  /**
   * @brief Set whether to parse dates as DD/MM versus MM/DD.
   *
   * @param val Boolean value to enable/disable day first parsing format
   */
  void enable_dayfirst(bool val) { _dayfirst = val; }

  /**
   * @brief Set whether the reader should keep quotes of string values.
   *
   * @param val Boolean value to indicate whether the reader should keep quotes
   * of string values
   */
  void enable_keep_quotes(bool val) { _keep_quotes = val; }

  /**
   * @brief Set whether the reader should enable normalization of single quotes around strings.
   *
   * @param val Boolean value to indicate whether the reader should normalize single quotes around
   * strings
   */
  void enable_normalize_single_quotes(bool val) { _normalize_single_quotes = val; }

  /**
   * @brief Set whether the reader should enable normalization of unquoted whitespace
   *
   * @param val Boolean value to indicate whether the reader should normalize unquoted whitespace
   * characters i.e. tabs and spaces
   */
  void enable_normalize_whitespace(bool val) { _normalize_whitespace = val; }

  /**
   * @brief Specifies the JSON reader's behavior on invalid JSON lines.
   *
   * @param val An enum value to indicate the JSON reader's behavior on invalid JSON lines.
   */
  void set_recovery_mode(json_recovery_mode_t val) { _recovery_mode = val; }

  /**
   * @brief Set whether strict validation is enabled or not.
   *
   * @param val Boolean value to indicate whether strict validation is enabled.
   */
  void set_strict_validation(bool val) { _strict_validation = val; }

  /**
   * @brief Set whether leading zeros are allowed in numeric values. Strict validation
   * must be enabled for this to work.
   *
   * @throw cudf::logic_error if `strict_validation` is not enabled before setting this option.
   *
   * @param val Boolean value to indicate whether leading zeros are allowed in numeric values
   */
  void allow_numeric_leading_zeros(bool val)
  {
    CUDF_EXPECTS(_strict_validation, "Strict validation must be enabled for this to work.");
    _allow_numeric_leading_zeros = val;
  }

  /**
   * @brief Set whether unquoted number values should be allowed NaN, +INF, -INF, +Infinity,
   * Infinity, and -Infinity. Strict validation must be enabled for this to work.
   *
   * @throw cudf::logic_error if `strict_validation` is not enabled before setting this option.
   *
   * @param val Boolean value to indicate whether leading zeros are allowed in numeric values
   */
  void allow_nonnumeric_numbers(bool val)
  {
    CUDF_EXPECTS(_strict_validation, "Strict validation must be enabled for this to work.");
    _allow_nonnumeric_numbers = val;
  }

  /**
   * @brief Set whether in a quoted string should characters greater than or equal to 0
   * and less than 32 be allowed without some form of escaping. Strict validation must
   * be enabled for this to work.
   *
   * @throw cudf::logic_error if `strict_validation` is not enabled before setting this option.
   *
   * @param val true to indicate whether unquoted control chars are allowed.
   */
  void allow_unquoted_control_chars(bool val)
  {
    CUDF_EXPECTS(_strict_validation, "Strict validation must be enabled for this to work.");
    _allow_unquoted_control_chars = val;
  }

  /**
   * @brief Sets additional values to recognize as null values.
   *
   * @param vals Vector of values to be considered to be null
   */
  void set_na_values(std::vector<std::string> vals) { _na_values = std::move(vals); }
};

/**
 * @brief Builds settings to use for `read_json()`.
 */
class json_reader_options_builder {
  json_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit json_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read avro file
   */
  explicit json_reader_options_builder(source_info src) : options{std::move(src)} {}

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Vector of dtypes
   * @return this for chaining
   */
  json_reader_options_builder& dtypes(std::vector<data_type> types)
  {
    options.set_dtypes(std::move(types));
    return *this;
  }

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Column name -> dtype map
   * @return this for chaining
   */
  json_reader_options_builder& dtypes(std::map<std::string, data_type> types)
  {
    options.set_dtypes(std::move(types));
    return *this;
  }

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Column name -> schema_element map
   * @return this for chaining
   */
  json_reader_options_builder& dtypes(std::map<std::string, schema_element> types)
  {
    options.set_dtypes(std::move(types));
    return *this;
  }

  /**
   * @brief Set data types for columns to be read.
   *
   * @param types Struct schema_element with Column name -> schema_element with map and order
   * @return this for chaining
   */
  json_reader_options_builder& dtypes(schema_element types)
  {
    options.set_dtypes(std::move(types));
    return *this;
  }

  /**
   * @brief Set the compression type.
   *
   * @param comp_type The compression type used
   * @return this for chaining
   */
  json_reader_options_builder& compression(compression_type comp_type)
  {
    options.set_compression(comp_type);
    return *this;
  }

  /**
   * @brief Set number of bytes to skip from source start.
   *
   * @param offset Number of bytes of offset
   * @return this for chaining
   */
  json_reader_options_builder& byte_range_offset(size_type offset)
  {
    options.set_byte_range_offset(offset);
    return *this;
  }

  /**
   * @brief Set number of bytes to read.
   *
   * @param size Number of bytes to read
   * @return this for chaining
   */
  json_reader_options_builder& byte_range_size(size_type size)
  {
    options.set_byte_range_size(size);
    return *this;
  }

  /**
   * @brief Set delimiter separating records in JSON lines
   *
   * @param delimiter Delimiter separating records in JSON lines
   * @return this for chaining
   */
  json_reader_options_builder& delimiter(char delimiter)
  {
    options.set_delimiter(delimiter);
    return *this;
  }

  /**
   * @brief Set whether to read the file as a json object per line.
   *
   * @param val Boolean value to enable/disable the option to read each line as a json object
   * @return this for chaining
   */
  json_reader_options_builder& lines(bool val)
  {
    options.enable_lines(val);
    return *this;
  }

  /**
   * @brief Set whether to parse mixed types as a string column.
   * Also enables forcing to read a struct as string column using schema.
   *
   * @param val Boolean value to enable/disable parsing mixed types as a string column
   * @return this for chaining
   */
  json_reader_options_builder& mixed_types_as_string(bool val)
  {
    options.enable_mixed_types_as_string(val);
    return *this;
  }

  /**
   * @brief Set whether to prune columns on read, selected based on the @ref dtypes option.
   *
   * When set as true, if the reader options include @ref dtypes, then
   * the reader will only return those columns which are mentioned in @ref dtypes.
   * If false, then all columns are returned, independent of the @ref dtypes setting.
   *
   * @param val Boolean value to enable/disable column pruning
   * @return this for chaining
   */
  json_reader_options_builder& prune_columns(bool val)
  {
    options.enable_prune_columns(val);
    return *this;
  }

  /**
   * @brief Set whether to enable experimental features.
   *
   * When set to true, experimental features, such as the new column tree construction,
   * utf-8 matching of field names will be enabled.
   *
   * @param val Boolean value to enable/disable experimental features
   * @return this for chaining
   */
  json_reader_options_builder& experimental(bool val)
  {
    options.enable_experimental(val);
    return *this;
  }

  /**
   * @brief Set whether to parse dates as DD/MM versus MM/DD.
   *
   * @param val Boolean value to enable/disable day first parsing format
   * @return this for chaining
   */
  json_reader_options_builder& dayfirst(bool val)
  {
    options.enable_dayfirst(val);
    return *this;
  }

  /**
   * @brief Set whether the reader should keep quotes of string values.
   *
   * @param val Boolean value to indicate whether the reader should keep quotes
   * of string values
   * @return this for chaining
   */
  json_reader_options_builder& keep_quotes(bool val)
  {
    options.enable_keep_quotes(val);
    return *this;
  }

  /**
   * @brief Set whether the reader should normalize single quotes around strings
   *
   * @param val Boolean value to indicate whether the reader should normalize single quotes
   * of strings
   * @return this for chaining
   */
  json_reader_options_builder& normalize_single_quotes(bool val)
  {
    options.enable_normalize_single_quotes(val);
    return *this;
  }

  /**
   * @brief Set whether the reader should normalize unquoted whitespace
   *
   * @param val Boolean value to indicate whether the reader should normalize unquoted
   * whitespace
   * @return this for chaining
   */
  json_reader_options_builder& normalize_whitespace(bool val)
  {
    options.enable_normalize_whitespace(val);
    return *this;
  }

  /**
   * @brief Specifies the JSON reader's behavior on invalid JSON lines.
   *
   * @param val An enum value to indicate the JSON reader's behavior on invalid JSON lines.
   * @return this for chaining
   */
  json_reader_options_builder& recovery_mode(json_recovery_mode_t val)
  {
    options.set_recovery_mode(val);
    return *this;
  }

  /**
   * @brief Set whether json validation should be strict or not.
   *
   * @param val Boolean value to indicate whether json validation should be strict or not.
   * @return this for chaining
   */
  json_reader_options_builder& strict_validation(bool val)
  {
    options.set_strict_validation(val);
    return *this;
  }

  /**
   * @brief Set Whether leading zeros are allowed in numeric values. Strict validation must
   * be enabled for this to have any effect.
   *
   * @throw cudf::logic_error if `strict_validation` is not enabled before setting this option.
   *
   * @param val Boolean value to indicate whether leading zeros are allowed in numeric values
   * @return this for chaining
   */
  json_reader_options_builder& numeric_leading_zeros(bool val)
  {
    options.allow_numeric_leading_zeros(val);
    return *this;
  }

  /**
   * @brief Set whether specific unquoted number values are valid JSON. The values are NaN,
   * +INF, -INF, +Infinity, Infinity, and -Infinity.
   * Strict validation must be enabled for this to have any effect.
   *
   * @throw cudf::logic_error if `strict_validation` is not enabled before setting this option.
   *
   * @param val Boolean value to indicate if unquoted nonnumeric values are valid json or not.
   * @return this for chaining
   */
  json_reader_options_builder& nonnumeric_numbers(bool val)
  {
    options.allow_nonnumeric_numbers(val);
    return *this;
  }

  /**
   * @brief Set whether chars >= 0 and < 32 are allowed in a quoted string without
   * some form of escaping. Strict validation must be enabled for this to have any effect.
   *
   * @throw cudf::logic_error if `strict_validation` is not enabled before setting this option.
   *
   * @param val Boolean value to indicate if unquoted control chars are allowed or not.
   * @return this for chaining
   */
  json_reader_options_builder& unquoted_control_chars(bool val)
  {
    options.allow_unquoted_control_chars(val);
    return *this;
  }

  /**
   * @brief Sets additional values to recognize as null values.
   *
   * @param vals Vector of values to be considered to be null
   * @return this for chaining
   */
  json_reader_options_builder& na_values(std::vector<std::string> vals)
  {
    options.set_na_values(std::move(vals));
    return *this;
  }

  /**
   * @brief move json_reader_options member once it's built.
   */
  operator json_reader_options&&() { return std::move(options); }

  /**
   * @brief move json_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `json_reader_options` object r-value reference
   */
  json_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads a JSON dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  auto source  = cudf::io::source_info("dataset.json");
 *  auto options = cudf::io::read_json_options::builder(source);
 *  auto result  = cudf::io::read_json(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata.
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_json(
  json_reader_options options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Optional diagnostics produced by `read_json_with_diagnostics`.
 *
 * Reserved for signals that don't belong on the result `table_metadata` because
 * they are reader-specific (and adding them to the shared `column_name_info` would
 * change the public ABI for that struct, breaking downstream packages that have
 * its destructor inlined into their binaries).
 */
struct json_reader_diagnostics {
  /**
   * @brief Top-level column names whose value tree contained at least one JSON node
   * whose category (`NC_STRUCT` / `NC_LIST` / `NC_VAL`) did not match the requested
   * schema type. Empty when the reader is invoked without a user-supplied schema or
   * when no mismatches were observed.
   *
   * Consumers (e.g. spark-rapids-jni) can use this signal to implement their own
   * per-column policy on schema mismatch (e.g. Spark's `JacksonParser` nulls the
   * entire depth-1 ancestor field for such rows).
   */
  std::vector<std::string> top_level_columns_with_schema_mismatch;
};

/**
 * @brief Result of `read_json_with_diagnostics`: the parsed table together with
 * reader-specific diagnostics.
 */
struct json_reader_result {
  table_with_metadata data;             ///< Parsed table and standard table metadata
  json_reader_diagnostics diagnostics;  ///< Reader-specific diagnostics
};

/**
 * @brief Optional row-level diagnostics produced by `read_json_with_row_diagnostics`.
 *
 * Reserved for reader-specific diagnostics whose collection has additional cost and whose payload
 * is not part of the ABI of `json_reader_diagnostics`.
 */
struct json_reader_row_diagnostics {
  /**
   * @brief Top-level column names whose value tree contained at least one JSON node
   * whose category (`NC_STRUCT` / `NC_LIST` / `NC_VAL`) did not match the requested
   * schema type. Empty when the reader is invoked without a user-supplied schema or
   * when no mismatches were observed.
   */
  std::vector<std::string> top_level_columns_with_schema_mismatch;

  /**
   * @brief Row indices for one top-level column that had schema mismatches.
   */
  struct schema_mismatch_rows {
    std::string column_name;             ///< Top-level output column name
    std::vector<size_type> row_indices;  ///< Rows whose depth-1 ancestor should be nulled
  };

  /**
   * @brief Row indices grouped by top-level output column for schema mismatches.
   *
   * Each entry names a top-level output column and lists the rows in that column
   * where at least one descendant JSON node's category did not match the requested
   * schema type. The indices are sorted, unique, and relative to the final
   * returned table, not to an internal reader batch. Entries are ordered consistently
   * with `top_level_columns_with_schema_mismatch`.
   */
  std::vector<schema_mismatch_rows> top_level_columns_with_schema_mismatch_rows;
};

/**
 * @brief Result of `read_json_with_row_diagnostics`: the parsed table together with
 * reader-specific row-level diagnostics.
 */
struct json_reader_result_with_row_diagnostics {
  table_with_metadata data;                 ///< Parsed table and standard table metadata
  json_reader_row_diagnostics diagnostics;  ///< Reader-specific row-level diagnostics
};

/**
 * @brief Reads a JSON dataset into a set of columns, additionally reporting reader
 * diagnostics that do not belong on the standard `table_metadata`.
 *
 * Behaves identically to `read_json` for column construction. The additional
 * `diagnostics` field carries reader-specific observations such as
 * `top_level_columns_with_schema_mismatch`.
 *
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * result.
 *
 * @return The parsed table, its metadata, and reader diagnostics
 */
json_reader_result read_json_with_diagnostics(
  json_reader_options options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reads a JSON dataset into a set of columns, additionally reporting reader
 * diagnostics with row-level schema mismatch details.
 *
 * Behaves identically to `read_json` for column construction. The additional
 * `diagnostics` field carries reader-specific observations such as
 * `top_level_columns_with_schema_mismatch_rows`.
 *
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * result.
 *
 * @return The parsed table, its metadata, and reader diagnostics with row-level details
 */
json_reader_result_with_row_diagnostics read_json_with_row_diagnostics(
  json_reader_options options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

/**
 * @addtogroup io_writers
 * @{
 */

/**
 *@brief Builder to build options for `writer_json()`.
 */
class json_writer_options_builder;

/**
 * @brief Settings to use for `write_json()`.
 */
class json_writer_options {
  // Specify the sink to use for writer output
  sink_info _sink;
  // Specify the compression format of the sink
  compression_type _compression = compression_type::NONE;
  // maximum number of rows to write in each chunk (limits memory use)
  size_type _rows_per_chunk = std::numeric_limits<size_type>::max();
  // Set of columns to output
  table_view _table;
  // string to use for null entries
  std::string _na_rep = "";
  // Indicates whether to output nulls as 'null' or exclude the field
  bool _include_nulls = false;
  // Indicates whether to use JSON lines for records format
  bool _lines = false;
  // string to use for values != 0 in INT8 types (default 'true')
  std::string _true_value = std::string{"true"};
  // string to use for values == 0 in INT8 types (default 'false')
  std::string _false_value = std::string{"false"};
  // Names of all columns; if empty, writer will generate column names
  std::optional<table_metadata> _metadata;  // Optional column names
  // Indicates whether to escape UTF-8 characters in JSON output
  bool _enable_utf8_escaped = true;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit json_writer_options(sink_info sink, table_view table)
    : _sink(std::move(sink)), _rows_per_chunk(table.num_rows()), _table(std::move(table))
  {
  }

  friend json_writer_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit json_writer_options() = default;

  /**
   * @brief Create builder to create `json_writer_options`.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build json_writer_options
   */
  static json_writer_options_builder builder(sink_info const& sink, table_view const& table);

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
   * @brief Returns metadata information.
   *
   * @return Metadata information
   */
  [[nodiscard]] std::optional<table_metadata> const& get_metadata() const { return _metadata; }

  /**
   * @brief Returns string to used for null entries.
   *
   * @return string to used for null entries
   */
  [[nodiscard]] std::string const& get_na_rep() const { return _na_rep; }

  /**
   * @brief Returns compression type used for sink
   *
   * @return compression type for sink
   */
  [[nodiscard]] compression_type get_compression() const { return _compression; }

  /**
   * @brief Whether to output nulls as 'null'.
   *
   * @return `true` if nulls are output as 'null'
   */
  [[nodiscard]] bool is_enabled_include_nulls() const { return _include_nulls; }

  /**
   * @brief Whether to use JSON lines for records format.
   *
   * @return `true` if JSON lines is used for records format
   */
  [[nodiscard]] bool is_enabled_lines() const { return _lines; }

  /**
   * @brief Enable or disable writing escaped UTF-8 characters in JSON output.
   *
   * Example:
   * With `enable_utf8_escaped(false)`, the string `"ẅ"` is written as-is instead of:
   * `"\u1e85"`.
   *
   * @note Enabling this is useful for producing more human-readable JSON.
   *
   * @param val Boolean value to enable/disable UTF-8 escaped output
   */
  void enable_utf8_escaped(bool val) { _enable_utf8_escaped = val; }

  /**
   * @brief Check whether UTF-8 escaped output is enabled.
   *
   * @return true if UTF-8 escaped output is enabled, false otherwise
   */
  [[nodiscard]] bool is_enabled_utf8_escaped() const { return _enable_utf8_escaped; }

  /**
   * @brief Returns maximum number of rows to process for each file write.
   *
   * @return Maximum number of rows to process for each file write
   */
  [[nodiscard]] size_type get_rows_per_chunk() const { return _rows_per_chunk; }

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

  // Setter

  /**
   * @brief Sets table to be written to output.
   *
   * @param tbl Table for the output
   */
  void set_table(table_view tbl) { _table = tbl; }

  /**
   * @brief Sets compression type to be used
   *
   * @param comptype Compression type for sink
   */
  void set_compression(compression_type comptype) { _compression = comptype; }

  /**
   * @brief Sets metadata.
   *
   * @param metadata Associated metadata
   */
  void set_metadata(table_metadata metadata) { _metadata = std::move(metadata); }

  /**
   * @brief Sets string to used for null entries.
   *
   * @param val String to represent null value
   */
  void set_na_rep(std::string val) { _na_rep = std::move(val); }

  /**
   * @brief Enables/Disables output of nulls as 'null'.
   *
   * @param val Boolean value to enable/disable
   */
  void enable_include_nulls(bool val) { _include_nulls = val; }

  /**
   * @brief Enables/Disables JSON lines for records format.
   *
   * @param val Boolean value to enable/disable JSON lines
   */
  void enable_lines(bool val) { _lines = val; }

  /**
   * @brief Sets maximum number of rows to process for each file write.
   *
   * @param val Number of rows per chunk
   */
  void set_rows_per_chunk(size_type val) { _rows_per_chunk = val; }

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
};

/**
 * @brief Builder to build options for `writer_json()`
 */
class json_writer_options_builder {
  json_writer_options options;  ///< Options to be built.

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit json_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit json_writer_options_builder(sink_info const& sink, table_view const& table)
    : options{sink, table}
  {
  }

  /**
   * @brief Sets table to be written to output.
   *
   * @param tbl Table for the output
   * @return this for chaining
   */
  json_writer_options_builder& table(table_view tbl)
  {
    options.set_table(tbl);
    return *this;
  }

  /**
   * @brief Sets compression type of output sink
   *
   * @param comptype Compression type used
   * @return this for chaining
   */
  json_writer_options_builder& compression(compression_type comptype)
  {
    options.set_compression(comptype);
    return *this;
  }

  /**
   * @brief Sets optional metadata (with column names).
   *
   * @param metadata metadata (with column names)
   * @return this for chaining
   */
  json_writer_options_builder& metadata(table_metadata metadata)
  {
    options.set_metadata(std::move(metadata));
    return *this;
  }

  /**
   * @brief Sets string to used for null entries.
   *
   * @param val String to represent null value
   * @return this for chaining
   */
  json_writer_options_builder& na_rep(std::string val)
  {
    options.set_na_rep(std::move(val));
    return *this;
  };

  /**
   * @brief Enables/Disables output of nulls as 'null'.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  json_writer_options_builder& include_nulls(bool val)
  {
    options.enable_include_nulls(val);
    return *this;
  }

  /**
   * @brief Enables/Disable UTF-8 escaped output for string fields.
   *
   * Default is `false`, which escapes all non-ASCII characters.
   *
   * @param val Boolean value to enable/disable escaped UTF-8 output
   * @return this for chaining
   */
  json_writer_options_builder& utf8_escaped(bool val)
  {
    options.enable_utf8_escaped(val);
    return *this;
  }

  /**
   * @brief Enables/Disables JSON lines for records format.
   *
   * @param val Boolean value to enable/disable
   * @return this for chaining
   */
  json_writer_options_builder& lines(bool val)
  {
    options.enable_lines(val);
    return *this;
  }

  /**
   * @brief Sets maximum number of rows to process for each file write.
   *
   * @param val Number of rows per chunk
   * @return this for chaining
   */
  json_writer_options_builder& rows_per_chunk(int val)
  {
    options.set_rows_per_chunk(val);
    return *this;
  }

  /**
   * @brief Sets string used for values != 0 in INT8 types.
   *
   * @param val String to represent values != 0 in INT8 types
   * @return this for chaining
   */
  json_writer_options_builder& true_value(std::string val)
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
  json_writer_options_builder& false_value(std::string val)
  {
    options.set_false_value(std::move(val));
    return *this;
  }

  /**
   * @brief move `json_writer_options` member once it's built.
   */
  operator json_writer_options&&() { return std::move(options); }

  /**
   * @brief move `json_writer_options` member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `json_writer_options` object's r-value reference
   */
  json_writer_options&& build() { return std::move(options); }
};

/**
 * @brief Writes a set of columns to JSON format.
 *
 * The following code snippet demonstrates how to write columns to a file:
 * @code
 *  auto destination = cudf::io::sink_info("dataset.json");
 *  auto options     = cudf::io::json_writer_options(destination, table->view())
 *    .na_rep(na)
 *    .lines(lines)
 *    .rows_per_chunk(rows_per_chunk);
 *
 *  cudf::io::write_json(options);
 * @endcode
 *
 * @param options Settings for controlling writing behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void write_json(json_writer_options const& options,
                rmm::cuda_stream_view stream = cudf::get_default_stream());

/// @cond
struct is_supported_json_write_type_fn {
  template <typename T>
  constexpr bool operator()() const
  {
    return cudf::io::detail::is_convertible_to_string_column<T>();
  }
};
/// @endcond

/**
 * @brief Checks if a cudf::data_type is supported for JSON writing.
 *
 * @param type The data_type to check.
 * @return true if the type is supported for JSON writing, false otherwise.
 */
constexpr bool is_supported_write_json(data_type type)
{
  return cudf::type_dispatcher(type, is_supported_json_write_type_fn{});
}

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
