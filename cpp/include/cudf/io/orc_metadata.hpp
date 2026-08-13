/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file orc_metadata.hpp
 * @brief Freeform APIs for ORC metadata
 */

#pragma once

#include <cudf/io/orc_types.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/export.hpp>

#include <optional>
#include <variant>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io {
/**
 * @addtogroup io_types
 * @{
 */

//! ORC data type
using cudf::io::orc::TypeKind;

/**
 * @brief Holds column names and buffers containing raw file-level and stripe-level statistics.
 *
 * The buffers can be parsed using a Protobuf parser. Alternatively, use `parsed_orc_statistics` to
 * get the statistics parsed into a libcudf representation.
 *
 * The `column_names` and `file_stats` members contain one element per column. The `stripes_stats`
 * contains one element per stripe, where each element contains column statistics for each column.
 */
struct raw_orc_statistics {
  std::vector<std::string> column_names;                ///< Column names
  std::vector<std::string> file_stats;                  ///< File-level statistics for each column
  std::vector<std::vector<std::string>> stripes_stats;  ///< Stripe-level statistics for each column
};

/**
 * @brief Reads file-level and stripe-level statistics of ORC dataset.
 *
 * @ingroup io_readers
 *
 * The following code snippet demonstrates how to read statistics of a dataset
 * from a file:
 * @code
 *  auto result = cudf::read_raw_orc_statistics(cudf::source_info("dataset.orc"));
 * @endcode
 *
 * @param src_info Dataset source
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Column names and encoded ORC statistics
 */
raw_orc_statistics read_raw_orc_statistics(
  source_info const& src_info, rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Monostate type alias for the statistics variant.
 */
using no_statistics = std::monostate;

/**
 * @brief Base class for column statistics that include optional minimum and maximum.
 *
 * Includes accessors for the minimum and maximum values.
 */
template <typename T>
struct minmax_statistics {
  std::optional<T> minimum;  ///< Minimum value
  std::optional<T> maximum;  ///< Maximum value
};

/**
 * @brief Base class for column statistics that include an optional sum.
 *
 * Includes accessors for the sum value.
 */
template <typename T>
struct sum_statistics {
  std::optional<T> sum;  ///< Sum of values in column
};

/**
 * @brief Statistics for integral columns.
 */
struct integer_statistics : minmax_statistics<int64_t>, sum_statistics<int64_t> {};

/**
 * @brief Statistics for floating point columns.
 */
struct double_statistics : minmax_statistics<double>, sum_statistics<double> {};

/**
 * @brief Statistics for string columns.
 *
 * The `minimum` and `maximum` are the first and last elements, respectively, in lexicographical
 * order. The `sum` is the total length of elements in the column.
 * Note: According to ORC specs, the sum should be signed, but pyarrow uses unsigned value
 */
struct string_statistics : minmax_statistics<std::string>, sum_statistics<int64_t> {};

/**
 * @brief Statistics for boolean columns.
 *
 * The `count` array contains the count of `true` values.
 */
struct bucket_statistics {
  std::vector<uint64_t> count;  ///< count of `true` values
};

/**
 * @brief Statistics for decimal columns.
 */
struct decimal_statistics : minmax_statistics<std::string>, sum_statistics<std::string> {};

/**
 * @brief Statistics for date(time) columns.
 */
using date_statistics = minmax_statistics<int32_t>;

/**
 * @brief Statistics for binary columns.
 *
 * The `sum` is the total number of bytes across all elements.
 */
using binary_statistics = sum_statistics<int64_t>;

/**
 * @brief Statistics for timestamp columns.
 *
 * The `minimum` and `maximum` min/max elements in the column, as the number of milliseconds since
 * the UNIX epoch. The `minimum_utc` and `maximum_utc` are the same values adjusted to UTC.
 */
struct timestamp_statistics : minmax_statistics<int64_t> {
  std::optional<int64_t> minimum_utc;     ///< minimum in milliseconds
  std::optional<int64_t> maximum_utc;     ///< maximum in milliseconds
  std::optional<uint32_t> minimum_nanos;  ///< nanoseconds part of the minimum
  std::optional<uint32_t> maximum_nanos;  ///< nanoseconds part of the maximum
};

/**
 * @brief Variant type for ORC type-specific column statistics.
 *
 * The variant can hold any of the supported column statistics types.
 */
using statistics_type = std::variant<no_statistics,
                                     integer_statistics,
                                     double_statistics,
                                     string_statistics,
                                     bucket_statistics,
                                     decimal_statistics,
                                     date_statistics,
                                     binary_statistics,
                                     timestamp_statistics>;

//! Orc I/O interfaces
namespace orc::detail {
// forward declare the type that protobuf_reader uses. The `cudf::io::column_statistics` objects,
// returned from `read_parsed_orc_statistics`, are constructed from
// `cudf::io::orc::detail::column_statistics` objects that `protobuf_reader` initializes.
struct column_statistics;
}  // namespace orc::detail

/**
 * @brief Contains per-column ORC statistics.
 *
 * All columns can have the `number_of_values` statistics. Depending on the data type, a column can
 * have additional statistics, accessible through `type_specific_stats` accessor.
 */
struct column_statistics {
  std::optional<uint64_t> number_of_values;  ///< number of statistics
  std::optional<bool> has_null;              ///< column has any nulls
  statistics_type type_specific_stats;       ///< type-specific statistics

  /**
   * @brief Construct a new column statistics object
   *
   * @param detail_statistics The statistics to initialize the object with
   */
  column_statistics(orc::detail::column_statistics&& detail_statistics);
};

/**
 * @brief Holds column names and parsed file-level and stripe-level statistics.
 *
 * The `column_names` and `file_stats` members contain one element per column. The `stripes_stats`
 * member contains one element per stripe, where each element contains column statistics for each
 * column.
 */
struct parsed_orc_statistics {
  std::vector<std::string> column_names;                      ///< column names
  std::vector<column_statistics> file_stats;                  ///< file-level statistics
  std::vector<std::vector<column_statistics>> stripes_stats;  ///< stripe-level statistics
};

/**
 * @brief Reads file-level and stripe-level statistics of ORC dataset.
 *
 * @ingroup io_readers
 *
 * @param src_info Dataset source
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Column names and decoded ORC statistics
 */
parsed_orc_statistics read_parsed_orc_statistics(
  source_info const& src_info, rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Schema of an ORC column, including the nested columns.
 */
struct orc_column_schema {
 public:
  /**
   * @brief constructor
   *
   * @param name column name
   * @param type ORC type
   * @param children child columns (empty for non-nested types)
   */
  orc_column_schema(std::string_view name, TypeKind type, std::vector<orc_column_schema> children)
    : _name{name}, _type_kind{type}, _children{std::move(children)}
  {
  }

  /**
   * @brief Returns ORC column name; can be empty.
   *
   * @return Column name
   */
  [[nodiscard]] auto name() const { return _name; }

  /**
   * @brief Returns ORC type of the column.
   *
   * @return Column ORC type
   */
  [[nodiscard]] auto type_kind() const { return _type_kind; }

  /**
   * @brief Returns schemas of all child columns.
   *
   * @return Children schemas
   */
  [[nodiscard]] auto const& children() const& { return _children; }

  /** @copydoc children
   * Children array is moved out of the object (rvalues only).
   *
   */
  [[nodiscard]] auto children() && { return std::move(_children); }

  /**
   * @brief Returns schema of the child with the given index.
   *
   * @param idx child index
   *
   * @return Child schema
   */
  [[nodiscard]] auto const& child(int idx) const& { return children().at(idx); }

  /** @copydoc child
   * Child is moved out of the object (rvalues only).
   *
   */
  [[nodiscard]] auto child(int idx) && { return std::move(children().at(idx)); }

  /**
   * @brief Returns the number of child columns.
   *
   * @return Children count
   */
  [[nodiscard]] auto num_children() const { return children().size(); }

 private:
  std::string _name;
  TypeKind _type_kind;
  std::vector<orc_column_schema> _children;
};

/**
 * @brief Schema of an ORC file.
 */
struct orc_schema {
 public:
  /**
   * @brief constructor
   *
   * @param root_column_schema root column
   */
  orc_schema(orc_column_schema root_column_schema) : _root{std::move(root_column_schema)} {}

  /**
   * @brief Returns the schema of the struct column that contains all columns as fields.
   *
   * @return Root column schema
   */
  [[nodiscard]] auto const& root() const& { return _root; }

  /** @copydoc root
   * Root column schema is moved out of the object (rvalues only).
   *
   */
  [[nodiscard]] auto root() && { return std::move(_root); }

 private:
  orc_column_schema _root;
};

/**
 * @brief Information about content of an ORC file.
 */
class orc_metadata {
 public:
  /**
   * @brief constructor
   *
   * @param schema ORC schema
   * @param num_rows number of rows
   * @param num_stripes number of stripes
   */
  orc_metadata(orc_schema schema, uint64_t num_rows, size_type num_stripes)
    : _schema{std::move(schema)}, _num_rows{num_rows}, _num_stripes{num_stripes}
  {
  }

  /**
   * @brief Returns the ORC schema.
   *
   * @return ORC schema
   */
  [[nodiscard]] auto const& schema() const { return _schema; }

  ///< Number of rows in the root column; can vary for nested columns
  /**
   * @brief Returns the number of rows of the root column.
   *
   * If a file contains list columns, nested columns can have a different number of rows.
   *
   * @return Number of rows
   */
  [[nodiscard]] auto num_rows() const { return _num_rows; }

  /**
   * @brief Returns the number of stripes in the file.
   *
   * @return Number of stripes
   */
  [[nodiscard]] auto num_stripes() const { return _num_stripes; }

 private:
  orc_schema _schema;
  uint64_t _num_rows;
  size_type _num_stripes;
};

/**
 * @brief Reads metadata of ORC dataset.
 *
 * @ingroup io_readers
 *
 * @param src_info Dataset source
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return orc_metadata with ORC schema, number of rows and number of stripes.
 */
orc_metadata read_orc_metadata(source_info const& src_info,
                               rmm::cuda_stream_view stream = cudf::get_default_stream());

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
