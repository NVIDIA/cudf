/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file parquet_metadata.hpp
 * @brief Freeform APIs for Parquet metadata
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/export.hpp>

#include <span>
#include <string_view>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io {
/**
 * @addtogroup io_types
 * @{
 */

//! Parquet physical `Type`
using cudf::io::parquet::Type;

/**
 * @brief Schema of a parquet column, including the nested columns.
 */
struct parquet_column_schema {
 public:
  /**
   * @brief Default constructor
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_column_schema() : _cudf_type{data_type{type_id::EMPTY}} {}

  /**
   * @brief constructor
   *
   * @param name column name
   * @param type parquet type
   * @param children child columns (empty for non-nested types)
   * @param cudf_type cudf data type
   */
  parquet_column_schema(std::string_view name,
                        Type type,
                        std::vector<parquet_column_schema>&& children,
                        data_type cudf_type)
    : _name{name}, _type{type}, _children{std::move(children)}, _cudf_type{cudf_type}
  {
  }

  /**
   * @brief Returns parquet column name; can be empty
   *
   * @return Column name
   */
  [[nodiscard]] auto name() const { return _name; }

  /**
   * @brief Returns parquet physical type of the column.
   *
   * @return Column parquet physical type
   */
  [[nodiscard]] auto type() const { return _type; }

  /**
   * @brief Returns schemas of all child columns
   *
   * @return Children schemas
   */
  [[nodiscard]] auto const& children() const& { return _children; }

  /** @copydoc children
   * Children array is moved out of the object (rvalues only)
   *
   */
  [[nodiscard]] auto children() && { return std::move(_children); }

  /**
   * @brief Returns schema of the child with the given index
   *
   * @param idx child index
   *
   * @return Child schema
   */
  [[nodiscard]] auto const& child(int idx) const& { return children().at(idx); }

  /** @copydoc child
   * Child is moved out of the object (rvalues only)
   *
   */
  [[nodiscard]] auto child(int idx) && { return std::move(children().at(idx)); }

  /**
   * @brief Returns the number of child columns
   *
   * @return Children count
   */
  [[nodiscard]] auto num_children() const { return children().size(); }

  /**
   * @brief Returns the cudf data type for this column
   *
   * This is the resolved cudf data type mapped from the Parquet physical/logical types.
   *
   * @return cudf data type
   */
  [[nodiscard]] auto cudf_type() const { return _cudf_type; }

 private:
  std::string _name;
  // 3 types available: Physical, Converted, Logical
  Type _type;  // Physical type
  std::vector<parquet_column_schema> _children;
  data_type _cudf_type;
};

/**
 * @brief Schema of a parquet file
 */
struct parquet_schema {
 public:
  /**
   * @brief Default constructor
   *
   * This has been added since Cython requires a default constructor to create objects on stack
   */
  explicit parquet_schema() = default;

  /**
   * @brief constructor
   *
   * @param root_column_schema root column
   */
  parquet_schema(parquet_column_schema root_column_schema) : _root{std::move(root_column_schema)} {}

  /**
   * @brief Returns the schema of the struct column that contains all columns as fields
   *
   * @return Root column schema
   */
  [[nodiscard]] auto const& root() const& { return _root; }

  /** @copydoc root
   * Root column schema is moved out of the object (rvalues only)
   *
   */
  [[nodiscard]] auto root() && { return std::move(_root); }

 private:
  parquet_column_schema _root;
};

/**
 * @brief Information about content of a parquet file
 */
class parquet_metadata {
 public:
  /// Key-value metadata in the file footer
  using key_value_metadata = std::unordered_map<std::string, std::string>;
  /// Row group metadata from each RowGroup element
  using row_group_metadata = std::unordered_map<std::string, int64_t>;
  /// Column chunk metadata from each ColumnChunkMetaData element
  using column_chunk_metadata = std::unordered_map<std::string, std::vector<int64_t>>;

  /**
   * @brief Default constructor
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit parquet_metadata() = default;

  /**
   * @brief constructor
   *
   * @param schema parquet schema
   * @param num_rows number of rows
   * @param num_rowgroups total number of row groups
   * @param num_rowgroups_per_file number of row groups per file
   * @param file_metadata key-value metadata in the file footer
   * @param rg_metadata vector of maps containing metadata for each row group
   * @param column_chunk_metadata map of column names to vectors of `total_uncompressed_size`
   *                              metadata from all their column chunks
   */
  parquet_metadata(parquet_schema schema,
                   int64_t num_rows,
                   size_type num_rowgroups,
                   std::vector<size_type> num_rowgroups_per_file,
                   key_value_metadata file_metadata,
                   std::vector<row_group_metadata> rg_metadata,
                   column_chunk_metadata column_chunk_metadata)
    : _schema{std::move(schema)},
      _num_rows{num_rows},
      _num_rowgroups{num_rowgroups},
      _num_rowgroups_per_file{std::move(num_rowgroups_per_file)},
      _file_metadata{std::move(file_metadata)},
      _rowgroup_metadata{std::move(rg_metadata)},
      _column_chunk_metadata{std::move(column_chunk_metadata)}
  {
  }

  /**
   * @brief Returns the parquet schema
   *
   * @return parquet schema
   */
  [[nodiscard]] auto const& schema() const { return _schema; }

  /**
   * @brief Returns the number of rows of the root column
   *
   * If a file contains list columns, nested columns can have a different number of rows.
   *
   * @return Number of rows
   */
  [[nodiscard]] auto num_rows() const { return _num_rows; }

  /**
   * @brief Returns the total number of rowgroups
   *
   * @return Total number of row groups
   */
  [[nodiscard]] auto num_rowgroups() const { return _num_rowgroups; }

  /**
   * @brief Returns the number of rowgroups in each file
   *
   * @return Number of row groups per file
   */
  [[nodiscard]] auto const& num_rowgroups_per_file() const { return _num_rowgroups_per_file; }

  /**
   * @brief Returns the Key value metadata in the file footer
   *
   * @return Key value metadata as a map
   */
  [[nodiscard]] auto const& metadata() const { return _file_metadata; }

  /**
   * @brief Returns the row group metadata in the file footer
   *
   * @return Vector of row group metadata as maps
   */
  [[nodiscard]] auto const& rowgroup_metadata() const { return _rowgroup_metadata; }

  /**
   * @brief Returns a map of column names to vectors of `total_uncompressed_size` metadata from
   *        all their column chunks
   *
   * @return Map of column names to vectors of `total_uncompressed_size` metadata from all their
   *         column chunks
   */
  [[nodiscard]] auto const& columnchunk_metadata() const { return _column_chunk_metadata; }

 private:
  parquet_schema _schema;
  int64_t _num_rows;
  size_type _num_rowgroups;
  std::vector<size_type> _num_rowgroups_per_file;
  key_value_metadata _file_metadata;
  std::vector<row_group_metadata> _rowgroup_metadata;
  column_chunk_metadata _column_chunk_metadata;
};

/**
 * @brief Reads metadata of parquet dataset
 *
 * @ingroup io_readers
 *
 * @param src_info Dataset source information
 *
 * @return parquet_metadata with parquet schema, number of rows, number of row groups and key-value
 * metadata
 */
parquet_metadata read_parquet_metadata(source_info const& src_info);

/**
 * @brief Constructs FileMetaData objects from parquet dataset
 *
 * @ingroup io_readers
 *
 * @param sources Input `datasource` objects to read the dataset from
 *
 * @return List of FileMetaData objects, one per parquet source
 */
std::vector<parquet::FileMetaData> read_parquet_footers(
  std::span<std::unique_ptr<cudf::io::datasource> const> sources);

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
