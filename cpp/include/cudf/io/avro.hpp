/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "types.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <string>
#include <vector>

/**
 * @file
 * @brief APIs for reading Avro files into a table.
 */

namespace CUDF_EXPORT cudf {
namespace io {
/**
 * @addtogroup io_readers
 * @{
 */

class avro_reader_options_builder;

/**
 * @brief Settings to use for `read_avro()`.
 */
class avro_reader_options {
  source_info _source;

  // Names of column to read; empty is all
  std::vector<std::string> _columns;

  // Rows to skip from the start;
  size_type _skip_rows = 0;
  // Rows to read; -1 is all
  size_type _num_rows = -1;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read avro file
   */
  explicit avro_reader_options(source_info src) : _source{std::move(src)} {}

  friend avro_reader_options_builder;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  avro_reader_options() = default;

  /**
   * @brief Returns source info.
   *
   * @return Source info
   */
  [[nodiscard]] source_info const& get_source() const { return _source; }

  /**
   * @brief Returns names of the columns to be read.
   *
   * @return Names of the columns to be read
   */
  [[nodiscard]] std::vector<std::string> get_columns() const { return _columns; }

  /**
   * @brief Returns number of rows to skip from the start.
   *
   * @return Number of rows to skip from the start
   */
  [[nodiscard]] size_type get_skip_rows() const { return _skip_rows; }

  /**
   * @brief Returns number of rows to read.
   *
   * @return Number of rows to read
   */
  [[nodiscard]] size_type get_num_rows() const { return _num_rows; }

  /**
   * @brief Sets source info.
   *
   * @param src The source info.
   */
  void set_source(source_info src) { _source = std::move(src); }

  /**
   * @brief Set names of the column to be read.
   *
   * @param col_names Vector of column names
   */
  void set_columns(std::vector<std::string> col_names) { _columns = std::move(col_names); }

  /**
   * @brief Sets number of rows to skip.
   *
   * @param val Number of rows to skip from start
   */
  void set_skip_rows(size_type val) { _skip_rows = val; }

  /**
   * @brief Sets number of rows to read.
   *
   * @param val Number of rows to read after skip
   */
  void set_num_rows(size_type val) { _num_rows = val; }

  /**
   * @brief create avro_reader_options_builder which will build avro_reader_options.
   *
   * @param src source information used to read avro file
   * @returns builder to build reader options
   */
  static avro_reader_options_builder builder(source_info src);
};

/**
 * @brief Builder to build options for `read_avro()`.
 */
class avro_reader_options_builder {
  avro_reader_options options;

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  avro_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src The source information used to read avro file
   */
  explicit avro_reader_options_builder(source_info src) : options{std::move(src)} {}

  /**
   * @brief Set names of the column to be read.
   *
   * @param col_names Vector of column names
   * @return this for chaining
   */
  avro_reader_options_builder& columns(std::vector<std::string> col_names)
  {
    options.set_columns(std::move(col_names));
    return *this;
  }

  /**
   * @brief Sets number of rows to skip.
   *
   * @param val Number of rows to skip from start
   * @return this for chaining
   */
  avro_reader_options_builder& skip_rows(size_type val)
  {
    options.set_skip_rows(val);
    return *this;
  }

  /**
   * @brief Sets number of rows to read.
   *
   * @param val Number of rows to read after skip
   * @return this for chaining
   */
  avro_reader_options_builder& num_rows(size_type val)
  {
    options.set_num_rows(val);
    return *this;
  }

  /**
   * @brief move avro_reader_options member once it's built.
   */
  operator avro_reader_options&&() { return std::move(options); }

  /**
   * @brief move avro_reader_options member once it's built.
   *
   * This has been added since Cython does not support overloading of conversion operators.
   *
   * @return Built `avro_reader_options` object's r-value reference
   */
  avro_reader_options&& build() { return std::move(options); }
};

/**
 * @brief Reads an Avro dataset into a set of columns.
 *
 * The following code snippet demonstrates how to read a dataset from a file:
 * @code
 *  auto source  = cudf::io::source_info("dataset.avro");
 *  auto options = cudf::io::avro_reader_options::builder(source);
 *  auto result  = cudf::io::read_avro(options);
 * @endcode
 *
 * @param options Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * table_with_metadata
 *
 * @return The set of columns along with metadata
 */
table_with_metadata read_avro(
  avro_reader_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
