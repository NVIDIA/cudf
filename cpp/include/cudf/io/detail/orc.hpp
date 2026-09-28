/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/detail/utils.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>
#include <string>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io {

// Forward declaration
class CUDF_EXPORT orc_reader_options;
class CUDF_EXPORT orc_writer_options;
class CUDF_EXPORT chunked_orc_writer_options;

namespace orc::detail {

// Forward declaration of the internal reader class
class reader_impl;

/**
 * @brief Class to read ORC dataset data into columns.
 */
class reader {
 private:
  std::unique_ptr<reader_impl> _impl;

 public:
  /**
   * @brief Constructor from an array of datasources
   *
   * @param sources Input `datasource` objects to read the dataset from
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                  orc_reader_options const& options,
                  cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr);

  /**
   * @brief Destructor explicitly declared to avoid inlining in header
   */
  ~reader();

  /**
   * @brief Reads the entire dataset.
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read();
};

/**
 * @brief The reader class that supports iterative reading from an array of data sources.
 */
class chunked_reader {
 private:
  std::unique_ptr<reader_impl> _impl;

 public:
  /**
   * @copydoc cudf::io::chunked_orc_reader::chunked_orc_reader(std::size_t, std::size_t, size_type,
   * orc_reader_options const&, cuda::stream_ref, rmm::device_async_resource_ref)
   *
   * @param sources Input `datasource` objects to read the dataset from
   */
  explicit chunked_reader(std::size_t chunk_read_limit,
                          std::size_t pass_read_limit,
                          size_type output_row_granularity,
                          std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                          orc_reader_options const& options,
                          cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr);
  /**
   * @copydoc cudf::io::chunked_orc_reader::chunked_orc_reader(std::size_t, std::size_t,
   * orc_reader_options const&, cuda::stream_ref, rmm::device_async_resource_ref)
   *
   * @param sources Input `datasource` objects to read the dataset from
   */
  explicit chunked_reader(std::size_t chunk_read_limit,
                          std::size_t pass_read_limit,
                          std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                          orc_reader_options const& options,
                          cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr);

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header.
   *
   * Since the declaration of the internal `_impl` object does not exist in this header, this
   * destructor needs to be defined in a separate source file which can access to that object's
   * declaration.
   */
  ~chunked_reader();

  /**
   * @copydoc cudf::io::chunked_orc_reader::has_next
   */
  [[nodiscard]] bool has_next() const;

  /**
   * @copydoc cudf::io::chunked_orc_reader::read_chunk
   */
  [[nodiscard]] table_with_metadata read_chunk() const;
};

/**
 * @brief Class to write ORC dataset data into columns.
 */
class writer {
 private:
  class impl;
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor for output to a file.
   *
   * @param sink The data sink to write the data to
   * @param options Settings for controlling writing behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  explicit writer(std::unique_ptr<cudf::io::data_sink> sink,
                  orc_writer_options const& options,
                  cudf::io::detail::single_write_mode mode,
                  cuda::stream_ref stream);

  /**
   * @brief Constructor with chunked writer options.
   *
   * @param sink The data sink to write the data to
   * @param options Settings for controlling writing behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  explicit writer(std::unique_ptr<cudf::io::data_sink> sink,
                  chunked_orc_writer_options const& options,
                  cudf::io::detail::single_write_mode mode,
                  cuda::stream_ref stream);

  /**
   * @brief Destructor explicitly declared to avoid inlining in header
   */
  ~writer();

  /**
   * @brief Writes a single subtable as part of a larger ORC file/table write.
   *
   * @param[in] table The table information to be written
   */
  void write(table_view const& table);

  /**
   * @brief Finishes the chunked/streamed write process.
   */
  void close();
};

}  // namespace orc::detail
}  // namespace io
}  // namespace CUDF_EXPORT cudf
