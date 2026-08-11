/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file parquet.hpp
 */

#pragma once

#include <cudf/io/detail/utils.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <span>
#include <string>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io {

// Forward declaration
class CUDF_EXPORT parquet_reader_options;
class CUDF_EXPORT parquet_writer_options;
class CUDF_EXPORT chunked_parquet_writer_options;

namespace parquet::detail {

// Forward declaration of the implementation class
class reader_impl;

/**
 * @brief Class to read Parquet dataset data into columns.
 */
class reader {
 protected:
  std::unique_ptr<reader_impl> _impl;

  /**
   * @brief Default constructor, needed for subclassing.
   */
  reader();

 public:
  /**
   * @brief Constructor from an array of datasources
   *
   * @param sources Input `datasource` objects to read the dataset from
   * @param parquet_metadatas Pre-materialized Parquet file metadata(s). Read from sources if empty
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                  std::vector<FileMetaData>&& parquet_metadatas,
                  parquet_reader_options const& options,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr);

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  virtual ~reader();

  /**
   * @brief Reads the dataset as per given options.
   *
   * @return The set of columns along with table metadata
   */
  table_with_metadata read();
};

/**
 * @brief The reader class that supports iterative reading of a given file.
 *
 * This class intentionally subclasses the `reader` class with private inheritance to hide the
 * `reader::read()` API. As such, only chunked reading APIs are supported.
 */
class chunked_reader : private reader {
 public:
  /**
   * @brief Constructor from an output size memory limit and an input size memory limit and an array
   * of data sources with reader options.
   *
   * The typical usage should be similar to this:
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
   * If `chunk_read_limit == 0` (i.e., no output limit), and `pass_read_limit == 0` (no input
   * temporary memory size limit), a call to `read_chunk()` will read the whole file (at most 2
   * billion rows at once) and return corresponding table(s).
   *
   * The chunk_read_limit parameter controls the size of the output chunks produces.  If the user
   * specifies 100 MB of data, the reader will attempt to return chunks containing tables that have
   * a total bytes size (over all columns) of 100 MB or less.  This is a soft limit and the code
   * will not fail if it cannot satisfy the limit.  It will make a best-effort attempt only.
   *
   * The pass_read_limit parameter controls how much temporary memory is used in the process of
   * decoding the file.  The primary contributor to this memory usage is the uncompressed size of
   * the data read out of the file and the decompressed (but not yet decoded) size of the data. The
   * granularity of a given pass is at the row group level. It will not attempt to read at the sub
   * row-group level.
   *
   * Combined, the way to visualize passes and chunks is as follows:
   *
   * @code{.pseudo}
   * for(each pass){
   *    for(each output chunk within a pass){
   *       return a table that fits within the output chunk limit
   *    }
   *  }
   * @endcode
   *
   * With a pass_read_limit of `0` you are simply saying you have one pass that reads up to 2
   * billion rows (or the entire file if smaller) as normal.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   * or `0` if there is no limit
   * @param pass_read_limit Limit on total amount of memory used for temporary computations during
   * loading, or `0` if there is no limit
   * @param sources Input `datasource` objects to read the dataset from
   * @param parquet_metadatas Pre-materialized Parquet file metadata(s). Read from sources if empty
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit chunked_reader(std::size_t chunk_read_limit,
                          std::size_t pass_read_limit,
                          std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
                          std::vector<parquet::FileMetaData>&& parquet_metadatas,
                          parquet_reader_options const& options,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr);

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header.
   *
   * Since the declaration of the internal `_impl` object does not exist in this header, this
   * destructor needs to be defined in a separate source file which can access to that object's
   * declaration.
   */
  ~chunked_reader() override;

  /**
   * @copydoc cudf::io::chunked_parquet_reader::has_next
   */
  [[nodiscard]] bool has_next() const;

  /**
   * @copydoc cudf::io::chunked_parquet_reader::read_chunk
   */
  [[nodiscard]] table_with_metadata read_chunk() const;
};

/**
 * @brief Class to write parquet dataset data into columns.
 */
class writer {
 private:
  class impl;
  std::unique_ptr<impl> _impl;

 public:
  /**
   * @brief Constructor for output to a file.
   *
   * @param sinks The data sinks to write the data to
   * @param options Settings for controlling writing behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  explicit writer(std::vector<std::unique_ptr<data_sink>> sinks,
                  parquet_writer_options const& options,
                  cudf::io::detail::single_write_mode mode,
                  rmm::cuda_stream_view stream);

  /**
   * @brief Constructor for writer to handle chunked parquet options.
   *
   * @param sinks The data sinks to write the data to
   * @param options Settings for controlling writing behavior for chunked writer
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list
   */
  explicit writer(std::vector<std::unique_ptr<data_sink>> sinks,
                  chunked_parquet_writer_options const& options,
                  cudf::io::detail::single_write_mode mode,
                  rmm::cuda_stream_view stream);

  /**
   * @brief Destructor explicitly-declared to avoid inlined in header
   */
  ~writer();

  /**
   * @brief Writes a single subtable as part of a larger parquet file/table write.
   *
   * @throws rmm::bad_alloc if there is insufficient space for temporary buffers
   *
   * @param[in] table The table information to be written
   * @param[in] partitions Optional partitions to divide the table into. If specified, must be same
   * size as number of sinks.
   */
  void write(table_view const& table, std::vector<partition_info> const& partitions = {});

  /**
   * @brief Finishes the chunked/streamed write process.
   *
   * @param[in] column_chunks_file_path Column chunks file path to be set in the raw output
   * metadata
   *
   * @return A parquet-compatible blob that contains the file header and footer metadata. If
   * `column_chunks_file_path` is non-empty, the output metadata blob will also have row group file
   * paths set.
   */
  std::unique_ptr<std::vector<uint8_t>> close(
    std::vector<std::string> const& column_chunks_file_path = {});

  /**
   * @brief Merges multiple metadata blobs returned by write_all into a single metadata blob
   *
   * @param[in] metadata_list List of input file metadata
   * @return A parquet-compatible blob that contains the data for all rowgroups in the list
   */
  static std::unique_ptr<std::vector<uint8_t>> merge_row_group_metadata(
    std::vector<std::unique_ptr<std::vector<uint8_t>>> const& metadata_list);
};

/**
 * @brief Reads metadata of parquet dataset.
 *
 * @param sources Dataset sources to read from
 *
 * @return parquet_metadata with parquet schema, number of rows, number of row groups and key-value
 * metadata.
 */
parquet_metadata read_parquet_metadata(host_span<std::unique_ptr<datasource> const> sources);

/**
 * @brief Constructs FileMetaData objects from parquet dataset
 *
 * @param sources Input `datasource` objects to read the dataset from
 *
 * @return List of FileMetaData objects, one per parquet source
 */
std::vector<parquet::FileMetaData> read_parquet_footers(
  std::span<std::unique_ptr<datasource> const> sources);

}  // namespace parquet::detail
}  // namespace io
}  // namespace CUDF_EXPORT cudf
