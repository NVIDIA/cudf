/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/packed_types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <utility>

namespace CUDF_EXPORT cudf {
namespace io::experimental {

/**
 * @file
 * @brief CudfTable binary format APIs for serialization and deserialization
 */

/**
 * @addtogroup io_writers
 * @{
 */

class cudftable_writer_options_builder;

/**
 * @brief Settings for `write_cudftable()`.
 */
class cudftable_writer_options {
 public:
  /// Default uncompressed block size (in bytes) used when compression is enabled.
  static constexpr uint32_t default_block_size = 256 * 1024;

 private:
  sink_info _sink;
  table_view _table;
  compression_type _compression = compression_type::NONE;
  uint32_t _block_size          = default_block_size;

  friend cudftable_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit cudftable_writer_options(sink_info sink, table_view table)
    : _sink(std::move(sink)), _table(std::move(table))
  {
  }

 public:
  /**
   * @brief Create builder to create `cudftable_writer_options`.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build cudftable_writer_options
   */
  static cudftable_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Returns sink used for writer output.
   *
   * @return sink used for writer output
   */
  [[nodiscard]] sink_info const& get_sink() const noexcept { return _sink; }

  /**
   * @brief Returns table that would be written to output.
   *
   * @return Table that would be written to output
   */
  [[nodiscard]] table_view const& get_table() const noexcept { return _table; }

  /**
   * @brief Returns compression type used for block compression.
   *
   * `compression_type::NONE` writes blocks uncompressed.
   *
   * @return Compression type
   */
  [[nodiscard]] compression_type get_compression() const noexcept { return _compression; }

  /**
   * @brief Returns the uncompressed block size in bytes for block compression.
   *
   * Has no effect when compression is `compression_type::NONE`.
   *
   * @return Block size in bytes
   */
  [[nodiscard]] uint32_t get_block_size() const noexcept { return _block_size; }

  /**
   * @brief Sets sink info.
   *
   * @param sink The sink info.
   */
  void set_sink(sink_info sink) { _sink = std::move(sink); }

  /**
   * @brief Sets the compression type used for block compression.
   *
   * `compression_type::NONE` writes blocks uncompressed.
   *
   * @param compression The compression type to use
   */
  void set_compression(compression_type compression) { _compression = compression; }

  /**
   * @brief Sets the uncompressed block size in bytes for block compression.
   *
   * Must be greater than zero when compression is enabled. Is ignored when
   * compression is `compression_type::NONE`; supplying a non-default value
   * together with `NONE` will emit a warning at write time.
   *
   * @param block_size Block size in bytes
   */
  void set_block_size(uint32_t block_size) { _block_size = block_size; }
};

/**
 * @brief Class to build `cudftable_writer_options`.
 */
class cudftable_writer_options_builder {
 public:
  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit cudftable_writer_options_builder(sink_info const& sink, table_view const& table)
    : _options(sink, table)
  {
  }

  /**
   * @brief Sets compression type.
   *
   * @param comp The compression type to use
   * @return this for chaining
   */
  cudftable_writer_options_builder& compression(compression_type comp)
  {
    _options._compression = comp;
    return *this;
  }

  /**
   * @brief Sets the uncompressed block size in bytes for block compression.
   *
   * Must be greater than zero when compression is enabled. Is ignored when
   * compression is `compression_type::NONE`; supplying a non-default value
   * together with `NONE` will emit a warning at write time.
   *
   * @param size Block size in bytes
   * @return this for chaining
   */
  cudftable_writer_options_builder& block_size(uint32_t size)
  {
    _options._block_size = size;
    return *this;
  }

  /**
   * @brief Build `cudftable_writer_options`.
   *
   * @return The constructed `cudftable_writer_options` object
   */
  [[nodiscard]] cudftable_writer_options build() const { return _options; }

 private:
  cudftable_writer_options _options;
};

/** @} */  // end of io_writers group

/**
 * @addtogroup io_readers
 * @{
 */

class cudftable_reader_options_builder;

/**
 * @brief Settings for `read_cudftable()`.
 */
class cudftable_reader_options {
  source_info _source;

  friend cudftable_reader_options_builder;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read cudftable file
   */
  explicit cudftable_reader_options(source_info src) : _source{std::move(src)} {}

 public:
  /**
   * @brief Creates a `cudftable_reader_options_builder` to build `cudftable_reader_options`.
   *
   * @param src Source information to read cudftable file
   * @return Builder to build reader options
   */
  static cudftable_reader_options_builder builder(source_info src = source_info{});

  /**
   * @brief Returns source info.
   *
   * @return Source info
   */
  [[nodiscard]] source_info const& get_source() const noexcept { return _source; }

  /**
   * @brief Sets source info.
   *
   * @param src The source info.
   */
  void set_source(source_info src) { _source = std::move(src); }
};

/**
 * @brief Class to build `cudftable_reader_options`.
 */
class cudftable_reader_options_builder {
 public:
  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read cudftable file
   */
  explicit cudftable_reader_options_builder(source_info src) : _options(std::move(src)) {}

  /**
   * @brief Build `cudftable_reader_options`.
   *
   * @return The constructed `cudftable_reader_options` object
   */
  [[nodiscard]] cudftable_reader_options build() const { return _options; }

 private:
  cudftable_reader_options _options;
};

/** @} */  // end of io_readers group

/**
 * @addtogroup io_writers
 * @{
 */

/**
 * @brief Write a table using the CudfTable binary format.
 *
 * This function uses `cudf::pack` to serialize a table into a contiguous format
 * and writes it to the specified sink. The on-disk layout is a single header
 * describing the compression type and block size, followed by the packed
 * metadata, a block index, and the (optionally compressed) data blocks. When
 * `options.get_compression() == compression_type::NONE`, the data is written
 * as a single uncompressed block.
 *
 * @param options Options specifying the sink, table, compression type and
 *                block size
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void write_cudftable(cudftable_writer_options const& options,
                     rmm::cuda_stream_view stream = cudf::get_default_stream());

/** @} */  // end of group
/**
 * @addtogroup io_readers
 * @{
 */

/**
 * @brief Read a table in CudfTable binary format.
 *
 * This function reads the header from the datasource, validates the format,
 * and uses `cudf::unpack` to deserialize the table.
 *
 * Returns a `packed_table` containing a `table_view` and the underlying `packed_columns`
 * data. After reading the data from the source, the unpacking operation creates views
 * without copying - the `table_view` points directly into the contiguous memory buffers
 * owned by `packed_columns`.
 *
 * It is the caller's responsibility to ensure the `table_view` does not outlive
 * the `packed_columns` data.
 *
 * @param options Options specifying the source to read from
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr An optional memory resource to use for all device allocations
 * @return A packed_table containing the deserialized table view and its backing data
 */
packed_table read_cudftable(
  cudftable_reader_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace io::experimental
}  // namespace CUDF_EXPORT cudf
