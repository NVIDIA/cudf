/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <string>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io {

/**
 * @file
 * @brief Optional Vortex file writer.
 * @addtogroup io_writers
 * @{
 */

class vortex_writer_options_builder;

/**
 * @brief Settings for `write_vortex()`.
 *
 * The table is borrowed and must remain valid until `write_vortex()` returns.
 * Only a single local-file sink is currently supported.
 */
class vortex_writer_options {
  sink_info _sink;
  table_view _table;
  std::vector<std::string> _names;
  size_type _rows_per_chunk{16 << 20};

  friend class vortex_writer_options_builder;

  explicit vortex_writer_options(sink_info sink, table_view table)
    : _sink{std::move(sink)}, _table{table}
  {
  }

 public:
  /**
   * @brief Construct empty options, to be populated before writing.
   */
  vortex_writer_options() = default;

  /**
   * @brief Create an options builder.
   * @param sink Destination (one local file)
   * @param table GPU-resident input table
   * @return Options builder
   */
  static vortex_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /// @brief Returns the destination. @return Destination information
  [[nodiscard]] sink_info const& get_sink() const noexcept { return _sink; }
  /// @brief Returns the borrowed input table. @return Input table
  [[nodiscard]] table_view const& get_table() const noexcept { return _table; }
  /// @brief Returns column names. @return Names, or empty for generated names
  [[nodiscard]] std::vector<std::string> const& get_names() const noexcept { return _names; }
  /// @brief Returns the row-block size. @return Maximum rows per staging chunk and physical block
  [[nodiscard]] size_type get_rows_per_chunk() const noexcept { return _rows_per_chunk; }

  /// @brief Set the destination. @param sink Destination information
  void set_sink(sink_info sink) { _sink = std::move(sink); }
  /// @brief Set the borrowed input table. @param table Input table
  void set_table(table_view table) { _table = table; }
  /**
   * @brief Set unique, NUL-free column names.
   * @param names One name per column, or empty to generate `_col0`, `_col1`, etc.
   */
  void set_names(std::vector<std::string> names) { _names = std::move(names); }
  /**
   * @brief Set maximum rows per host-staging chunk and physical Vortex row block.
   *
   * This is a row bound, not a byte or total-memory bound. The original GPU input remains resident.
   * @param rows Positive row count (default: 16 Mi rows)
   */
  void set_rows_per_chunk(size_type rows) { _rows_per_chunk = rows; }
};

/** @brief Builder for `vortex_writer_options`. */
class vortex_writer_options_builder {
  vortex_writer_options _options;

 public:
  /**
   * @brief Construct a builder from a destination and input table.
   * @param sink Destination information
   * @param table Borrowed input table
   */
  explicit vortex_writer_options_builder(sink_info const& sink, table_view const& table)
    : _options{sink, table}
  {
  }

  /**
   * @brief Set column names.
   * @param names One unique, NUL-free name per column, or empty for generated names
   * @return This builder
   */
  vortex_writer_options_builder& names(std::vector<std::string> names)
  {
    _options.set_names(std::move(names));
    return *this;
  }

  /**
   * @brief Set staging and physical row-block size.
   * @param rows Positive row count
   * @return This builder
   */
  vortex_writer_options_builder& rows_per_chunk(size_type rows)
  {
    _options.set_rows_per_chunk(rows);
    return *this;
  }

  /** @brief Build the writer options. @return Writer options */
  [[nodiscard]] vortex_writer_options build() const { return _options; }
};

/**
 * @brief Write a GPU-resident table to a local Vortex file.
 *
 * Requires libcudf built with `CUDF_WITH_VORTEX=ON` (Linux, shared-library builds).
 * The current implementation requires CUDA device 0 to be current and uses Vortex's experimental
 * CUDA-flat layout. Files are tied to the pinned Vortex revision; cross-version compatibility of
 * this layout is not guaranteed.
 *
 * Supported columns are signed/unsigned integers, floating point, booleans, strings,
 * decimal32/64/128, and day-resolution timestamps. Nulls, slices and zero-row tables are supported;
 * at least one column is required. Nested, dictionary, duration and other timestamp types are not
 * supported. Only `sink_info` containing one nonempty, NUL-free local path is accepted; buffers,
 * custom sinks and remote URIs are not supported.
 *
 * Input is copied to host Arrow in row-bounded chunks (sliced strings are compacted on device),
 * then compressed and encoded on the CPU. This is not a GPU compression API. `stream` orders cuDF
 * staging work; `mr` controls cuDF device staging buffers, not host or Vortex allocations.
 * Scratch allocations inside cuDF slicing and string compaction use the current device resource.
 * No CUDA memory-pool retention policy is changed by this function.
 *
 * The function blocks until cuDF staging and file finalization complete. It overwrites an existing
 * file and does not guarantee durable storage (no fsync) or atomic replacement. A failed write can
 * leave a partial/invalid file. Input and the supplied resource must remain alive until return.
 *
 * @throws cudf::logic_error If options, types or the current device are unsupported, or Vortex
 * support was disabled at build time
 * @throws cudf::cuda_error If a CUDA operation fails
 * @throws std::runtime_error If Vortex conversion, compression or file I/O fails
 *
 * @param options Destination, borrowed table, names and chunk size
 * @param stream CUDA stream used for cuDF staging
 * @param mr Device resource used for cuDF staging buffers (not all internal scratch allocations)
 */
void write_vortex(vortex_writer_options const& options,
                  cuda::stream_ref stream           = cudf::get_default_stream(),
                  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */

}  // namespace io
}  // namespace CUDF_EXPORT cudf
