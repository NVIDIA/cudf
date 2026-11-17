/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file writer_impl.hpp
 * @brief cuDF-IO Parquet writer class implementation header
 */

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>

#include <memory>
#include <string>
#include <vector>

namespace cudf::io::parquet::detail {

// Forward internal classes
struct aggregate_writer_metadata;

using cudf::detail::device_2dspan;
using cudf::detail::host_2dspan;
using cudf::detail::hostdevice_2dvector;

/**
 * @brief Implementation for parquet writer
 */
class writer::impl {
 public:
  /**
   * @brief Constructor with writer options.
   *
   * @param sinks data_sink's for storing dataset
   * @param options Settings for controlling behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  explicit impl(std::vector<std::unique_ptr<data_sink>> sinks,
                parquet_writer_options const& options,
                cudf::io::detail::single_write_mode mode,
                cuda::stream_ref stream);

  /**
   * @brief Constructor with chunked writer options.
   *
   * @param sinks data_sink's for storing dataset
   * @param options Settings for controlling behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  explicit impl(std::vector<std::unique_ptr<data_sink>> sinks,
                chunked_parquet_writer_options const& options,
                cudf::io::detail::single_write_mode mode,
                cuda::stream_ref stream);

  /**
   * @brief Destructor to complete any incomplete write and release resources.
   */
  ~impl();

  impl(impl const&)            = delete;
  impl& operator=(impl const&) = delete;
  impl(impl&&)                 = delete;
  impl& operator=(impl&&)      = delete;

  /**
   * @brief Initializes the states before writing.
   */
  void init_state();

  /**
   * @brief Updates writer-level statistics with data from the current table.
   *
   * @param compression_stats Optional compression statistics from the current table
   */
  void update_compression_statistics(
    std::optional<writer_compression_statistics> const& compression_stats);

  /**
   * @brief Writes a single subtable as part of a larger parquet file/table write,
   * normally used for chunked writing.
   *
   * @throws rmm::bad_alloc if there is insufficient space for temporary buffers
   *
   * @param[in] table The table information to be written
   * @param[in] partitions Optional partitions to divide the table into. If specified, must be same
   * size as number of sinks.
   */
  void write(table_view const& table, std::vector<partition_info> const& partitions);

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

 private:
  /**
   * @brief Write the intermediate Parquet data into the data sink.
   *
   * The intermediate data is generated from processing (compressing/encoding) a cuDF input table
   * by `convert_table_to_parquet_data` called in the `write()` function.
   *
   * @param updated_agg_meta The updated aggregate data after processing the input
   * @param pages Encoded pages
   * @param chunks Column chunks
   * @param global_rowgroup_base Numbers of rowgroups in each file/partition
   * @param first_rg_in_part The first rowgroup in each partition
   * @param rg_to_part A map from rowgroup to partition
   * @param[out] bounce_buffer Temporary host output buffer
   */
  void write_parquet_data_to_sink(std::unique_ptr<aggregate_writer_metadata>& updated_agg_meta,
                                  device_span<EncPage const> pages,
                                  host_2dspan<EncColumnChunk const> chunks,
                                  host_span<size_t const> global_rowgroup_base,
                                  host_span<int const> first_rg_in_part,
                                  host_span<int const> rg_to_part,
                                  host_span<uint8_t> bounce_buffer);

  // Cuda stream to be used
  cuda::stream_ref _stream;

  // Writer options.
  compression_type const _compression;
  size_t const _max_row_group_size;
  size_type const _max_row_group_rows;
  size_t const _max_page_size_bytes;
  size_type const _max_page_size_rows;
  statistics_freq const _stats_granularity;
  dictionary_policy const _dict_policy;
  size_t const _max_dictionary_size;
  std::optional<size_type> const _max_page_fragment_size;
  bool const _int96_timestamps;
  bool const _utc_timestamps;
  bool const _write_v2_headers;
  bool const _page_level_compression;
  bool const _write_arrow_schema;
  std::optional<std::vector<sorting_column>> _sorting_columns;
  int32_t const _column_index_truncate_length;
  std::vector<std::map<std::string, std::string>> const _kv_meta;  // Optional user metadata.
  cudf::io::detail::single_write_mode const
    _single_write_mode;  // Special parameter only used by `write()` to
                         // indicate that we are guaranteeing a single table
                         // write. This enables some internal optimizations.
  std::vector<std::unique_ptr<data_sink>> const _out_sink;

  // Internal states, filled during `write()` and written to sink during `write` and `close()`.
  std::unique_ptr<table_input_metadata> _table_meta;
  std::unique_ptr<aggregate_writer_metadata> _agg_meta;
  std::vector<std::size_t> _current_chunk_offset;  // To track if the last write(table) call
                                                   // completed successfully current write
                                                   // position for rowgroups/chunks.
  std::shared_ptr<writer_compression_statistics> _compression_statistics;  // Optional output
  bool _last_write_successful = false;
  bool _closed                = false;  // To track if the output has been written to sink.
};

}  // namespace cudf::io::parquet::detail
