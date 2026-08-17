/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/orc/aggregate_orc_metadata.hpp"
#include "io/orc/reader_impl_chunking.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <io/utilities/column_buffer.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace cudf::io::orc::detail {

struct reader_column_meta;

/**
 * @brief Implementation for ORC reader.
 */
class reader_impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   *
   * This constructor will call the other constructor with `chunk_read_limit` and `pass_read_limit`
   * set to `0` and `output_row_granularity` set to `DEFAULT_OUTPUT_ROW_GRANULARITY`.
   *
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit reader_impl(std::vector<std::unique_ptr<datasource>>&& sources,
                       orc_reader_options const& options,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr);

  /**
   * @copydoc cudf::io::orc::detail::chunked_reader::chunked_reader(std::size_t, std::size_t,
   * orc_reader_options const&, cuda::stream_ref, rmm::device_async_resource_ref)
   */
  explicit reader_impl(std::size_t chunk_read_limit,
                       std::size_t pass_read_limit,
                       std::vector<std::unique_ptr<datasource>>&& sources,
                       orc_reader_options const& options,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr);

  /**
   * @copydoc cudf::io::orc::detail::chunked_reader::chunked_reader(std::size_t, std::size_t,
   * size_type, orc_reader_options const&, cuda::stream_ref, rmm::device_async_resource_ref)
   */
  explicit reader_impl(std::size_t chunk_read_limit,
                       std::size_t pass_read_limit,
                       size_type output_row_granularity,
                       std::vector<std::unique_ptr<datasource>>&& sources,
                       orc_reader_options const& options,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr);

  reader_impl(reader_impl const&)            = delete;
  reader_impl& operator=(reader_impl const&) = delete;
  reader_impl(reader_impl&&)                 = delete;
  reader_impl& operator=(reader_impl&&)      = delete;

  /**
   * @copydoc cudf::io::orc::detail::reader::read
   */
  table_with_metadata read();

  /**
   * @copydoc cudf::io::chunked_orc_reader::has_next
   */
  bool has_next();

  /**
   * @copydoc cudf::io::chunked_orc_reader::read_chunk
   */
  table_with_metadata read_chunk();

 private:
  /**
   * @brief The enum indicating whether the data sources are read all at once or chunk by chunk.
   */
  enum class read_mode { READ_ALL, CHUNKED_READ };

  /**
   * @brief Perform all the necessary data preprocessing before creating an output table.
   *
   * This is the proxy to call all other data preprocessing functions, which are prerequisite
   * for generating the output.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void prepare_data(read_mode mode);

  /**
   * @brief Perform a preprocessing step on the input data sources that executes exactly once
   * for the entire duration of the reader.
   *
   * In this step, the metadata of all stripes in the data sources is parsed, and information about
   * data streams of the selected columns in all stripes are generated. If the reader has a data
   * read limit, sizes of these streams are used to split the list of all stripes into multiple
   * subsets, each of which will be loaded into memory in the `load_next_stripe_data()` step. These
   * subsets are computed such that memory usage will be kept to be around a fixed size limit.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void preprocess_file(read_mode mode);

  /**
   * @brief Load stripes from the input data sources into memory.
   *
   * If there is a data read limit, only a subset of stripes are read at a time such that
   * their total data size does not exceed a fixed size limit. Then, the data is probed to
   * estimate its uncompressed sizes, which are in turn used to split that stripe subset into
   * smaller subsets, each of which to be decompressed and decoded in the next step
   * `decompress_and_decode_stripes()`. This is to ensure that loading data from data sources
   * together with decompression and decoding will be capped around the given data read limit.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void load_next_stripe_data(read_mode mode);

  /**
   * @brief Decompress and decode stripe data in the internal buffers, and store the result into
   * an intermediate table.
   *
   * This function expects that the other preprocessing steps (`global preprocess()` and
   * `load_next_stripe_data()`) have already been done.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void decompress_and_decode_stripes(read_mode mode);

  /**
   * @brief Create the output table from the intermediate table and return it along with metadata.
   *
   * @return The output table along with columns' metadata
   */
  table_with_metadata make_output_chunk();

  /**
   * @brief Create the output table metadata storing user data in source metadata.
   *
   * @return Columns' user data to output with the table read from file
   */
  table_metadata get_meta_with_user_data();

  cuda::stream_ref const _stream;
  rmm::device_async_resource_ref const _mr;

  // Reader configs.
  struct {
    data_type timestamp_type;  // override output timestamp resolution
    bool use_index;            // enable or disable attempt to use row index for parsing
    bool use_np_dtypes;        // enable or disable the conversion to numpy-compatible dtypes
    std::vector<std::string> decimal128_columns;  // control decimals conversion
    bool ignore_timezone_in_stripe_footer;        // ignore writer timezone in strip footer

    // User specified reading rows/stripes selection.
    int64_t const skip_rows;
    std::optional<int64_t> num_read_rows;
    std::vector<std::vector<size_type>> const selected_stripes;
  } const _options;

  // Intermediate data for reading.
  std::unique_ptr<reader_column_meta> const _col_meta;  // Track of orc mapping and child details
  std::vector<std::unique_ptr<datasource>> const _sources;  // Unused but owns data for `_metadata`
  aggregate_orc_metadata _metadata;
  column_hierarchy const _selected_columns;  // Construct from `_metadata` thus declare after it
  file_intermediate_data _file_itm_data;
  chunk_read_data _chunk_read_data;

  // Intermediate data for output.
  std::unique_ptr<table_metadata> _meta_with_user_data;
  table_metadata _out_metadata;
  std::vector<std::vector<cudf::io::detail::column_buffer>> _out_buffers;

  // The default value used for subdividing the decoded table for final output.
  // Larger values will reduce the computation time but will make the output table less granular.
  // Smaller values (minimum is `1`) will increase the computation time but the output table will
  // have size closer to the given `chunk_read_limit`.
  static inline constexpr size_type DEFAULT_OUTPUT_ROW_GRANULARITY = 10'000;
};

}  // namespace cudf::io::orc::detail
