/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/utilities/hostdevice_vector.hpp"
#include "orc.hpp"
#include "orc_gpu.hpp"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/host_vector.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudf::io::orc::detail {
// Forward internal classes
class orc_table_view;

using namespace cudf::io::detail;
using cudf::detail::device_2dspan;
using cudf::detail::host_2dspan;
using cudf::detail::hostdevice_2dvector;

/**
 * @brief Indices of rowgroups contained in a stripe.
 *
 * Provides a container-like interface to iterate over rowgroup indices.
 */
struct stripe_rowgroups {
  size_type id;     // stripe id
  size_type first;  // first rowgroup in the stripe
  size_type size;   // number of rowgroups in the stripe
  [[nodiscard]] auto cbegin() const { return cuda::counting_iterator{first}; }
  [[nodiscard]] auto cend() const { return cuda::counting_iterator{first + size}; }
};

/**
 * @brief Holds the sizes of encoded elements of decimal columns.
 */
struct encoder_decimal_info {
  std::map<uint32_t, rmm::device_uvector<uint32_t>>
    elem_sizes;  ///< Column index -> per-element size map
  std::map<uint32_t, cudf::detail::host_vector<uint32_t>>
    rg_sizes;  ///< Column index -> per-rowgroup size map
};

/**
 * @brief List of per-column ORC streams.
 *
 * Provides interface to calculate their offsets.
 */
class orc_streams {
 public:
  orc_streams(std::vector<Stream> streams, std::vector<int32_t> ids, std::vector<TypeKind> types)
    : streams{std::move(streams)}, ids{std::move(ids)}, types{std::move(types)}
  {
  }
  Stream const& operator[](int idx) const { return streams[idx]; }
  Stream& operator[](int idx) { return streams[idx]; }
  auto id(int idx) const { return ids[idx]; }
  auto& id(int idx) { return ids[idx]; }
  auto type(int idx) const { return types[idx]; }
  auto size() const { return streams.size(); }

  operator std::vector<Stream> const&() const { return streams; }

 private:
  std::vector<Stream> streams;
  std::vector<int32_t> ids;
  std::vector<TypeKind> types;
};
/**
 * @brief Description of how the ORC file is segmented into stripes and rowgroups.
 */
struct file_segmentation {
  hostdevice_2dvector<rowgroup_rows> rowgroups;
  cudf::detail::host_vector<stripe_rowgroups> stripes;

  auto num_rowgroups() const noexcept { return rowgroups.size().first; }
  auto num_stripes() const noexcept { return stripes.size(); }
};

/**
 * @brief ORC per-chunk streams of encoded data.
 *
 * The encoded bytes of each (stripe, stream) pair occupy an aligned byte range (extent) within one
 * of the arenas below. Streams with size that is not known in advance are written into the
 * transient arena, which is freed as soon as gathering completes.
 */
struct encoded_data {
  rmm::device_uvector<uint8_t> persistent_buffer;       // extents that may be read in place
  rmm::device_uvector<uint8_t> transient_buffer;        // extents always copied out by the gather
  rmm::device_uvector<uint8_t> gathered_buffer;         // arena for gather_stripes output
  std::vector<std::vector<device_span<uint8_t>>> data;  // [stripe][strm_id] views
  hostdevice_2dvector<encoder_chunk_streams> streams;   // streams of encoded data, per chunk
};

/**
 * @brief Dictionary data for string columns and their device views, per column.
 */
struct string_dictionaries {
  std::vector<rmm::device_uvector<uint32_t>> data;
  std::vector<rmm::device_uvector<uint32_t>> index;
  rmm::device_uvector<device_span<uint32_t>> d_data_view;
  rmm::device_uvector<device_span<uint32_t>> d_index_view;
  // Dictionaries are currently disabled for columns with a rowgroup larger than 2^15
  thrust::host_vector<bool> dictionary_enabled;
};

/**
 * @brief Maximum size of stripes in the output file.
 */
struct stripe_size_limits {
  size_t bytes;
  size_type rows;
};

/**
 * @brief Statistics data stored between calls to write for chunked writes
 *
 */
struct intermediate_statistics {
  explicit intermediate_statistics(cuda::stream_ref stream) : stripe_stat_chunks(0, stream) {}

  intermediate_statistics(orc_table_view const& table, cuda::stream_ref stream);

  intermediate_statistics(std::vector<col_stats_blob> rb,
                          rmm::device_uvector<statistics_chunk> sc,
                          cudf::detail::hostdevice_vector<statistics_merge_group> smg,
                          std::vector<statistics_dtype> sdt,
                          std::vector<data_type> sct)
    : rowgroup_blobs(std::move(rb)),
      stripe_stat_chunks(std::move(sc)),
      stripe_stat_merge(std::move(smg)),
      stats_dtypes(std::move(sdt)),
      col_types(std::move(sct))
  {
  }

  // blobs for the rowgroups. Not persisted
  std::vector<col_stats_blob> rowgroup_blobs;

  rmm::device_uvector<statistics_chunk> stripe_stat_chunks;
  cudf::detail::hostdevice_vector<statistics_merge_group> stripe_stat_merge;
  std::vector<statistics_dtype> stats_dtypes;
  std::vector<data_type> col_types;
};

/**
 * @brief used for chunked writes to persist data between calls to write.
 *
 */
struct persisted_statistics {
  void clear()
  {
    stripe_stat_chunks.clear();
    stripe_stat_merge.clear();
    string_pools.clear();
    stats_dtypes.clear();
    col_types.clear();
    num_rows = 0;
  }

  void persist(uint64_t num_table_rows,
               single_write_mode write_mode,
               intermediate_statistics&& intermediate_stats,
               cuda::stream_ref stream);

  std::vector<rmm::device_uvector<statistics_chunk>> stripe_stat_chunks;
  std::vector<cudf::detail::hostdevice_vector<statistics_merge_group>> stripe_stat_merge;
  std::vector<rmm::device_uvector<char>> string_pools;
  std::vector<statistics_dtype> stats_dtypes;
  std::vector<data_type> col_types;
  uint64_t num_rows = 0;
};

/**
 * @brief Protobuf encoded statistics created at file close
 *
 */
struct encoded_footer_statistics {
  std::vector<col_stats_blob> stripe_level;
  std::vector<col_stats_blob> file_level;
};

enum class writer_state {
  NO_DATA_WRITTEN,  // No table data has been written to the sink; if the writer is closed or
                    // destroyed in this state, it should not write the footer.
  DATA_WRITTEN,     // At least one table has been written to the sink; when the writer is closed,
                    // it should write the footer.
  CLOSED            // Writer has been closed; no further writes are allowed.
};

/**
 * @brief Implementation for ORC writer
 */
class writer::impl {
  // ORC datasets start with a 3 byte header
  static constexpr char const* MAGIC = "ORC";

 public:
  /**
   * @brief Constructor with writer options.
   *
   * @param sink Output sink
   * @param options Settings for controlling behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  explicit impl(std::unique_ptr<data_sink> sink,
                orc_writer_options const& options,
                single_write_mode mode,
                cuda::stream_ref stream);

  /**
   * @brief Constructor with chunked writer options.
   *
   * @param sink Output sink
   * @param options Settings for controlling behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  explicit impl(std::unique_ptr<data_sink> sink,
                chunked_orc_writer_options const& options,
                single_write_mode mode,
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
   * @brief Writes a single subtable as part of a larger ORC file/table write.
   *
   * @param table The table information to be written
   */
  void write(table_view const& table);

  /**
   * @brief Finishes the chunked/streamed write process.
   */
  void close();

 private:
  /**
   * @brief Write the intermediate ORC data into the data sink.
   *
   * The intermediate data is generated from processing (compressing/encoding) an cuDF input table
   * by `convert_table_to_orc_data` called in the `write()` function.
   *
   * @param[in] enc_data ORC per-chunk streams of encoded data
   * @param[in] segmentation Description of how the ORC file is segmented into stripes and rowgroups
   * @param[in] orc_table Non-owning view of a cuDF table that includes ORC-related information
   * @param[in] compressed_data Compressed stream data
   * @param[in] comp_results Status of data compression
   * @param[in] strm_descs List of stream descriptors
   * @param[in] rg_stats row group level statistics
   * @param[in,out] streams List of stream descriptors
   * @param[in,out] stripes List of stripe description
   * @param[out] bounce_buffer Temporary host output buffer
   */
  void write_orc_data_to_sink(encoded_data const& enc_data,
                              file_segmentation const& segmentation,
                              orc_table_view const& orc_table,
                              device_span<uint8_t const> compressed_data,
                              host_span<codec_exec_result const> comp_results,
                              host_2dspan<stripe_stream const> strm_descs,
                              host_span<col_stats_blob const> rg_stats,
                              orc_streams& streams,
                              host_span<StripeInformation> stripes,
                              host_span<uint8_t> bounce_buffer);

  /**
   * @brief Add the processed table data into the internal file footer.
   *
   * @param orc_table Non-owning view of a cuDF table that includes ORC-related information
   * @param stripes List of stripe description
   */
  void add_table_to_footer_data(orc_table_view const& orc_table,
                                std::vector<StripeInformation>& stripes);

  /**
   * @brief Update writer-level statistics with data from the current table.
   *
   * @param num_rows Number of rows in the current table
   * @param single_table_stats Statistics data from the current table
   * @param compression_stats Compression statistics from the current table
   */
  void update_statistics(size_type num_rows,
                         intermediate_statistics&& single_table_stats,
                         std::optional<writer_compression_statistics> const& compression_stats);

 private:
  // CUDA stream.
  cuda::stream_ref const _stream;

  // Writer options.
  stripe_size_limits const _max_stripe_size;
  size_type const _row_index_stride;
  compression_type const _compression;
  size_t const _compression_blocksize;
  std::shared_ptr<writer_compression_statistics> _compression_statistics;  // Optional output
  statistics_freq const _stats_freq;
  bool const _sort_dictionaries;
  single_write_mode const _single_write_mode;  // Special parameter only used by `write()` to
                                               // indicate that we are guaranteeing a single table
                                               // write. This enables some internal optimizations.
  std::map<std::string, std::string> const _kv_meta;  // Optional user metadata.
  std::unique_ptr<data_sink> const _out_sink;

  // Debug parameter---currently not yet supported to be user-specified.
  static bool constexpr _enable_dictionary = true;

  // Internal states, filled during `write()` and written to sink during `write` and `close()`.
  std::unique_ptr<table_input_metadata> _table_meta;
  Footer _footer;
  Metadata _orc_meta;
  persisted_statistics _persisted_stripe_statistics;  // Statistics data saved between calls.
  writer_state _state = writer_state::NO_DATA_WRITTEN;
};

}  // namespace cudf::io::orc::detail
