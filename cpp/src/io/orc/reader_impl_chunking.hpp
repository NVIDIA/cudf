/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/utilities/hostdevice_vector.hpp"
#include "orc_gpu.hpp"

#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <unordered_map>

namespace cudf::io::orc::detail {

/**
 * @brief Struct representing a range of of data offsets.
 */
struct range {
  std::size_t begin{0};
  std::size_t end{0};

  [[nodiscard]] auto size() const { return end - begin; }
};

/**
 * @brief Expand a range of ranges into a simple range of data.
 *
 * @param input_ranges The list of all data ranges
 * @param selected_ranges A range of ranges from `input_ranges`
 * @return The range of data span by the selected range of ranges
 */
inline range merge_selected_ranges(host_span<range const> input_ranges,
                                   range const& selected_ranges)
{
  // The first and last range.
  auto const& first_range = input_ranges[selected_ranges.begin];
  auto const& last_range  = input_ranges[selected_ranges.end - 1];

  // The range of data covered from the first to the last range.
  return {first_range.begin, last_range.end};
}

// Store information to identify where to read a chunk of data from source.
// Each read corresponds to one or more consecutive streams combined.
struct stream_data_read_info {
  uint64_t offset;         // offset in data source
  std::size_t dst_pos;     // offset to store data in memory relative to start of raw stripe data
  std::size_t length;      // data length to read
  std::size_t source_idx;  // the data source id
  std::size_t stripe_idx;  // global stripe index
  std::size_t level;       // nested level
};

/**
 * @brief Compression information for a stripe at a specific nested level.
 */
struct stripe_level_comp_info {
  std::size_t num_compressed_blocks{0};
  std::size_t num_uncompressed_blocks{0};
  std::size_t total_decomp_size{0};
  std::size_t max_uncompressed_block_size{0};
};

/**
 * @brief Struct that stores source information of an ORC streams.
 */
struct stream_source_info {
  std::size_t stripe_idx;  // global stripe id throughout all data sources
  std::size_t level;       // level of the nested column
  uint32_t orc_col_idx;    // orc column id
  StreamKind kind;         // stream kind

  struct hash {
    std::size_t operator()(stream_source_info const& id) const
    {
      auto const col_kind =
        static_cast<std::size_t>(id.orc_col_idx) | (static_cast<std::size_t>(id.kind) << 32);
      auto const hasher = std::hash<size_t>{};
      return hasher(id.stripe_idx) ^ hasher(id.level) ^ hasher(col_kind);
    }
  };
  struct equal_to {
    bool operator()(stream_source_info const& lhs, stream_source_info const& rhs) const
    {
      return lhs.stripe_idx == rhs.stripe_idx && lhs.level == rhs.level &&
             lhs.orc_col_idx == rhs.orc_col_idx && lhs.kind == rhs.kind;
    }
  };
};

/**
 * @brief Map to lookup a value from stream source.
 */
template <typename T>
using stream_source_map =
  std::unordered_map<stream_source_info, T, stream_source_info::hash, stream_source_info::equal_to>;

/**
 * @brief Struct that stores information of an ORC stream.
 */
struct orc_stream_info {
  // Data info:
  uint64_t offset;      // offset in data source
  std::size_t dst_pos;  // offset to store data in memory relative to start of raw stripe data
  std::size_t length;   // stream length to read

  // Store source of the stream in the stripe, so we can look up where this stream comes from.
  stream_source_info source;
};

/**
 * @brief Struct storing intermediate processing data loaded from data sources.
 */
struct file_intermediate_data {
  file_intermediate_data() = default;

  file_intermediate_data(file_intermediate_data const&)            = delete;
  file_intermediate_data& operator=(file_intermediate_data const&) = delete;
  file_intermediate_data(file_intermediate_data&&)                 = default;
  file_intermediate_data& operator=(file_intermediate_data&&)      = default;

  int64_t rows_to_skip;
  int64_t rows_to_read;
  std::vector<metadata::orc_stripe_info> selected_stripes;

  // Check if there is data to read.
  bool has_data() const { return rows_to_read > 0 && !selected_stripes.empty(); }

  // For each stripe, we perform a number of reads for its streams.
  // Those reads are identified by a chunk of consecutive read info stored in `data_read_info`.
  std::vector<range> stripe_data_read_ranges;

  // Identify what data to read from source.
  std::vector<stream_data_read_info> data_read_info;

  // Store the compression information for each data stream.
  stream_source_map<stripe_level_comp_info> compinfo_map;

  // Store info for each ORC stream at each nested level.
  std::vector<std::vector<orc_stream_info>> lvl_stream_info;

  // At each nested level, the streams for each stripe are stored consecutively in lvl_stream_info.
  // This is used to identify the range of streams for each stripe from that vector.
  std::vector<std::vector<range>> lvl_stripe_stream_ranges;

  // The buffers to store raw data read from disk, initialized for each reading stripe chunks.
  // After decoding, such buffers can be released.
  // This can only be implemented after chunked output is ready.
  std::vector<std::vector<rmm::device_buffer>> lvl_stripe_data;

  // Store the size of each stripe at each nested level.
  // This is used to initialize the stripe_data buffers.
  std::vector<std::vector<std::size_t>> lvl_stripe_sizes;

  // List of column data types at each nested level.
  std::vector<std::vector<data_type>> lvl_column_types;

  // List of nested type columns at each nested level.
  std::vector<std::vector<orc_column_meta>> lvl_nested_cols;

  // Table for converting timestamp columns from local to UTC time.
  std::unique_ptr<cudf::table> tz_table;

  bool global_preprocessed{false};
};

/**
 * @brief Struct collecting data necessary for chunked reading.
 */
struct chunk_read_data {
  explicit chunk_read_data(std::size_t output_size_limit_,
                           std::size_t data_read_limit_,
                           size_type output_row_granularity_)
    : chunk_read_limit{output_size_limit_},
      pass_read_limit{data_read_limit_},
      output_row_granularity{output_row_granularity_}
  {
    CUDF_EXPECTS(output_row_granularity > 0,
                 "The value of `output_row_granularity` must be positive.");
  }

  std::size_t const
    chunk_read_limit;  // maximum size (in bytes) of an output chunk, or 0 for no limit
  std::size_t const pass_read_limit;  // approximate maximum size (in bytes) used for store
                                      // intermediate data, or 0 for no limit
  size_type const output_row_granularity;

  // Memory limits for loading data and decoding are computed as
  // `*_limit_ratio * pass_read_limit`.
  // This is to maintain the total memory usage to be **around** the given `pass_read_limit`.
  // Note that sum of these limits may not be `1.0`, and their values are set empirically.
  static double constexpr load_limit_ratio{0.25};
  static double constexpr decompress_and_decode_limit_ratio{0.6};

  // Chunks of stripes that can be loaded into memory such that their data size is within the user
  // specified limit.
  std::vector<range> load_stripe_ranges;
  std::size_t curr_load_stripe_range{0};
  bool more_stripes_to_load() const { return curr_load_stripe_range < load_stripe_ranges.size(); }

  // Chunks of stripes such that their decompression size is within the user specified size limit.
  std::vector<range> decode_stripe_ranges;
  std::size_t curr_decode_stripe_range{0};
  bool more_stripes_to_decode() const
  {
    return curr_decode_stripe_range < decode_stripe_ranges.size();
  }

  // Chunk of rows in the internal decoded table to output for each `read_chunk()`.
  std::vector<range> output_table_ranges;
  std::size_t curr_output_table_range{0};
  std::unique_ptr<cudf::table> decoded_table;
  bool more_table_chunks_to_output() const
  {
    return curr_output_table_range < output_table_ranges.size();
  }

  bool has_next() const
  {
    // Only has more chunk to output if:
    return more_stripes_to_load() || more_stripes_to_decode() || more_table_chunks_to_output();
  }
};

/**
 * @brief Struct to accumulate counts and sizes of some types such as stripes or rows.
 */
struct cumulative_size {
  std::size_t count{0};
  std::size_t size_bytes{0};
};

/**
 * @brief Struct to accumulate counts, sizes, and number of rows of some types such as stripes or
 * rows in tables.
 */
struct cumulative_size_and_row : public cumulative_size {
  std::size_t num_rows{0};
};

/**
 * @brief Functor to sum up cumulative data.
 */
struct cumulative_size_plus {
  __device__ cumulative_size operator()(cumulative_size const& a, cumulative_size const& b) const
  {
    return cumulative_size{a.count + b.count, a.size_bytes + b.size_bytes};
  }

  __device__ cumulative_size_and_row operator()(cumulative_size_and_row const& a,
                                                cumulative_size_and_row const& b) const
  {
    return cumulative_size_and_row{
      a.count + b.count, a.size_bytes + b.size_bytes, a.num_rows + b.num_rows};
  }
};

/**
 * @brief Find the splits of the input data such that each split range has cumulative size less than
 * a given `size_limit`.
 *
 * Note that the given limit is just a soft limit. The function will always output ranges that
 * have at least one count, even such ranges have sizes exceed the value of `size_limit`.
 *
 * @param cumulative_sizes The input cumulative sizes to compute split ranges
 * @param total_count The total count in the entire input
 * @param size_limit The given soft limit to compute splits; must be positive
 * @return A vector of ranges as splits of the input
 */
template <typename T>
std::vector<range> find_splits(host_span<T const> cumulative_sizes,
                               std::size_t total_count,
                               std::size_t size_limit);

/**
 * @brief Function that populates descriptors for either individual streams or chunks of column
 * data, but not both.
 *
 * This function is firstly used in the global step, to gather information for streams of all
 * stripes in the data sources (when `stream_info` is present). Later on, it is used again to
 * populate column descriptors (`chunks` is present) during decompression and decoding. The two
 * steps share most of the execution path thus this function takes mutually exclusive parameters
 * `stream_info` or `chunks` depending on each use case.
 *
 * @param stripe_id The index of the current stripe, can be global index or local decoding index
 * @param level The current processing nested level
 * @param stripeinfo The pointer to current stripe's information
 * @param stripefooter The pointer to current stripe's footer
 * @param orc2gdf The mapping from ORC column ids to gdf column ids
 * @param types The schema type
 * @param use_index Whether to use the row index for parsing
 * @param apply_struct_map Indicating if this is the root level
 * @param num_dictionary_entries The number of dictionary entries
 * @param local_stream_order For retrieving 0-based orders of streams in the decoding step
 * @param stream_info The vector of streams' information
 * @param chunks The vector of column descriptors
 * @return The number of bytes in the gathered streams
 */
std::size_t gather_stream_info_and_column_desc(
  std::size_t stripe_id,
  std::size_t level,
  StripeInformation const* stripeinfo,
  StripeFooter const* stripefooter,
  host_span<size_type const> orc2gdf,
  host_span<SchemaType const> types,
  bool use_index,
  bool apply_struct_map,
  int64_t* num_dictionary_entries,
  std::size_t* local_stream_order,
  std::vector<orc_stream_info>* stream_info,
  cudf::detail::hostdevice_2dvector<column_desc>* chunks);

}  // namespace cudf::io::orc::detail
