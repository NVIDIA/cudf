/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reader_impl_helpers.hpp"

#include <cudf/types.hpp>

namespace cudf::io::parquet::detail {

/**
 * @brief Struct to store file-level data that remains constant for
 * all passes/chunks in the file.
 */
struct file_intermediate_data {
  file_intermediate_data() = default;

  file_intermediate_data(file_intermediate_data const&)            = delete;
  file_intermediate_data& operator=(file_intermediate_data const&) = delete;
  file_intermediate_data(file_intermediate_data&&)                 = default;
  file_intermediate_data& operator=(file_intermediate_data&&)      = default;

  // all row groups to read
  std::vector<row_group_info> row_groups{};

  // all chunks from the selected row groups.
  std::vector<ColumnChunkDesc> chunks{};

  // an array of offsets into _file_itm_data::global_chunks. Each pair of offsets represents
  // the start/end of the chunks to be loaded for a given pass.
  std::vector<std::size_t> input_pass_row_group_offsets{};

  // start row counts per input-pass. this includes all rows in the row groups of the pass and
  // is not capped by global_skip_rows and global_num_rows.
  std::vector<std::size_t> input_pass_start_row_count{};

  // number of rows to be read from each data source
  std::vector<std::size_t> num_rows_per_source{};

  // partial sum of the number of rows per data source
  std::vector<std::size_t> exclusive_sum_num_rows_per_source{};

  size_type num_input_row_groups{0};  // total number of input row groups across all data sources

  // struct containing the number of remaining row groups after each predicate pushdown filter
  surviving_row_group_metrics surviving_row_groups;

  size_t _current_input_pass{0};  // current input pass index
  size_t _output_chunk_count{0};  // how many output chunks we have produced

  // skip_rows/num_rows values for the entire file.
  size_t global_skip_rows;
  size_t global_num_rows;

  [[nodiscard]] size_t num_passes() const
  {
    return input_pass_row_group_offsets.size() == 0 ? 0 : input_pass_row_group_offsets.size() - 1;
  }
};

/**
 * @brief Struct to identify a range of rows.
 */
struct row_range {
  size_t skip_rows;
  size_t num_rows;
};

/**
 * @brief Passes are broken down into subpasses based on temporary memory constraints.
 */
struct subpass_intermediate_data {
  subpass_intermediate_data()                                            = delete;
  subpass_intermediate_data(subpass_intermediate_data const&)            = delete;
  subpass_intermediate_data& operator=(subpass_intermediate_data const&) = delete;
  subpass_intermediate_data(subpass_intermediate_data&&)                 = default;
  subpass_intermediate_data& operator=(subpass_intermediate_data&&)      = default;
  subpass_intermediate_data(cuda::stream_ref stream)
    : decomp_page_data(0, stream),
      level_decode_data(0, stream),
      page_buf(0, stream),
      page_src_index{0, stream},
      page_string_offset_indices(0, stream),
      string_offset_buffer(0, stream),
      delta_temp_buf(0, stream),
      page_nesting_info(0, stream),
      page_nesting_decode_info(0, stream)
  {
  }

  rmm::device_buffer decomp_page_data;

  rmm::device_buffer level_decode_data;
  cudf::detail::hostdevice_span<PageInfo> pages{};

  cudf::detail::hostdevice_vector<PageInfo> page_buf;

  // for each page in the subpass, the index of our source page in the pass
  rmm::device_uvector<size_t> page_src_index;

  // For each page, the index into the column's string offset buffer
  // Used for non-dictionary, non-FLBA string columns
  rmm::device_uvector<size_t> page_string_offset_indices;

  // String offset buffer for non-dictionary, non-FLBA string columns
  // Contains pre-computed offsets into the string data. Allocated once per subpass
  // in preprocess_subpass_pages() and reused across all output chunks in the subpass.
  rmm::device_uvector<uint32_t> string_offset_buffer;

  // temporary space for DELTA_BYTE_ARRAY decoding. this only needs to live until
  // gpu::DecodeDeltaByteArray returns.
  rmm::device_uvector<uint8_t> delta_temp_buf;

  // for each column in the file (indexed by _input_columns.size())
  // the number of associated pages for this subpass
  std::vector<size_t> column_page_count{};
  cudf::detail::hostdevice_vector<PageNestingInfo> page_nesting_info;
  cudf::detail::hostdevice_vector<PageNestingDecodeInfo> page_nesting_decode_info;

  std::vector<row_range> output_chunk_read_info{};

  // skip_rows and num_rows values for this particular subpass. in absolute row indices.
  size_t skip_rows{0};
  size_t num_rows{0};
  size_t current_output_chunk{0};

  uint32_t kernel_mask{0};

  // optimization. if the single_subpass flag is set, it means we will only be doing
  // one subpass for the entire pass. this allows us to skip various pieces of work
  // during processing. notably, page_buf will not be allocated to hold a compacted
  // copy of the pages specific to the subpass.
  bool single_subpass{false};
};

/**
 * @brief Struct to store pass-level data that remains constant for a single pass.
 *
 * A pass is defined as a set of rowgroups read but not yet decompressed. This set of
 * rowgroups may represent less than all of the rowgroups to be read for the file.
 */
struct pass_intermediate_data {
  pass_intermediate_data()                                         = delete;
  pass_intermediate_data(pass_intermediate_data const&)            = delete;
  pass_intermediate_data& operator=(pass_intermediate_data const&) = delete;
  pass_intermediate_data(pass_intermediate_data&&)                 = default;
  pass_intermediate_data& operator=(pass_intermediate_data&&)      = default;
  pass_intermediate_data(cuda::stream_ref stream)
    : raw_page_data{},
      row_groups{},
      chunks(0, stream),
      pages(0, stream),
      page_offsets{0, stream},
      decomp_dict_data{0, stream},
      decomp_scratch_sizes{0, stream},
      string_offset_sizes{0, stream},
      level_decode_sizes{0, stream},
      str_dict_index{0, stream}
  {
  }

  std::vector<rmm::device_buffer> raw_page_data{};

  // rowgroup, chunk and page information for the current pass.
  std::vector<row_group_info> row_groups{};
  cudf::detail::hostdevice_vector<ColumnChunkDesc> chunks;
  cudf::detail::hostdevice_vector<PageInfo> pages;

  // offsets to each group of input pages (by column/schema, indexed by _input_columns.size())
  // so if we had 2 columns/schemas, with page keys
  //
  // 1 1 1 1 1 2 2 2
  //
  // page_offsets would be 0, 5, 8
  rmm::device_uvector<size_type> page_offsets;

  rmm::device_buffer decomp_dict_data;
  rmm::device_uvector<size_t> decomp_scratch_sizes;
  rmm::device_uvector<size_t> string_offset_sizes;
  rmm::device_uvector<size_t> level_decode_sizes;
  rmm::device_uvector<string_index_pair> str_dict_index;

  // currently active subpass
  std::unique_ptr<subpass_intermediate_data> subpass{};

  // base memory used for the pass itself (compressed data in the loaded chunks and any
  // decompressed dictionary pages)
  size_t base_mem_size{0};

  // skip_rows / num_rows for this pass.
  // NOTE: skip_rows is the absolute row index in the file.
  size_t skip_rows{0};
  size_t num_rows{0};
  // number of rows we have processed so far (out of num_rows). note that this
  // only includes the number of rows we have processed before starting the current
  // subpass. it does not get updated as a subpass iterates through output chunks.
  size_t processed_rows{0};

  int level_type_size{0};

  bool has_compressed_data{false};
};

}  // namespace cudf::io::parquet::detail
