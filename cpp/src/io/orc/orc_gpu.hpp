/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/statistics/statistics.cuh"
#include "orc.hpp"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/timezone.cuh>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/orc_types.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_map.cuh>

namespace cudf::io::orc::detail {

using cudf::detail::device_2dspan;
using cudf::detail::host_2dspan;

using key_type    = size_type;
using mapped_type = size_type;
using slot_type   = cuco::pair<key_type, mapped_type>;
auto constexpr map_cg_size =
  1;  ///< A CUDA Cooperative Group of 1 thread (set for best performance) to handle each subset.
      ///< Note: Adjust insert and find loops to use `cg::tile<map_cg_size>` if increasing this.
auto constexpr bucket_size =
  1;  ///< Number of concurrent slots (set for best performance) handled by each thread.
auto constexpr occupancy_factor = 1.43f;  ///< cuCollections suggests using a hash map of size
                                          ///< N * (1/0.7) = 1.43 to target a 70% occupancy factor.
using storage_type     = cuco::bucket_storage<slot_type,
                                              bucket_size,
                                              cuco::extent<std::size_t>,
                                              rmm::mr::polymorphic_allocator<char>>;
using storage_ref_type = typename storage_type::ref_type;

auto constexpr KEY_SENTINEL   = size_type{-1};
auto constexpr VALUE_SENTINEL = size_type{-1};

struct compressed_stream_info {
  compressed_stream_info() = default;
  explicit constexpr compressed_stream_info(uint8_t const* compressed_data_,
                                            size_t compressed_size_)
    : compressed_data(compressed_data_), compressed_data_size(compressed_size_)
  {
  }
  uint8_t const* compressed_data{};  // [in] base ptr to compressed stream data
  uint8_t*
    uncompressed_data{};  // [in] base ptr to uncompressed stream data or NULL if not known yet
  size_t compressed_data_size{};             // [in] compressed data size for this stream
  device_span<uint8_t const>* dec_in_ctl{};  // [in] input buffer to decompress
  device_span<uint8_t>* dec_out_ctl{};       // [in] output buffer to decompress into
  device_span<cudf::io::detail::codec_exec_result> dec_res{};  // [in] results of decompression
  device_span<uint8_t const>* copy_in_ctl{};                   // [out] input buffer to copy
  device_span<uint8_t>* copy_out_ctl{};                        // [out] output buffer to copy to
  uint32_t num_compressed_blocks{};    // [in,out] number of entries in decctl(in), number of
                                       // compressed blocks(out)
  uint32_t num_uncompressed_blocks{};  // [in,out] number of entries in dec_in_ctl(in), number of
                                       // uncompressed blocks(out)
  uint64_t max_uncompressed_size{};    // [out] maximum uncompressed data size of stream
  uint32_t max_uncompressed_block_size{};  // [out] maximum uncompressed size of any block in stream
};

enum stream_index_type {
  CI_DATA = 0,    // Primary data stream
  CI_DATA2,       // Secondary/Length stream
  CI_PRESENT,     // Present stream
  CI_DICTIONARY,  // Dictionary stream
  CI_INDEX,       // Index stream
  CI_NUM_STREAMS
};

/**
 * @brief Struct to describe a single entry in the global dictionary
 */
struct dictionary_entry {
  uint32_t pos;  // Position in data stream
  uint32_t len;  // Length in data stream
};

/**
 * @brief Struct to describe per stripe's column information
 */
struct column_desc {
  uint8_t const* streams[CI_NUM_STREAMS];  // ptr to data stream index
  uint32_t strm_id[CI_NUM_STREAMS];        // stream ids
  int64_t strm_len[CI_NUM_STREAMS];        // stream length
  uint32_t* valid_map_base;                // base pointer of valid bit map for this column
  void* column_data_base;                  // base pointer of column data
  int64_t start_row;                       // starting row of the stripe
  int64_t num_rows;                        // number of rows in stripe
  int64_t column_num_rows;                 // number of rows in whole column
  int64_t num_child_rows;                  // store number of child rows if it's list column
  uint32_t num_rowgroups;                  // number of rowgroups in the chunk
  int64_t dictionary_start;                // start position in global dictionary
  uint32_t dict_len;                       // length of local dictionary
  int64_t null_count;                      // number of null values in this stripe's column
  int64_t skip_count;                      // number of non-null values to skip
  uint32_t rowgroup_id;                    // row group position
  ColumnEncodingKind encoding_kind;        // column encoding kind
  TypeKind type_kind;                      // column data type
  uint8_t dtype_len;          // data type length (for types that can be mapped to different sizes)
  type_id dtype_id;           // TODO
  int32_t decimal_scale;      // number of fractional decimal digits for decimal type
  type_id timestamp_type_id;  // output timestamp type id (type_id::EMPTY by default)
  column_validity_info parent_validity_info;  // consists of parent column valid_map and null count
  uint32_t* parent_null_count_prefix_sums;  // per-stripe prefix sums of parent column's null count
};

/**
 * @brief Struct to describe a groups of row belonging to a column stripe
 */
struct row_group {
  uint32_t chunk_id;        // Column chunk this entry belongs to
  int64_t strm_offset[2];   // Index offset for CI_DATA and CI_DATA2 streams
  uint16_t run_pos[2];      // Run position for CI_DATA and CI_DATA2
  uint32_t num_rows;        // number of rows in rowgroup
  int64_t start_row;        // starting row of the rowgroup
  uint32_t num_child_rows;  // number of rows of children in rowgroup in case of list type
};

/**
 * @brief Struct to describe an encoder data chunk
 */
struct encoder_chunk {
  int64_t start_row;                 // start row of this chunk
  uint32_t num_rows;                 // number of rows in this chunk
  int64_t null_mask_start_row;       // adjusted to multiple of 8
  uint32_t null_mask_num_rows;       // adjusted to multiple of 8
  ColumnEncodingKind encoding_kind;  // column encoding kind
  TypeKind type_kind;                // column data type
  uint8_t dtype_len;                 // data type length
  int32_t scale;                     // scale for decimals or timestamps

  uint32_t* dict_index;       // dictionary index from row index
  uint32_t* dict_data_order;  // map from data to sorted data indices
  uint32_t* decimal_offsets;
  orc_column_device_view const* column;
};

/**
 * @brief Struct to describe the streams that correspond to a single `encoder_chunk`.
 */
struct encoder_chunk_streams {
  uint8_t* data_ptrs[CI_NUM_STREAMS];  // encoded output
  int32_t ids[CI_NUM_STREAMS];         // stream id; -1 if stream is not present
  uint32_t lengths[CI_NUM_STREAMS];    // in: max length, out: actual length
};

/**
 * @brief Struct to describe a column stream within a stripe
 */
struct stripe_stream {
  uint8_t* data_ptr;        // encoded and gathered output
  size_t bfr_offset;        // Offset of this stream in compressed buffer
  uint32_t stream_size;     // Size of stream in bytes
  uint32_t first_chunk_id;  // First chunk of the stripe
  uint32_t num_chunks;      // Number of chunks in the stripe
  uint32_t column_id;       // column index
  uint32_t first_block;     // First compressed block
  uint8_t stream_type;      // Stream index type
  uint8_t pad[3];
};

/**
 * @brief Struct to describe a stripe dictionary
 */
struct stripe_dictionary {
  // input
  device_span<slot_type> map_slots;  // hash map (slots) storage
  uint32_t column_idx      = 0;      // column index
  size_type start_row      = 0;      // first row in the stripe
  size_type start_rowgroup = 0;      // first rowgroup in the stripe
  size_type num_rows       = 0;      // number of rows in the stripe

  // output
  device_span<uint32_t> data;        // index of elements in the column to include in the dictionary
  device_span<uint32_t> index;       // index into the dictionary for each row in the column
  device_span<uint32_t> data_order;  // map from data to sorted data indices
  size_type entry_count = 0;         // number of entries in the dictionary
  size_type char_count  = 0;         // number of characters in the dictionary
  bool is_enabled       = false;     // true if dictionary encoding is enabled for this stripe
};

/**
 * @brief Initializes the hash maps storage for dictionary encoding to sentinel values.
 *
 * @param dictionaries Dictionary descriptors
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void initialize_dictionary_hash_maps(device_2dspan<stripe_dictionary> dictionaries,
                                     rmm::cuda_stream_view stream);

/**
 * @brief Populates the hash maps with unique values from the stripe.
 *
 * @param dictionaries Dictionary descriptors
 * @param columns  Pre-order flattened device array of ORC column views
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void populate_dictionary_hash_maps(device_2dspan<stripe_dictionary> dictionaries,
                                   device_span<orc_column_device_view const> columns,
                                   rmm::cuda_stream_view stream);

/**
 * @brief Stores the indices of the hash map entries in the dictionary data buffer.
 *
 * @param dictionaries Dictionary descriptors
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void collect_map_entries(device_2dspan<stripe_dictionary> dictionaries,
                         rmm::cuda_stream_view stream);

/**
 * @brief Stores the corresponding dictionary indices for each row in the column.
 *
 * @param dictionaries Dictionary descriptors
 * @param columns Pre-order flattened device array of ORC column views
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void get_dictionary_indices(device_2dspan<stripe_dictionary> dictionaries,
                            device_span<orc_column_device_view const> columns,
                            rmm::cuda_stream_view stream);

constexpr uint32_t encode_block_size = 512;

/**
 * @brief Launches kernel for parsing the compressed stripe data
 *
 * @param[in] strm_info List of compressed streams
 * @param[in] num_streams Number of compressed streams
 * @param[in] compression_block_size maximum size of compressed blocks (up to 16M)
 * @param[in] log2maxcr log2 of maximum compression ratio (used to infer max uncompressed size from
 * compressed size)
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void parse_compressed_stripe_data(compressed_stream_info* strm_info,
                                  int32_t num_streams,
                                  uint64_t compression_block_size,
                                  uint32_t log2maxcr,
                                  rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for re-assembling decompressed blocks into a single contiguous block
 *
 * @param[in] strm_info List of compressed streams
 * @param[in] num_streams Number of compressed streams
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void post_decompression_reassemble(compressed_stream_info* strm_info,
                                   int32_t num_streams,
                                   rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for constructing rowgroup from index streams
 *
 * @param[out] row_groups row_group device array [rowgroup][column]
 * @param[in] strm_info List of compressed streams (or NULL if uncompressed)
 * @param[in] chunks column_desc device array [stripe][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] rowidx_stride Row index stride
 * @param[in] use_base_stride Whether to use base stride obtained from meta or use the computed
 * value
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void parse_row_group_index(row_group* row_groups,
                           compressed_stream_info* strm_info,
                           column_desc* chunks,
                           size_type num_columns,
                           size_type num_stripes,
                           size_type rowidx_stride,
                           bool use_base_stride,
                           rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for decoding NULLs and building string dictionary index tables
 *
 * @param[in] chunks column_desc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] first_row Crop all rows below first_row
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void decode_nulls_and_string_dictionaries(column_desc* chunks,
                                          dictionary_entry* global_dictionary,
                                          size_type num_columns,
                                          size_type num_stripes,
                                          int64_t first_row,
                                          rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for decoding column data
 *
 * @param[in] chunks column_desc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] first_row Crop all rows below first_row
 * @param[in] tz_table Timezone translation table
 * @param[in] row_groups Optional row index data [rowgroup][column]
 * @param[in] num_rowgroups Number of row groups in row index data
 * @param[in] rowidx_stride Row index stride
 * @param[in] level Current nesting level being processed
 * @param[out] error_count Number of errors during decode
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void decode_column_data(column_desc* chunks,
                        dictionary_entry* global_dictionary,
                        device_2dspan<row_group> row_groups,
                        size_type num_columns,
                        size_type num_stripes,
                        int64_t first_row,
                        table_device_view tz_table,
                        int64_t num_rowgroups,
                        size_type rowidx_stride,
                        size_t level,
                        size_type* error_count,
                        rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for encoding column data
 *
 * @param[in] chunks  encoder chunk device array [column][rowgroup]
 * @param[in, out] streams chunk streams device array [column][rowgroup]
 * @param[in] base_epoch Instant that encoded timestamps are stored relative to
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void encode_orc_column_data(device_2dspan<encoder_chunk const> chunks,
                            device_2dspan<encoder_chunk_streams> streams,
                            duration_s base_epoch,
                            rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for encoding column dictionaries
 *
 * @param[in] stripes Stripe dictionaries device array
 * @param[in] columns Pre-order flattened device array of ORC column views
 * @param[in] chunks encoder chunk device array [column][rowgroup]
 * @param[in] num_string_columns Number of string columns
 * @param[in] num_stripes Number of stripes
 * @param[in,out] enc_streams chunk streams device array [column][rowgroup]
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void encode_stripe_dictionaries(stripe_dictionary const* stripes,
                                device_span<orc_column_device_view const> columns,
                                device_2dspan<encoder_chunk const> chunks,
                                size_type num_string_columns,
                                size_type num_stripes,
                                device_2dspan<encoder_chunk_streams> enc_streams,
                                rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for compacting chunked column data prior to compression
 *
 * @param[in,out] strm_desc stripe_stream device array [stripe][stream]
 * @param[in,out] enc_streams chunk streams device array [column][rowgroup]
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void compact_orc_data_streams(device_2dspan<stripe_stream> strm_desc,
                              device_2dspan<encoder_chunk_streams> enc_streams,
                              rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel(s) for compressing data streams
 *
 * @param[in] compressed_data Output compressed blocks
 * @param[in] num_compressed_blocks Total number of compressed blocks
 * @param[in] compression Type of compression
 * @param[in] comp_blk_size Compression block size
 * @param[in] max_comp_blk_size Max size of any block after compression
 * @param[in] comp_block_align Required alignment for compressed blocks
 * @param[in] collect_statistics Whether to collect compression statistics
 * @param[in,out] strm_desc stripe_stream device array [stripe][stream]
 * @param[in,out] enc_streams chunk streams device array [column][rowgroup]
 * @param[out] comp_res Per-block compression status
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Compression statistics (if requested)
 */
std::optional<writer_compression_statistics> compress_orc_data_streams(
  device_span<uint8_t> compressed_data,
  uint32_t num_compressed_blocks,
  compression_type compression,
  uint32_t comp_blk_size,
  uint32_t max_comp_blk_size,
  uint32_t comp_block_align,
  bool collect_statistics,
  device_2dspan<stripe_stream> strm_desc,
  device_2dspan<encoder_chunk_streams> enc_streams,
  device_span<cudf::io::detail::codec_exec_result> comp_res,
  rmm::cuda_stream_view stream);

/**
 * @brief Counts the number of characters in each rowgroup of each string column.
 *
 * @param counts Output array of character counts [column][rowgroup]
 * @param orc_columns Pre-order flattened device array of ORC column views
 * @param rowgroup_bounds Ranges of rows in each rowgroup [rowgroup][column]
 * @param str_col_indexes Indexes of string columns in orc_columns
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void rowgroup_char_counts(device_2dspan<size_type> counts,
                          device_span<orc_column_device_view const> orc_columns,
                          device_2dspan<rowgroup_rows const> rowgroup_bounds,
                          device_span<uint32_t const> str_col_indexes,
                          rmm::cuda_stream_view stream);

/**
 * @brief Converts sizes of decimal elements to offsets within the rowgroup.
 *
 * @note The conversion is done in-place. After the conversion, the device vectors in \p elem_sizes
 * hold the offsets.
 *
 * @param rg_bounds Ranges of rows in each rowgroup [rowgroup][column]
 * @param elem_sizes Map between column indexes and decimal element sizes
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void decimal_sizes_to_offsets(device_2dspan<rowgroup_rows const> rg_bounds,
                              std::map<uint32_t, rmm::device_uvector<uint32_t>>& elem_sizes,
                              rmm::cuda_stream_view stream);

/**
 * @brief Launches kernels to initialize statistics collection
 *
 * @param[out] groups Statistics groups (rowgroup-level)
 * @param[in] cols Column descriptors
 * @param[in] rowgroup_bounds Ranges of rows in each rowgroup [rowgroup][column]
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void orc_init_statistics_groups(statistics_group* groups,
                                stats_column_desc const* cols,
                                device_2dspan<rowgroup_rows const> rowgroup_bounds,
                                rmm::cuda_stream_view stream);

/**
 * @brief Launches kernels to return statistics buffer offsets and sizes
 *
 * @param[in,out] groups Statistics merge groups
 * @param[in] chunks Statistics chunks
 * @param[in] statistics_count Number of statistics buffers to encode
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void orc_init_statistics_buffersize(statistics_merge_group* groups,
                                    statistics_chunk const* chunks,
                                    uint32_t statistics_count,
                                    rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel to encode statistics in ORC protobuf format
 *
 * @param[out] blob_bfr Output buffer for statistics blobs
 * @param[in,out] groups Statistics merge groups
 * @param[in,out] chunks Statistics data
 * @param[in] statistics_count Number of statistics buffers
 * @param[in] timestamps_are_utc Whether the written timestamps are relative to UTC
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void orc_encode_statistics(uint8_t* blob_bfr,
                           statistics_merge_group* groups,
                           statistics_chunk const* chunks,
                           uint32_t statistics_count,
                           bool timestamps_are_utc,
                           rmm::cuda_stream_view stream);

/**
 * @brief Number of set bits in pushdown masks, per rowgroup.
 *
 * @param[in] orc_columns Pre-order flattened device array of ORC column views
 * @param[in] rowgroup_bounds Ranges of rows in each rowgroup [rowgroup][column]
 * @param[out] set_counts Per rowgroup number of set bits
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void reduce_pushdown_masks(device_span<orc_column_device_view const> orc_columns,
                           device_2dspan<rowgroup_rows const> rowgroup_bounds,
                           device_2dspan<cudf::size_type> set_counts,
                           rmm::cuda_stream_view stream);

}  // namespace cudf::io::orc::detail
