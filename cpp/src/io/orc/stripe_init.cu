/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/block_utils.cuh"
#include "orc_gpu.hpp"

#include <cudf/detail/null_mask.cuh>
#include <cudf/io/orc_types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/cub.cuh>
#include <cuda/std/array>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace cudf::io::orc::detail {

struct comp_in_out {
  uint8_t const* in_ptr{};
  size_t in_size{};
  uint8_t* out_ptr{};
  size_t out_size{};
};
struct compressed_stream_s {
  compressed_stream_info info{};
  comp_in_out ctl{};
};

// blockDim {128,1,1}
CUDF_KERNEL void __launch_bounds__(128, 8)
  parse_compressed_stripe_data_kernel(compressed_stream_info* strm_info,
                                      int32_t num_streams,
                                      uint64_t compression_block_size,
                                      uint32_t log2maxcr)
{
  __shared__ compressed_stream_s strm_g[4];

  compressed_stream_s* const s = &strm_g[threadIdx.x / 32];
  int strm_id                  = blockIdx.x * 4 + (threadIdx.x / 32);
  int lane_id                  = threadIdx.x % 32;

  if (strm_id < num_streams && lane_id == 0) { s->info = strm_info[strm_id]; }

  __syncthreads();
  if (strm_id < num_streams) {
    // Walk through the compressed blocks
    uint8_t const* cur                   = s->info.compressed_data;
    uint8_t const* end                   = cur + s->info.compressed_data_size;
    uint8_t* uncompressed                = s->info.uncompressed_data;
    size_t max_uncompressed_size         = 0;
    uint64_t max_uncompressed_block_size = 0;
    uint32_t num_compressed_blocks       = 0;
    uint32_t num_uncompressed_blocks     = 0;
    while (cur + block_header_size < end) {
      uint32_t block_len = shuffle((lane_id == 0) ? cur[0] | (cur[1] << 8) | (cur[2] << 16) : 0);
      auto const is_uncompressed = static_cast<bool>(block_len & 1);
      uint64_t uncompressed_size;
      device_span<uint8_t const>* init_in_ctl = nullptr;
      device_span<uint8_t>* init_out_ctl      = nullptr;
      block_len >>= 1;
      cur += block_header_size;
      if (block_len > compression_block_size || cur + block_len > end) {
        // Fatal
        num_compressed_blocks       = 0;
        max_uncompressed_size       = 0;
        max_uncompressed_block_size = 0;
        break;
      }
      // TBD: For some codecs like snappy, it wouldn't be too difficult to get the actual
      // uncompressed size and avoid waste due to block size alignment For now, rely on the max
      // compression ratio to limit waste for the most extreme cases (small single-block streams)
      uncompressed_size = (is_uncompressed) ? block_len
                          : (block_len < (compression_block_size >> log2maxcr))
                            ? block_len << log2maxcr
                            : compression_block_size;
      if (is_uncompressed) {
        if (uncompressed_size <= 32) {
          // For short blocks, copy the uncompressed data to output
          if (uncompressed &&
              max_uncompressed_size + uncompressed_size <= s->info.max_uncompressed_size &&
              lane_id < uncompressed_size) {
            uncompressed[max_uncompressed_size + lane_id] = cur[lane_id];
          }
        } else {
          init_in_ctl =
            (s->info.copy_in_ctl && num_uncompressed_blocks < s->info.num_uncompressed_blocks)
              ? &s->info.copy_in_ctl[num_uncompressed_blocks]
              : nullptr;
          init_out_ctl =
            (s->info.copy_out_ctl && num_uncompressed_blocks < s->info.num_uncompressed_blocks)
              ? &s->info.copy_out_ctl[num_uncompressed_blocks]
              : nullptr;
          num_uncompressed_blocks++;
        }
      } else {
        init_in_ctl = (s->info.dec_in_ctl && num_compressed_blocks < s->info.num_compressed_blocks)
                        ? &s->info.dec_in_ctl[num_compressed_blocks]
                        : nullptr;
        init_out_ctl =
          (s->info.dec_out_ctl && num_compressed_blocks < s->info.num_compressed_blocks)
            ? &s->info.dec_out_ctl[num_compressed_blocks]
            : nullptr;
        num_compressed_blocks++;
      }
      if (!lane_id && init_in_ctl) {
        s->ctl = {cur, block_len, uncompressed + max_uncompressed_size, uncompressed_size};
      }
      __syncwarp();
      if (init_in_ctl && lane_id == 0) {
        *init_in_ctl  = {s->ctl.in_ptr, s->ctl.in_size};
        *init_out_ctl = {s->ctl.out_ptr, s->ctl.out_size};
      }
      cur += block_len;
      max_uncompressed_size += uncompressed_size;
      max_uncompressed_block_size = max(max_uncompressed_block_size, uncompressed_size);
    }
    __syncwarp();
    if (!lane_id) {
      s->info.num_compressed_blocks       = num_compressed_blocks;
      s->info.num_uncompressed_blocks     = num_uncompressed_blocks;
      s->info.max_uncompressed_size       = max_uncompressed_size;
      s->info.max_uncompressed_block_size = max_uncompressed_block_size;
    }
  }

  __syncthreads();
  if (strm_id < num_streams && lane_id == 0) strm_info[strm_id] = s->info;
}

// blockDim {128,1,1}
CUDF_KERNEL void __launch_bounds__(128, 8)
  post_decompression_reassemble_kernel(compressed_stream_info* strm_info, int32_t num_streams)
{
  __shared__ compressed_stream_s strm_g[4];

  compressed_stream_s* const s = &strm_g[threadIdx.x / 32];
  int strm_id                  = blockIdx.x * 4 + (threadIdx.x / 32);
  int lane_id                  = threadIdx.x % 32;

  if (strm_id < num_streams && lane_id == 0) s->info = strm_info[strm_id];

  __syncthreads();
  if (strm_id < num_streams &&
      s->info.num_compressed_blocks + s->info.num_uncompressed_blocks > 0 &&
      s->info.max_uncompressed_size > 0) {
    // Walk through the compressed blocks
    uint8_t const* cur              = s->info.compressed_data;
    uint8_t const* end              = cur + s->info.compressed_data_size;
    auto dec_out                    = s->info.dec_out_ctl;
    auto dec_result                 = s->info.dec_res;
    uint8_t* uncompressed_actual    = s->info.uncompressed_data;
    uint8_t* uncompressed_estimated = uncompressed_actual;
    uint32_t num_compressed_blocks  = 0;
    uint32_t max_compressed_blocks  = s->info.num_compressed_blocks;

    while (cur + block_header_size < end) {
      uint32_t block_len = shuffle((lane_id == 0) ? cur[0] | (cur[1] << 8) | (cur[2] << 16) : 0);
      auto const is_uncompressed = static_cast<bool>(block_len & 1);
      uint32_t uncompressed_size_est, uncompressed_size_actual;
      block_len >>= 1;
      cur += block_header_size;
      if (cur + block_len > end) { break; }
      if (is_uncompressed) {
        uncompressed_size_est    = block_len;
        uncompressed_size_actual = block_len;
      } else {
        if (num_compressed_blocks > max_compressed_blocks) { break; }
        uint32_t const dst_size      = dec_out[num_compressed_blocks].size();
        uncompressed_size_est        = shuffle((lane_id == 0) ? dst_size : 0);
        uint32_t const bytes_written = dec_result[num_compressed_blocks].bytes_written;
        uncompressed_size_actual     = shuffle((lane_id == 0) ? bytes_written : 0);
      }
      // In practice, this should never happen with a well-behaved writer, as we would expect the
      // uncompressed size to always be equal to the compression block size except for the last
      // block
      if (uncompressed_actual < uncompressed_estimated) {
        // warp-level memmove
        for (int i = lane_id; i < (int)uncompressed_size_actual; i += 32) {
          uncompressed_actual[i] = uncompressed_estimated[i];
        }
      }
      cur += block_len;
      num_compressed_blocks += 1 - is_uncompressed;
      uncompressed_estimated += uncompressed_size_est;
      uncompressed_actual += uncompressed_size_actual;
    }
    // Update info with actual uncompressed size
    if (!lane_id) {
      size_t total_uncompressed_size = uncompressed_actual - s->info.uncompressed_data;
      // Set uncompressed size to zero if there were any errors
      strm_info[strm_id].max_uncompressed_size =
        (num_compressed_blocks == s->info.num_compressed_blocks) ? total_uncompressed_size : 0;
    }
  }
}

/**
 * @brief Shared mem state for parse_row_group_index_kernel
 */
struct rowindex_state_s {
  column_desc chunk{};
  uint32_t rowgroup_start{};
  uint32_t rowgroup_end{};
  int is_compressed{};
  uint32_t row_index_entry[3]
                          [CI_PRESENT]{};  // NOTE: Assumes CI_PRESENT follows CI_DATA and CI_DATA2
  compressed_stream_info strm_info[2]{};
  row_group rowgroups[128]{};
  uint32_t compressed_offset[128][2]{};
};

enum row_entry_state_e {
  NOT_FOUND = 0,
  GET_LENGTH,
  SKIP_VARINT,
  SKIP_FIXEDLEN,
  STORE_INDEX0,
  STORE_INDEX1,
  STORE_INDEX2,
};

/**
 * @brief Calculates the order of index streams based on the index types present in the column.
 *
 * @param index_types_bitmap The bitmap of index types showing which index streams are present
 *
 * @return The order of index streams
 */
static auto __device__ index_order_from_index_types(uint32_t index_types_bitmap)
{
  constexpr cuda::std::array full_order = {CI_PRESENT, CI_DATA, CI_DATA2};

  cuda::std::array<uint32_t, full_order.size()> partial_order;
  thrust::copy_if(thrust::seq,
                  full_order.cbegin(),
                  full_order.cend(),
                  partial_order.begin(),
                  [index_types_bitmap] __device__(auto index_type) {
                    // Check if the index type is present
                    return index_types_bitmap & (1 << index_type);
                  });

  return partial_order;
}

/**
 * @brief Decode a single row group index entry
 *
 * @param[in,out] s row group index state
 * @param[in] start start position in byte stream
 * @param[in] end end of byte stream
 * @return bytes consumed
 */
static uint32_t __device__ protobuf_parse_row_index_entry(rowindex_state_s* s,
                                                          uint8_t const* const start,
                                                          uint8_t const* const end)
{
  constexpr uint32_t pb_rowindexentry_id = ProtofType::FIXEDLEN + 8;
  auto const stream_order                = index_order_from_index_types(s->chunk.skip_count);

  uint8_t const* cur      = start;
  row_entry_state_e state = NOT_FOUND;
  uint32_t length         = 0;
  uint32_t idx_id         = 0;
  uint32_t pos_end        = 0;
  uint32_t ci_id          = CI_NUM_STREAMS;
  while (cur < end) {
    uint32_t v = 0;
    for (uint32_t l = 0; l <= 28; l += 7) {
      uint32_t c = (cur < end) ? *cur++ : 0;
      v |= (c & 0x7f) << l;
      if (c <= 0x7f) break;
    }
    switch (state) {
      case NOT_FOUND:
        if (v == pb_rowindexentry_id) {
          state = GET_LENGTH;
        } else {
          v &= 7;
          if (v == ProtofType::FIXED64)
            cur += 8;
          else if (v == ProtofType::FIXED32)
            cur += 4;
          else if (v == ProtofType::VARINT)
            state = SKIP_VARINT;
          else if (v == ProtofType::FIXEDLEN)
            state = SKIP_FIXEDLEN;
        }
        break;
      case SKIP_VARINT: state = NOT_FOUND; break;
      case SKIP_FIXEDLEN:
        cur += v;
        state = NOT_FOUND;
        break;
      case GET_LENGTH:
        if (length == 0) {
          length = (uint32_t)(cur + v - start);
          state = NOT_FOUND;  // Scan for positions (same field id & low-level type as RowIndexEntry
                              // entry)
        } else {
          pos_end = min((uint32_t)(cur + v - start), length);
          state   = STORE_INDEX0;
        }
        break;
      case STORE_INDEX0:
        // Start of a new entry; determine the stream index types
        ci_id = stream_order[idx_id++];
        if (s->is_compressed) {
          if (ci_id < CI_PRESENT) s->row_index_entry[0][ci_id] = v;
          if (cur >= start + pos_end) return length;
          state = STORE_INDEX1;
          break;
        } else {
          if (ci_id < CI_PRESENT) s->row_index_entry[0][ci_id] = 0;
          // Fall through to STORE_INDEX1 for uncompressed (always block0)
        }
      case STORE_INDEX1:
        if (ci_id < CI_PRESENT) s->row_index_entry[1][ci_id] = v;
        if (cur >= start + pos_end) return length;
        state = (ci_id == CI_DATA && s->chunk.encoding_kind != DICTIONARY &&
                 s->chunk.encoding_kind != DICTIONARY_V2 &&
                 (s->chunk.type_kind == STRING || s->chunk.type_kind == BINARY ||
                  s->chunk.type_kind == VARCHAR || s->chunk.type_kind == CHAR ||
                  s->chunk.type_kind == DECIMAL || s->chunk.type_kind == FLOAT ||
                  s->chunk.type_kind == DOUBLE))
                  ? STORE_INDEX0
                  : STORE_INDEX2;
        break;
      case STORE_INDEX2:
        if (ci_id < CI_PRESENT) {
          // Boolean columns have an extra byte to indicate the position of the bit within the byte
          s->row_index_entry[2][ci_id] = (s->chunk.type_kind == BOOLEAN) ? (v << 3) + *cur : v;
        }
        if (ci_id == CI_PRESENT || s->chunk.type_kind == BOOLEAN) cur++;
        if (cur >= start + pos_end) return length;
        state = STORE_INDEX0;
        break;
    }
  }
  return (uint32_t)(end - start);
}

/**
 * @brief Decode row group index entries
 *
 * @param[in,out] s row group index state
 * @param[in] num_rowgroups Number of index entries to read
 */
static __device__ void read_row_group_index_entries(rowindex_state_s* s, int num_rowgroups)
{
  uint8_t const* index_data = s->chunk.streams[CI_INDEX];
  int index_data_len        = s->chunk.strm_len[CI_INDEX];
  for (int i = 0; i < num_rowgroups; i++) {
    s->row_index_entry[0][0] = 0;
    s->row_index_entry[0][1] = 0;
    s->row_index_entry[1][0] = 0;
    s->row_index_entry[1][1] = 0;
    s->row_index_entry[2][0] = 0;
    s->row_index_entry[2][1] = 0;
    if (index_data_len > 0) {
      int len = protobuf_parse_row_index_entry(s, index_data, index_data + index_data_len);
      index_data += len;
      index_data_len = max(index_data_len - len, 0);
      for (int j = 0; j < 2; j++) {
        s->rowgroups[i].strm_offset[j] = s->row_index_entry[1][j];
        s->rowgroups[i].run_pos[j]     = s->row_index_entry[2][j];
        s->compressed_offset[i][j]     = s->row_index_entry[0][j];
      }
    }
  }
  s->chunk.streams[CI_INDEX]  = index_data;
  s->chunk.strm_len[CI_INDEX] = index_data_len;
}

/**
 * @brief Translate block+offset compressed position into an uncompressed offset
 *
 * @param[in,out] s row group index state
 * @param[in] ci_id index to convert (CI_DATA or CI_DATA2)
 * @param[in] num_rowgroups Number of index entries
 * @param[in] t thread id
 */
static __device__ void map_row_index_to_uncompressed(rowindex_state_s* s,
                                                     int ci_id,
                                                     int num_rowgroups,
                                                     int t)
{
  int32_t strm_len = s->chunk.strm_len[ci_id];
  if (strm_len > 0) {
    int32_t compressed_offset = (t < num_rowgroups) ? s->compressed_offset[t][ci_id] : 0;
    if (compressed_offset > 0) {
      uint8_t const* start   = s->strm_info[ci_id].compressed_data;
      uint8_t const* cur     = start;
      uint8_t const* end     = cur + s->strm_info[ci_id].compressed_data_size;
      auto dec_result        = s->strm_info[ci_id].dec_res.data();
      uint32_t uncomp_offset = 0;
      for (;;) {
        uint32_t block_len;

        if (cur + block_header_size > end || cur + block_header_size >= start + compressed_offset) {
          break;
        }
        block_len = cur[0] | (cur[1] << 8) | (cur[2] << 16);
        cur += block_header_size;
        auto const is_uncompressed = static_cast<bool>(block_len & 1);
        block_len >>= 1;
        cur += block_len;
        if (cur > end) { break; }
        if (is_uncompressed) {
          uncomp_offset += block_len;
        } else {
          uncomp_offset += dec_result->bytes_written;
          dec_result++;
        }
      }
      s->rowgroups[t].strm_offset[ci_id] += uncomp_offset;
    }
  }
}

/**
 * @brief Decode index streams
 *
 * @param[out] row_groups row_group device array [rowgroup][column]
 * @param[in] strm_info List of compressed streams (or NULL if uncompressed)
 * @param[in] chunks column_desc device array [stripe][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] num_rowgroups Number of row groups
 * @param[in] rowidx_stride Row index stride
 * @param[in] use_base_stride Whether to use base stride obtained from meta or use the computed
 * value
 */
// blockDim {128,1,1}
CUDF_KERNEL void __launch_bounds__(128, 8)
  parse_row_group_index_kernel(row_group* row_groups,
                               compressed_stream_info* strm_info,
                               column_desc* chunks,
                               size_type num_columns,
                               size_type num_stripes,
                               size_type rowidx_stride,
                               bool use_base_stride)
{
  __shared__ __align__(16) rowindex_state_s state_g;
  rowindex_state_s* const s = &state_g;
  auto const col_idx        = blockIdx.x / num_stripes;
  auto const stripe_idx     = blockIdx.x % num_stripes;
  uint32_t chunk_id         = stripe_idx * num_columns + col_idx;
  int t                     = threadIdx.x;

  if (t == 0) {
    s->chunk = chunks[chunk_id];
    if (strm_info) {
      if (s->chunk.strm_len[0] > 0) s->strm_info[0] = strm_info[s->chunk.strm_id[0]];
      if (s->chunk.strm_len[1] > 0) s->strm_info[1] = strm_info[s->chunk.strm_id[1]];
    }

    uint32_t rowgroups_in_chunk = s->chunk.num_rowgroups;
    s->rowgroup_start           = s->chunk.rowgroup_id;
    s->rowgroup_end             = s->rowgroup_start + rowgroups_in_chunk;
    s->is_compressed            = (strm_info != nullptr);
  }
  __syncthreads();
  while (s->rowgroup_start < s->rowgroup_end) {
    int num_rowgroups = min(s->rowgroup_end - s->rowgroup_start, 128);
    int rowgroup_size4, t4, t32;

    s->rowgroups[t].chunk_id = chunk_id;
    if (t == 0) { read_row_group_index_entries(s, num_rowgroups); }
    __syncthreads();
    if (s->is_compressed) {
      // Convert the block + blk_offset pair into a raw offset into the decompressed stream
      if (s->chunk.strm_len[CI_DATA] > 0) {
        map_row_index_to_uncompressed(s, CI_DATA, num_rowgroups, t);
      }
      if (s->chunk.strm_len[CI_DATA2] > 0) {
        map_row_index_to_uncompressed(s, CI_DATA2, num_rowgroups, t);
      }
      __syncthreads();
    }
    rowgroup_size4 = sizeof(row_group) / sizeof(uint32_t);
    t4             = t & 3;
    t32            = t >> 2;
    for (int i = t32; i < num_rowgroups; i += 32) {
      auto const num_rows =
        (use_base_stride) ? rowidx_stride
                          : row_groups[(s->rowgroup_start + i) * num_columns + col_idx].num_rows;
      auto const start_row =
        (use_base_stride) ? i * rowidx_stride
                          : row_groups[(s->rowgroup_start + i) * num_columns + col_idx].start_row;
      for (int j = t4; j < rowgroup_size4; j += 4) {
        ((uint32_t*)&row_groups[(s->rowgroup_start + i) * num_columns + col_idx])[j] =
          ((uint32_t*)&s->rowgroups[i])[j];
      }
      row_groups[(s->rowgroup_start + i) * num_columns + col_idx].num_rows = num_rows;
      // Updating in case of struct
      row_groups[(s->rowgroup_start + i) * num_columns + col_idx].num_child_rows = num_rows;
      row_groups[(s->rowgroup_start + i) * num_columns + col_idx].start_row      = start_row;
    }
    __syncthreads();
    if (t == 0) { s->rowgroup_start += num_rowgroups; }
    __syncthreads();
  }
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  reduce_pushdown_masks_kernel(device_span<orc_column_device_view const> orc_columns,
                               device_2dspan<rowgroup_rows const> rowgroup_bounds,
                               device_2dspan<size_type> set_counts)
{
  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto const column_id   = blockIdx.x / rowgroup_bounds.size().first;
  auto const rowgroup_id = blockIdx.x % rowgroup_bounds.size().first;
  auto const column      = orc_columns[column_id];
  auto const t           = threadIdx.x;

  auto const use_child_rg = column.type().id() == type_id::LIST;
  auto const rg           = rowgroup_bounds[rowgroup_id][column_id + (use_child_rg ? 1 : 0)];

  if (column.pushdown_mask == nullptr) {
    // All elements are valid if the null mask is not present
    if (t == 0) { set_counts[rowgroup_id][column_id] = rg.size(); }
    return;
  };

  size_type count                          = 0;
  static constexpr size_type bits_per_word = sizeof(bitmask_type) * 8;
  for (auto row = t * bits_per_word + rg.begin; row < rg.end; row += block_size * bits_per_word) {
    auto const begin_bit = row;
    auto const end_bit   = min(static_cast<size_type>(row + bits_per_word), rg.end);
    auto const mask_len  = end_bit - begin_bit;
    auto const mask_word =
      cudf::detail::get_mask_offset_word(column.pushdown_mask, 0, row, end_bit) &
      ((1 << mask_len) - 1);
    count += __popc(mask_word);
  }

  count = BlockReduce(temp_storage).Sum(count);
  if (t == 0) { set_counts[rowgroup_id][column_id] = count; }
}

void __host__ parse_compressed_stripe_data(compressed_stream_info* strm_info,
                                           int32_t num_streams,
                                           uint64_t compression_block_size,
                                           uint32_t log2maxcr,
                                           rmm::cuda_stream_view stream)
{
  auto const num_blocks = (num_streams + 3) >> 2;  // 1 stream per warp, 4 warps per block
  if (num_blocks > 0) {
    parse_compressed_stripe_data_kernel<<<num_blocks, 128, 0, stream.value()>>>(
      strm_info, num_streams, compression_block_size, log2maxcr);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

void __host__ post_decompression_reassemble(compressed_stream_info* strm_info,
                                            int32_t num_streams,
                                            rmm::cuda_stream_view stream)
{
  auto const num_blocks = (num_streams + 3) >> 2;  // 1 stream per warp, 4 warps per block
  if (num_blocks > 0) {
    post_decompression_reassemble_kernel<<<num_blocks, 128, 0, stream.value()>>>(strm_info,
                                                                                 num_streams);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

void __host__ parse_row_group_index(row_group* row_groups,
                                    compressed_stream_info* strm_info,
                                    column_desc* chunks,
                                    size_type num_columns,
                                    size_type num_stripes,
                                    size_type rowidx_stride,
                                    bool use_base_stride,
                                    rmm::cuda_stream_view stream)
{
  auto const num_blocks = num_columns * num_stripes;
  parse_row_group_index_kernel<<<num_blocks, 128, 0, stream.value()>>>(
    row_groups, strm_info, chunks, num_columns, num_stripes, rowidx_stride, use_base_stride);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void __host__ reduce_pushdown_masks(device_span<orc_column_device_view const> columns,
                                    device_2dspan<rowgroup_rows const> rowgroups,
                                    device_2dspan<cudf::size_type> valid_counts,
                                    rmm::cuda_stream_view stream)
{
  auto const num_blocks    = columns.size() * rowgroups.size().first;  // 1 block per rowgroup
  constexpr int block_size = 128;
  reduce_pushdown_masks_kernel<block_size>
    <<<num_blocks, block_size, 0, stream.value()>>>(columns, rowgroups, valid_counts);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::orc::detail
