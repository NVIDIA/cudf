/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "avro_gpu.hpp"
#include "io/utilities/block_utils.cuh"

#include <rmm/cuda_stream_view.hpp>

using cudf::device_span;

namespace cudf {
namespace io {
namespace avro {
namespace gpu {
constexpr int num_warps             = 16;
constexpr int max_shared_schema_len = 1000;

/*
 * Avro varint encoding - see
 * https://avro.apache.org/docs/1.2.0/spec.html#binary_encoding
 */
static inline int64_t __device__ avro_decode_zigzag_varint(uint8_t const*& cur, uint8_t const* end)
{
  uint64_t u = 0;
  if (cur < end) {
    u = *cur++;
    if (u > 0x7f) {
      uint64_t scale = 128;
      u &= 0x7f;
      while (cur < end) {
        uint32_t c = *cur++;
        u += (c & 0x7f) * scale;
        scale <<= 7;
        if (c < 0x80) break;
      }
    }
  }
  return (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
}

/**
 * @brief Decode a row of values given an avro schema
 *
 * @param[in] schema Schema description
 * @param[in] schema_g Global schema in device mem
 * @param[in] schema_len Number of schema entries
 * @param[in] first_row First row to start saving decoded data
 * @param[in] row Current row
 * @param[in] end_row One past the last row to save
 * @param[in] row_offset Absolute row offset of this row in the
 *                       destination data.
 * @param[in] cur Current input data pointer
 * @param[in] end End of input data
 * @param[in] global_dictionary Global dictionary entries
 * @param[out] skipped_row Whether the row was skipped; set to false
 *                         if the row was saved (caller should ensure
 *                         this is initialized to true)
 *
 * @return data pointer at the end of the row (start of next row)
 */
static uint8_t const* __device__
avro_decode_row(schemadesc_s const* schema,
                schemadesc_s* schema_g,
                uint32_t schema_len,
                size_t first_row,
                size_t row,
                size_t end_row,
                size_t row_offset,
                uint8_t const* cur,
                uint8_t const* end,
                device_span<string_index_pair const> global_dictionary,
                bool* skipped_row)
{
  // `dst_row` depicts the offset of the decoded row in the destination
  // `dataptr` array, adjusted for skip rows, if applicable.  For example,
  // if `row` == 5 and `first_row` == 3, then this is the second row we'll
  // be storing (5-3).  If `first_row` is greater than `row`, this routine
  // simply decodes the row and adjusts the returned data pointer, but does
  // *not* actually store the row in the destination `dataptr` array.  This
  // is enforced by all writes to the destination memory being guarded in the
  // following fashion:
  //    if (dataptr != nullptr && dst_row > 0) {
  //      static_cast<int32_t*>(dataptr)[dst_row] = static_cast<int32_t>(v);
  //      *skipped_row = false;
  //    }
  // The actual value is calculated by subtracting the first row from this given
  // row value, and then adding the absolute row offset.  The row offset is
  // required to ensure we write to the correct destination location when we're
  // processing multiple blocks, i.e. this block could only have 10 rows, but
  // it's the 3rd block (where each block has 10 rows), so we need to write to
  // the 30th row in the destination array.
  ptrdiff_t const dst_row =
    (row >= first_row && row < end_row ? static_cast<ptrdiff_t>((row - first_row) + row_offset)
                                       : -1);
  // Critical invariant checks: dst_row should be -1 or greater, and
  // *skipped_row should always be true at this point (we set it to false only
  // if we write the decoded value to the destination array).
  if (dst_row < -1) { CUDF_UNREACHABLE("dst_row should be -1 or greater"); }
  if (*skipped_row != true) { CUDF_UNREACHABLE("skipped_row should be true"); }

  uint32_t array_start = 0, array_repeat_count = 0;
  int array_children = 0;
  for (uint32_t i = 0; i < schema_len;) {
    type_kind_e kind                = schema[i].kind;
    logicaltype_kind_e logical_kind = schema[i].logical_kind;
    int skip                        = 0;

    if (is_supported_logical_type(logical_kind)) { kind = static_cast<type_kind_e>(logical_kind); }

    if (kind == type_union) {
      int skip_after;
      if (cur >= end) break;
      skip       = (*cur++) >> 1;  // NOTE: Assumes 1-byte union member
      skip_after = schema[i].count - skip - 1;
      ++i;
      while (skip > 0 && i < schema_len) {
        if (schema[i].kind >= type_record) { skip += schema[i].count; }
        ++i;
        --skip;
      }
      if (i >= schema_len || skip_after < 0) break;
      kind         = schema[i].kind;
      logical_kind = schema[i].logical_kind;
      if (is_supported_logical_type(logical_kind)) {
        kind = static_cast<type_kind_e>(logical_kind);
      }
      skip = skip_after;
    }

    void* dataptr = schema[i].dataptr;
    switch (kind) {
      case type_null:
        if (dataptr != nullptr && dst_row >= 0) {
          atomicAnd(static_cast<uint32_t*>(dataptr) + (dst_row >> 5), ~(1 << (dst_row & 0x1f)));
          atomicAdd(&schema_g[i].count, 1U);
          *skipped_row = false;
        }
        break;

      case type_int: {
        int64_t v = avro_decode_zigzag_varint(cur, end);
        if (dataptr != nullptr && dst_row >= 0) {
          static_cast<int32_t*>(dataptr)[dst_row] = static_cast<int32_t>(v);
          *skipped_row                            = false;
        }
      } break;

      case type_long: {
        int64_t v = avro_decode_zigzag_varint(cur, end);
        if (dataptr != nullptr && dst_row >= 0) {
          static_cast<int64_t*>(dataptr)[dst_row] = v;
          *skipped_row                            = false;
        }
      } break;

      case type_bytes: [[fallthrough]];
      case type_string: [[fallthrough]];
      case type_enum: {
        int64_t v       = avro_decode_zigzag_varint(cur, end);
        size_t count    = 0;
        char const* ptr = nullptr;
        if (kind == type_enum) {  // dictionary
          size_t idx = schema[i].count + v;
          if (idx < global_dictionary.size()) {
            ptr   = global_dictionary[idx].first;
            count = global_dictionary[idx].second;
          }
        } else if (v >= 0 && cur + v <= end) {  // string or bytes
          ptr   = reinterpret_cast<char const*>(cur);
          count = (size_t)v;
          cur += count;
        }
        if (dataptr != nullptr && dst_row >= 0) {
          static_cast<string_index_pair*>(dataptr)[dst_row].first  = ptr;
          static_cast<string_index_pair*>(dataptr)[dst_row].second = count;
          *skipped_row                                             = false;
        }
      } break;

      case type_float:
        if (dataptr != nullptr && dst_row >= 0) {
          uint32_t v;
          if (cur + 3 < end) {
            v = unaligned_load<uint32_t>(cur);
            cur += 4;
          } else {
            v = 0;
          }
          static_cast<uint32_t*>(dataptr)[dst_row] = v;
          *skipped_row                             = false;
        } else {
          cur += 4;
        }
        break;

      case type_double:
        if (dataptr != nullptr && dst_row >= 0) {
          uint64_t v;
          if (cur + 7 < end) {
            v = unaligned_load<uint64_t>(cur);
            cur += 8;
          } else {
            v = 0;
          }
          static_cast<uint64_t*>(dataptr)[dst_row] = v;
          *skipped_row                             = false;
        } else {
          cur += 8;
        }
        break;

      case type_boolean:
        if (dataptr != nullptr && dst_row >= 0) {
          uint8_t v                               = (cur < end) ? *cur : 0;
          static_cast<uint8_t*>(dataptr)[dst_row] = (v) ? 1 : 0;
          *skipped_row                            = false;
        }
        cur++;
        break;

      case type_array: {
        int32_t array_block_count = avro_decode_zigzag_varint(cur, end);
        if (array_block_count < 0) {
          avro_decode_zigzag_varint(cur, end);  // block size in bytes, ignored
          array_block_count = -array_block_count;
        }
        array_start        = i;
        array_repeat_count = array_block_count;
        array_children     = 1;
        if (array_repeat_count == 0) {
          skip += schema[i].count;  // Should always be 1
        }
      } break;

      case type_duration: {
        // A duration logical type annotates Avro fixed type of size 12, which
        // stores three little-endian unsigned integers that represent durations
        // at different granularities of time. The first stores a number in
        // months, the second stores a number in days, and the third stores a
        // number in milliseconds.
        CUDF_UNREACHABLE("avro type 'duration' not yet implemented");
      } break;

      // N.B. These aren't handled yet, see the discussion on
      //      https://github.com/rapidsai/cudf/pull/12788.  The decoding logic
      //      is correct, though, so there's no harm in having them here.
      case type_timestamp_millis: [[fallthrough]];
      case type_timestamp_micros: [[fallthrough]];
      case type_local_timestamp_millis: [[fallthrough]];
      case type_local_timestamp_micros: [[fallthrough]];
      case type_time_millis: [[fallthrough]];
      case type_time_micros: {
        // N.B. time-millis is stored as a 32-bit int, however, cudf expects an
        //      int64 for DURATION_MILLISECONDS.  From our perspective, the fact
        //      that time-millis comes from a 32-bit int is hidden from us by
        //      way of the zig-zag varint encoding, so we can safely treat them
        //      both as int64_t.  Everything else is 64-bit in both avro and
        //      cudf.
        CUDF_UNREACHABLE("avro time/timestamp types not yet implemented");
        //
        // When we do implement these, the following decoding logic should
        // be correct:
        //
        // int64_t v = avro_decode_zigzag_varint(cur, end);
        // if (dataptr != nullptr && dst_row >= 0) {
        //   static_cast<int64_t*>(dataptr)[dst_row] = v;
        //   *skipped_row = false;
        // }
      } break;

      case type_date: {
        int64_t v = avro_decode_zigzag_varint(cur, end);
        if (dataptr != nullptr && dst_row >= 0) {
          static_cast<int32_t*>(dataptr)[dst_row] = static_cast<int32_t>(v);
          *skipped_row                            = false;
        }
      } break;
    }

    if (array_repeat_count != 0) {
      array_children--;
      if (schema[i].kind >= type_record) { array_children += schema[i].count; }
    }
    i++;
    while (skip > 0 && i < schema_len) {
      if (schema[i].kind >= type_record) { skip += schema[i].count; }
      ++i;
      --skip;
    }
    // If within an array, check if we reached the last item
    if (array_repeat_count != 0 && array_children <= 0 && cur < end) {
      if (!--array_repeat_count) {
        i = array_start;  // Restart at the array parent
      } else {
        i              = array_start + 1;  // Restart after the array parent
        array_children = schema[array_start].count;
      }
    }
  }
  return cur;
}

/**
 * @brief Decode column data
 *
 * @param[in] blocks Data block descriptions
 * @param[in] schema_g Schema description
 * @param[in] global_dictionary Global dictionary entries
 * @param[in] avro_data Raw block data
 * @param[in] schema_len Number of entries in schema
 * @param[in] min_row_size Minimum size in bytes of a row
 */
// blockDim {32,num_warps,1}
CUDF_KERNEL void __launch_bounds__(num_warps * 32, 2)
  gpuDecodeAvroColumnData(device_span<block_desc_s const> blocks,
                          schemadesc_s* schema_g,
                          device_span<string_index_pair const> global_dictionary,
                          uint8_t const* avro_data,
                          uint32_t schema_len,
                          uint32_t min_row_size)
{
  __shared__ __align__(8) schemadesc_s g_shared_schema[max_shared_schema_len];
  __shared__ __align__(8) block_desc_s blk_g[num_warps];

  schemadesc_s* schema;
  block_desc_s* const blk = &blk_g[threadIdx.y];
  uint32_t block_id       = blockIdx.x * num_warps + threadIdx.y;

  // Fetch schema into shared mem if possible
  if (schema_len <= max_shared_schema_len) {
    for (int i = threadIdx.y * 32 + threadIdx.x; i < schema_len; i += num_warps * 32) {
      g_shared_schema[i] = schema_g[i];
    }
    __syncthreads();
    schema = g_shared_schema;
  } else {
    schema = schema_g;
  }

  if (block_id < blocks.size() and threadIdx.x == 0) { *blk = blocks[block_id]; }
  __syncthreads();
  if (block_id >= blocks.size()) { return; }

  uint8_t const* cur      = avro_data + blk->offset;
  uint8_t const* end      = cur + blk->size;
  size_t first_row        = blk->first_row + blk->row_offset;
  size_t cur_row          = blk->row_offset;
  size_t end_row          = first_row + blk->num_rows;
  uint32_t rows_remaining = blk->num_rows;

  while (cur < end) {
    uint32_t nrows;
    uint8_t const* start = cur;

    if (cur + min_row_size * rows_remaining == end) {
      // We're dealing with predictable fixed-size rows, which means we can
      // process up to 32 rows (warp-width) at a time.  This will be the case
      // when we're dealing with fixed-size data, e.g. of floats or doubles,
      // which are always 4 or 8 bytes respectively.
      nrows = min(rows_remaining, 32);
      cur += threadIdx.x * min_row_size;
    } else {
      // We're dealing with variable-size data, so only one row can be processed
      // by one thread at a time.
      nrows = 1;
    }

    if (threadIdx.x < nrows) {
      bool skipped_row = true;
      cur              = avro_decode_row(schema,
                            schema_g,
                            schema_len,
                            first_row,
                            cur_row + threadIdx.x,
                            end_row,
                            blk->row_offset,
                            cur,
                            end,
                            global_dictionary,
                            &skipped_row);
      if (!skipped_row) { rows_remaining -= nrows; }
    }
    __syncwarp();

    cur_row += nrows;
    if (nrows == 1) {
      // Only lane 0 (i.e. 'threadIdx.x == 0') was active, so we need to
      // broadcast the new value of 'cur' and 'rows_remaining' to all other
      // threads in the warp.
      cur = start + shuffle(static_cast<uint32_t>(cur - start));
      // rows_remaining is already uint32_t, so we don't need to do the
      // start + shuffle(this - start) dance like we do above.
      rows_remaining = shuffle(rows_remaining);
    } else if (nrows > 1) {
      cur = start + (nrows * min_row_size);
    }
  }
}

/**
 * @brief Launches kernel for decoding column data
 *
 * @param[in] blocks Data block descriptions
 * @param[in] schema Schema description
 * @param[in] global_dictionary Global dictionary entries
 * @param[in] avro_data Raw block data
 * @param[in] schema_len Number of entries in schema
 * @param[in] min_row_size Minimum size in bytes of a row
 * @param[in] stream CUDA stream to use
 */
void DecodeAvroColumnData(device_span<block_desc_s const> blocks,
                          schemadesc_s* schema,
                          device_span<string_index_pair const> global_dictionary,
                          uint8_t const* avro_data,
                          uint32_t schema_len,
                          uint32_t min_row_size,
                          rmm::cuda_stream_view stream)
{
  // num_warps warps per threadblock
  dim3 const dim_block(32, num_warps);
  // 1 warp per datablock, num_warps datablocks per threadblock
  dim3 const dim_grid((blocks.size() + num_warps - 1) / num_warps, 1);

  gpuDecodeAvroColumnData<<<dim_grid, dim_block, 0, stream.value()>>>(
    blocks, schema, global_dictionary, avro_data, schema_len, min_row_size);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace gpu
}  // namespace avro
}  // namespace io
}  // namespace cudf
