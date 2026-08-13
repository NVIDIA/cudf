/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "orc_gpu.hpp"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/orc_types.hpp>
#include <cudf/strings/detail/convert/fixed_point_to_string.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/utility>

namespace cudf::io::orc::detail {

using strings::detail::fixed_point_string_size;

// Nanosecond statistics should not be enabled until the spec version is set correctly in the output
// files. See https://github.com/rapidsai/cudf/issues/14325 for more details
constexpr bool enable_nanosecond_statistics = true;

constexpr unsigned int init_threads_per_group = 32;
constexpr unsigned int init_groups_per_block  = 4;
constexpr unsigned int init_threads_per_block = init_threads_per_group * init_groups_per_block;

CUDF_KERNEL void __launch_bounds__(init_threads_per_block)
  gpu_init_statistics_groups(statistics_group* groups,
                             stats_column_desc const* cols,
                             device_2dspan<rowgroup_rows const> rowgroup_bounds)
{
  __shared__ __align__(4) statistics_group group_g[init_groups_per_block];
  auto const col_id = blockIdx.x % rowgroup_bounds.size().second;
  auto const chunk_id =
    (blockIdx.x / rowgroup_bounds.size().second * init_groups_per_block) + threadIdx.y;
  auto const t             = threadIdx.x;
  auto const num_rowgroups = rowgroup_bounds.size().first;
  statistics_group* group  = &group_g[threadIdx.y];
  if (chunk_id < num_rowgroups and t == 0) {
    group->col                                = &cols[col_id];
    group->start_row                          = rowgroup_bounds[chunk_id][col_id].begin;
    group->num_rows                           = rowgroup_bounds[chunk_id][col_id].size();
    groups[col_id * num_rowgroups + chunk_id] = *group;
  }
}

/**
 * @brief Get the buffer size and offsets of encoded statistics
 *
 * @param[in,out] groups Statistics merge groups
 * @param[in] statistics_count Number of statistics buffers
 */
constexpr unsigned int buffersize_reduction_dim = 32;
constexpr unsigned int block_size        = buffersize_reduction_dim * buffersize_reduction_dim;
constexpr unsigned int pb_fld_hdrlen     = 1;
constexpr unsigned int pb_fld_hdrlen32   = 5;
constexpr unsigned int pb_fldlen_int32   = 5;
constexpr unsigned int pb_fldlen_int64   = 10;
constexpr unsigned int pb_fldlen_float64 = 8;
constexpr unsigned int pb_fldlen_bucket1 = 1 + pb_fldlen_int64;
// statistics field number + number of values + has null
constexpr unsigned int pb_fldlen_common =
  pb_fld_hdrlen + (pb_fld_hdrlen + pb_fldlen_int64) + 2 * pb_fld_hdrlen;

template <unsigned int block_size>
CUDF_KERNEL void __launch_bounds__(block_size, 1)
  gpu_init_statistics_buffersize(statistics_merge_group* groups,
                                 statistics_chunk const* chunks,
                                 uint32_t statistics_count)
{
  using block_scan = cub::BlockScan<uint32_t, block_size, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename block_scan::TempStorage temp_storage;
  uint32_t stats_size = 0;
  auto t              = threadIdx.x;
  __syncthreads();
  for (thread_index_type start = 0; start < statistics_count; start += block_size) {
    uint32_t stats_len = 0, stats_pos;
    auto idx           = start + t;
    if (idx < statistics_count) {
      statistics_dtype const dtype = groups[idx].stats_dtype;
      switch (dtype) {
        case dtype_bool: stats_len = pb_fldlen_common + pb_fld_hdrlen + pb_fldlen_bucket1; break;
        case dtype_int8:
        case dtype_int16:
        case dtype_int32:
        case dtype_int64:
          stats_len = pb_fldlen_common + pb_fld_hdrlen + 3 * (pb_fld_hdrlen + pb_fldlen_int64);
          break;
        case dtype_date32:
          stats_len = pb_fldlen_common + pb_fld_hdrlen + 2 * (pb_fld_hdrlen + pb_fldlen_int64);
          break;
        case dtype_timestamp64:
          stats_len = pb_fldlen_common + pb_fld_hdrlen + 4 * (pb_fld_hdrlen + pb_fldlen_int64);
          if constexpr (enable_nanosecond_statistics) {
            stats_len += 2 * (pb_fld_hdrlen + pb_fldlen_int32);
          }
          break;
        case dtype_float32:
        case dtype_float64:
          stats_len = pb_fldlen_common + pb_fld_hdrlen + 3 * (pb_fld_hdrlen + pb_fldlen_float64);
          break;
        case dtype_decimal64:
        case dtype_decimal128: {
          auto const scale    = groups[idx].col_dtype.scale();
          auto const min_size = fixed_point_string_size(chunks[idx].min_value.d128_val, scale);
          auto const max_size = fixed_point_string_size(chunks[idx].max_value.d128_val, scale);
          auto const sum_size = fixed_point_string_size(chunks[idx].sum.d128_val, scale);
          // common + total field length + encoded string lengths + strings
          stats_len = pb_fldlen_common + pb_fld_hdrlen32 + 3 * (pb_fld_hdrlen + pb_fld_hdrlen32) +
                      min_size + max_size + sum_size;
        } break;
        case dtype_string:
          stats_len = pb_fldlen_common + pb_fld_hdrlen32 + 3 * (pb_fld_hdrlen + pb_fld_hdrlen32) +
                      chunks[idx].min_value.str_val.length + chunks[idx].max_value.str_val.length;
          break;
        case dtype_none: stats_len = pb_fldlen_common;
        default: break;
      }
    }
    uint32_t tmp_stats_size;
    block_scan(temp_storage).ExclusiveSum(stats_len, stats_pos, tmp_stats_size);
    stats_pos += stats_size;
    stats_size += tmp_stats_size;
    if (idx < statistics_count) {
      groups[idx].start_chunk = stats_pos;
      groups[idx].num_chunks  = stats_len;
    }
    __syncthreads();
  }
}

struct stats_state_s {
  uint8_t* base{};  ///< Output buffer start
  uint8_t* end{};   ///< Output buffer end
  statistics_chunk chunk{};
  statistics_merge_group group{};
  statistics_dtype stats_dtype{};  //!< Statistics data type for this column
};

/*
 * Protobuf encoding - see
 * https://protobuf.dev/programming-guides/encoding/
 */
// Protobuf varint encoding for unsigned int
__device__ inline uint8_t* pb_encode_uint(uint8_t* p, uint64_t v)
{
  while (v > 0x7f) {
    *p++ = ((uint32_t)v | 0x80);
    v >>= 7;
  }
  *p++ = v;
  return p;
}

// Protobuf field encoding for unsigned int
__device__ inline uint8_t* pb_put_uint(uint8_t* p, uint32_t id, uint64_t v)
{
  p[0] = id * 8 + static_cast<ProtofType>(ProtofType::VARINT);  // NOTE: Assumes id < 16
  return pb_encode_uint(p + 1, v);
}

// Protobuf field encoding for signed int
__device__ inline uint8_t* pb_put_int(uint8_t* p, uint32_t id, int64_t v)
{
  int64_t s = (v < 0);
  return pb_put_uint(p, id, (v ^ -s) * 2 + s);
}

// Protobuf field encoding for 'packed' unsigned int (single value)
__device__ inline uint8_t* pb_put_packed_uint(uint8_t* p, uint32_t id, uint64_t v)
{
  uint8_t* p2 = pb_encode_uint(p + 2, v);
  p[0]        = id * 8 + ProtofType::FIXEDLEN;
  p[1]        = static_cast<uint8_t>(p2 - (p + 2));
  return p2;
}

// Protobuf field encoding for binary/string
__device__ inline uint8_t* pb_put_binary(uint8_t* p, uint32_t id, void const* bytes, uint32_t len)
{
  p[0] = id * 8 + ProtofType::FIXEDLEN;
  p    = pb_encode_uint(p + 1, len);
  memcpy(p, bytes, len);
  return p + len;
}

__device__ inline uint8_t* pb_put_decimal(
  uint8_t* p, uint32_t id, __int128_t value, int32_t scale, int32_t len)
{
  p[0] = id * 8 + ProtofType::FIXEDLEN;
  p    = pb_encode_uint(p + 1, len);
  strings::detail::fixed_point_to_string(value, scale, reinterpret_cast<char*>(p));
  return p + len;
}

// Protobuf field encoding for 64-bit raw encoding (double)
__device__ inline uint8_t* pb_put_fixed64(uint8_t* p, uint32_t id, void const* raw64)
{
  p[0] = id * 8 + ProtofType::FIXED64;
  memcpy(p + 1, raw64, 8);
  return p + 9;
}

// Splits a nanosecond timestamp into milliseconds and nanoseconds
__device__ cuda::std::pair<int64_t, int32_t> split_nanosecond_timestamp(int64_t nano_count)
{
  auto const ns           = cuda::std::chrono::nanoseconds(nano_count);
  auto const ms_floor     = cuda::std::chrono::floor<cuda::std::chrono::milliseconds>(ns);
  auto const ns_remainder = ns - ms_floor;
  return {ms_floor.count(), ns_remainder.count()};
}

/**
 * @brief Encode statistics in ORC protobuf format
 *
 * @param[in,out] groups Statistics merge groups
 * @param[in,out] chunks Statistics data
 * @param[in] statistics_count Number of statistics buffers
 *
 * ORC statistics format from https://orc.apache.org/specification/ORCv1/
 *
 * message ColumnStatistics {
 *  // the number of values
 *  optional uint64 numberOfValues = 1;
 *  // At most one of these has a value for any column
 *  optional IntegerStatistics intStatistics = 2;
 *  optional DoubleStatistics doubleStatistics = 3;
 *  optional StringStatistics stringStatistics = 4;
 *  optional BucketStatistics bucketStatistics = 5;
 *  optional DecimalStatistics decimalStatistics = 6;
 *  optional DateStatistics dateStatistics = 7;
 *  optional BinaryStatistics binaryStatistics = 8;
 *  optional TimestampStatistics timestampStatistics = 9;
 *  optional bool hasNull = 10;
 * }
 */
constexpr unsigned int encode_threads_per_chunk = 32;
constexpr unsigned int encode_chunks_per_block  = 4;
constexpr unsigned int encode_threads_per_block =
  encode_threads_per_chunk * encode_chunks_per_block;

CUDF_KERNEL void __launch_bounds__(encode_threads_per_block)
  gpu_encode_statistics(uint8_t* blob_bfr,
                        statistics_merge_group* groups,
                        statistics_chunk const* chunks,
                        uint32_t statistics_count)
{
  __shared__ __align__(8) stats_state_s state_g[encode_chunks_per_block];
  auto t                 = threadIdx.x;
  auto idx               = blockIdx.x * encode_chunks_per_block + threadIdx.y;
  stats_state_s* const s = &state_g[threadIdx.y];

  // Encode and update actual bfr size
  if (idx < statistics_count && t == 0) {
    s->chunk       = chunks[idx];
    s->group       = groups[idx];
    s->stats_dtype = s->group.stats_dtype;
    s->base        = blob_bfr + s->group.start_chunk;
    s->end         = blob_bfr + s->group.start_chunk + s->group.num_chunks;
    uint8_t* cur   = pb_put_uint(s->base, 1, s->chunk.non_nulls);
    cur            = pb_put_uint(cur, 10, s->chunk.null_count != 0);  // hasNull (bool)

    uint8_t* fld_start = cur;
    switch (s->stats_dtype) {
      case dtype_int8:
      case dtype_int16:
      case dtype_int32:
      case dtype_int64:
        // intStatistics = 2
        // message IntegerStatistics {
        //  optional sint64 minimum = 1;
        //  optional sint64 maximum = 2;
        //  optional sint64 sum = 3;
        // }
        {
          *cur = 2 * 8 + ProtofType::FIXEDLEN;
          cur += 2;
          if (s->chunk.has_minmax) {
            cur = pb_put_int(cur, 1, s->chunk.min_value.i_val);
            cur = pb_put_int(cur, 2, s->chunk.max_value.i_val);
          }
          if (s->chunk.has_sum) { cur = pb_put_int(cur, 3, s->chunk.sum.i_val); }
          fld_start[1] = cur - (fld_start + 2);
        }
        break;
      case dtype_float32:
      case dtype_float64:
        // doubleStatistics = 3
        // message DoubleStatistics {
        //  optional double minimum = 1;
        //  optional double maximum = 2;
        //  optional double sum = 3;
        // }
        {
          *cur = 3 * 8 + ProtofType::FIXEDLEN;
          cur += 2;
          if (s->chunk.has_minmax) {
            cur = pb_put_fixed64(cur, 1, &s->chunk.min_value.fp_val);
            cur = pb_put_fixed64(cur, 2, &s->chunk.max_value.fp_val);
          }
          if (s->chunk.has_sum) { cur = pb_put_fixed64(cur, 3, &s->chunk.sum.fp_val); }
          fld_start[1] = cur - (fld_start + 2);
        }
        break;
      case dtype_string:
        // stringStatistics = 4
        // message StringStatistics {
        //  optional string minimum = 1;
        //  optional string maximum = 2;
        //  optional sint64 sum = 3; // sum will store the total length of all strings
        // }
        {
          uint32_t sz = 0;
          if (s->chunk.has_minmax) {
            sz += (pb_put_uint(cur, 1, s->chunk.min_value.str_val.length) - cur) +
                  (pb_put_uint(cur, 2, s->chunk.max_value.str_val.length) - cur) +
                  s->chunk.min_value.str_val.length + s->chunk.max_value.str_val.length;
          }
          sz += pb_put_int(cur, 3, s->chunk.sum.i_val) - cur;

          cur[0] = 4 * 8 + ProtofType::FIXEDLEN;
          cur    = pb_encode_uint(cur + 1, sz);

          if (s->chunk.has_minmax) {
            cur = pb_put_binary(
              cur, 1, s->chunk.min_value.str_val.ptr, s->chunk.min_value.str_val.length);
            cur = pb_put_binary(
              cur, 2, s->chunk.max_value.str_val.ptr, s->chunk.max_value.str_val.length);
          }
          cur = pb_put_int(cur, 3, s->chunk.sum.i_val);
        }
        break;
      case dtype_bool:
        // bucketStatistics = 5
        // message BucketStatistics {
        //  repeated uint64 count = 1 [packed=true];
        // }
        {
          cur[0] = 5 * 8 + ProtofType::FIXEDLEN;
          // count is equal to the number of 'true' values, despite what specs say
          cur          = pb_put_packed_uint(cur + 2, 1, s->chunk.sum.u_val);
          fld_start[1] = cur - (fld_start + 2);
        }
        break;
      case dtype_decimal64:
      case dtype_decimal128:
        // decimalStatistics = 6
        // message DecimalStatistics {
        //  optional string minimum = 1;
        //  optional string maximum = 2;
        //  optional string sum = 3;
        // }
        {
          auto const scale = s->group.col_dtype.scale();

          uint32_t sz = 0;
          auto const min_size =
            s->chunk.has_minmax ? fixed_point_string_size(s->chunk.min_value.d128_val, scale) : 0;
          auto const max_size =
            s->chunk.has_minmax ? fixed_point_string_size(s->chunk.max_value.d128_val, scale) : 0;
          if (s->chunk.has_minmax) {
            // encoded string lengths, plus the strings
            sz += (pb_put_uint(cur, 1, min_size) - cur) + min_size +
                  (pb_put_uint(cur, 1, max_size) - cur) + max_size;
          }
          auto const sum_size = fixed_point_string_size(s->chunk.sum.d128_val, scale);
          sz += (pb_put_uint(cur, 1, sum_size) - cur) + sum_size;

          cur[0] = 6 * 8 + ProtofType::FIXEDLEN;
          cur    = pb_encode_uint(cur + 1, sz);

          if (s->chunk.has_minmax) {
            cur = pb_put_decimal(cur, 1, s->chunk.min_value.d128_val, scale, min_size);  //  minimum
            cur = pb_put_decimal(cur, 2, s->chunk.max_value.d128_val, scale, max_size);  // maximum
          }
          cur = pb_put_decimal(cur, 3, s->chunk.sum.d128_val, scale, sum_size);  // sum
        }
        break;
      case dtype_date32:
        // dateStatistics = 7
        // message DateStatistics { // min,max values saved as days since epoch
        //  optional sint32 minimum = 1;
        //  optional sint32 maximum = 2;
        // }
        {
          cur[0] = 7 * 8 + ProtofType::FIXEDLEN;
          cur += 2;
          if (s->chunk.has_minmax) {
            cur = pb_put_int(cur, 1, s->chunk.min_value.i_val);
            cur = pb_put_int(cur, 2, s->chunk.max_value.i_val);
          }
          fld_start[1] = cur - (fld_start + 2);
        }
        break;
      case dtype_timestamp64:
        // timestampStatistics = 9
        // message TimestampStatistics {
        //  optional sint64 minimum = 1; // min,max values saved as milliseconds since epoch
        //  optional sint64 maximum = 2;
        //  optional sint64 minimumUtc = 3; // min,max values saved as milliseconds since UNIX epoch
        //  optional sint64 maximumUtc = 4;
        //  optional int32 minimumNanos = 5; // lower 6 TS digits for min/max to achieve nanosecond
        //  precision
        // optional int32 maximumNanos = 6;
        // }
        {
          cur[0] = 9 * 8 + ProtofType::FIXEDLEN;
          cur += 2;
          if (s->chunk.has_minmax) {
            auto const [min_ms, min_ns_remainder] =
              split_nanosecond_timestamp(s->chunk.min_value.i_val);
            auto const [max_ms, max_ns_remainder] =
              split_nanosecond_timestamp(s->chunk.max_value.i_val);

            // minimum/maximum are the same as minimumUtc/maximumUtc as we always write files in UTC
            cur = pb_put_int(cur, 1, min_ms);  // minimum
            cur = pb_put_int(cur, 2, max_ms);  // maximum
            cur = pb_put_int(cur, 3, min_ms);  // minimumUtc
            cur = pb_put_int(cur, 4, max_ms);  // maximumUtc

            if constexpr (enable_nanosecond_statistics) {
              if (min_ns_remainder != DEFAULT_MIN_NANOS) {
                // using uint because positive values are not zigzag encoded
                cur = pb_put_uint(cur, 5, min_ns_remainder + 1);  // minimumNanos
              }
              if (max_ns_remainder != DEFAULT_MAX_NANOS) {
                // using uint because positive values are not zigzag encoded
                cur = pb_put_uint(cur, 6, max_ns_remainder + 1);  // maximumNanos
              }
            }
          }
          fld_start[1] = cur - (fld_start + 2);
        }
        break;
      default: break;
    }
    groups[idx].num_chunks = static_cast<uint32_t>(cur - s->base);
  }
}

void orc_init_statistics_groups(statistics_group* groups,
                                stats_column_desc const* cols,
                                device_2dspan<rowgroup_rows const> rowgroup_bounds,
                                rmm::cuda_stream_view stream)
{
  auto const num_blocks =
    cudf::util::div_rounding_up_safe<size_t>(rowgroup_bounds.size().first, init_groups_per_block) *
    rowgroup_bounds.size().second;

  dim3 dim_block(init_threads_per_group, init_groups_per_block);
  gpu_init_statistics_groups<<<num_blocks, dim_block, 0, stream.value()>>>(
    groups, cols, rowgroup_bounds);
  CUDF_CUDA_TRY(cudaGetLastError());
}

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
                                    rmm::cuda_stream_view stream)
{
  gpu_init_statistics_buffersize<block_size>
    <<<1, block_size, 0, stream.value()>>>(groups, chunks, statistics_count);
  CUDF_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief Launches kernel to encode statistics in ORC protobuf format
 *
 * @param[out] blob_bfr Output buffer for statistics blobs
 * @param[in,out] groups Statistics merge groups
 * @param[in,out] chunks Statistics data
 * @param[in] statistics_count Number of statistics buffers
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void orc_encode_statistics(uint8_t* blob_bfr,
                           statistics_merge_group* groups,
                           statistics_chunk const* chunks,
                           uint32_t statistics_count,
                           rmm::cuda_stream_view stream)
{
  auto const num_blocks =
    cudf::util::div_rounding_up_safe(statistics_count, encode_chunks_per_block);
  dim3 dim_block(encode_threads_per_chunk, encode_chunks_per_block);
  gpu_encode_statistics<<<num_blocks, dim_block, 0, stream.value()>>>(
    blob_bfr, groups, chunks, statistics_count);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::orc::detail
