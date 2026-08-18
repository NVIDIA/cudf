/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "page_data.cuh"
#include "page_decode.cuh"
#include "page_state_composed.cuh"
#include "page_string_utils.cuh"
#include "parquet_gpu.hpp"
#include "rle_stream.cuh"

#include <cudf/detail/utilities/cuda.cuh>

#include <cooperative_groups.h>
#include <cuda/std/bit>
#include <cuda/std/iterator>

namespace cudf::io::parquet::detail {

namespace {

// Unlike cub's algorithm, this provides warp-wide and block-wide results simultaneously.
// Also, this provides the ability to compute warp_bits & lane_mask manually, which we need for
// lists.
struct block_scan_results {
  uint32_t warp_bits;
  int thread_count_within_warp;
  int warp_count;

  int thread_count_within_block;
  int block_count;
};

template <int decode_block_size>
using block_scan_temp_storage = int[decode_block_size / cudf::detail::warp_size];

// Similar to CUB, must __syncthreads() after calling if reusing temp_storage
template <int decode_block_size>
__device__ inline static void scan_block_exclusive_sum(
  int thread_bit,
  block_scan_results& results,
  block_scan_temp_storage<decode_block_size>& temp_storage)
{
  int const t              = threadIdx.x;
  int const warp_index     = t / cudf::detail::warp_size;
  int const warp_lane      = t % cudf::detail::warp_size;
  uint32_t const lane_mask = (uint32_t(1) << warp_lane) - 1;

  uint32_t warp_bits = ballot(thread_bit);
  scan_block_exclusive_sum<decode_block_size>(
    warp_bits, warp_lane, warp_index, lane_mask, results, temp_storage);
}

// Similar to CUB, must __syncthreads() after calling if reusing temp_storage
template <int decode_block_size>
__device__ static void scan_block_exclusive_sum(
  uint32_t warp_bits,
  int warp_lane,
  int warp_index,
  uint32_t lane_mask,
  block_scan_results& results,
  block_scan_temp_storage<decode_block_size>& temp_storage)
{
  // Compute # warps
  constexpr int num_warps = decode_block_size / cudf::detail::warp_size;

  // Compute the warp-wide results
  results.warp_bits                = warp_bits;
  results.warp_count               = __popc(results.warp_bits);
  results.thread_count_within_warp = __popc(results.warp_bits & lane_mask);

  // Share the warp counts amongst the block threads
  if (warp_lane == 0) { temp_storage[warp_index] = results.warp_count; }
  __syncthreads();  // Sync to share counts between threads/warps

  // Compute block-wide results
  results.block_count               = 0;
  results.thread_count_within_block = results.thread_count_within_warp;
  for (int warp_idx = 0; warp_idx < num_warps; ++warp_idx) {
    results.block_count += temp_storage[warp_idx];
    if (warp_idx < warp_index) { results.thread_count_within_block += temp_storage[warp_idx]; }
  }
}

/**
 * @brief Write a batch of decoded dictionary indices directly as INT32 output values.
 *
 * Used by the Parquet-dict → DICTIONARY32 transcode path: instead of materializing the dictionary
 * keys, the per-row dictionary indices are emitted verbatim as the INT32 indices child of the
 * output DICTIONARY32 column.
 *
 * @tparam block_size Number of threads per block
 * @tparam has_lists_t Whether the column has a list (repetition) level
 * @tparam copy_mode_t Whether destination positions are direct or indirect (nz_idx) mapped
 * @tparam state_buf Page state buffer type providing the decoded dictionary indices
 * @param s Page decode state for the current page
 * @param sb Page state buffers holding the decoded dictionary indices
 * @param start First value position (within the page) to write in this batch
 * @param end One-past-the-last value position to write in this batch
 * @param t Thread index within the block
 */
template <int block_size, bool has_lists_t, copy_mode copy_mode_t, typename state_buf>
__device__ void decode_dict_indices_as_int32(
  auto* s, state_buf* const sb, int start, int end, int t)
{
  constexpr int num_warps      = block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  int const leaf_level_index = s->setup.col.max_nesting_depth - 1;
  auto const data_out        = s->nesting.nesting_info[leaf_level_index].data_out;

  int const skipped_leaf_values = s->setup.page.skipped_leaf_values;

  int pos = start;
  while (pos < end) {
    int const batch_size = min(max_batch_size, end - pos);
    int const target_pos = pos + batch_size;
    int const thread_pos = pos + t;

    int const dst_pos = [&]() {
      if constexpr (copy_mode_t == copy_mode::DIRECT) {
        return thread_pos - s->setup.first_row;
      } else {
        int dst_pos = sb->nz_idx[rolling_index<state_buf::nz_buf_size>(thread_pos)];
        if constexpr (!has_lists_t) { dst_pos -= s->setup.first_row; }
        return dst_pos;
      }
    }();

    if (thread_pos < target_pos && dst_pos >= 0) {
      int const src_pos = [&]() {
        if constexpr (has_lists_t) { return thread_pos + skipped_leaf_values; }
        return thread_pos;
      }();

      auto* dst           = reinterpret_cast<int32_t*>(data_out) + dst_pos;
      auto const num_keys = static_cast<uint32_t>(s->stream.dict_size / sizeof(string_index_pair));
      auto const idx      = sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)];
      if (idx >= num_keys) {
        s->set_error_code(decode_error::DATA_STREAM_OVERRUN);
      } else {
        *dst = idx;
      }
    }

    pos += batch_size;
    __syncthreads();
  }
}

template <int block_size, bool has_lists_t, copy_mode copy_mode_t, typename state_buf>
__device__ void decode_fixed_width_values(auto* s, state_buf* const sb, int start, int end, int t)
{
  constexpr int num_warps      = block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  // nesting level that is storing actual leaf values
  int const leaf_level_index = s->setup.col.max_nesting_depth - 1;
  auto const data_out        = s->nesting.nesting_info[leaf_level_index].data_out;

  Type const dtype         = s->setup.col.physical_type;
  uint32_t const dtype_len = s->output_cvt.dtype_len;

  int const skipped_leaf_values = s->setup.page.skipped_leaf_values;

  // decode values
  int thread_pos = start + t;
  while (thread_pos < end) {
    // Index from value buffer (doesn't include nulls) to final array (has gaps for nulls)
    int const dst_pos = [&]() {
      if constexpr (copy_mode_t == copy_mode::DIRECT) {
        return thread_pos - s->setup.first_row;
      } else {
        int dst_pos = sb->nz_idx[rolling_index<state_buf::nz_buf_size>(thread_pos)];
        if constexpr (!has_lists_t) { dst_pos -= s->setup.first_row; }
        return dst_pos;
      }
    }();

    // nesting level that is storing actual leaf values

    // src_pos represents the logical row position we want to read from. But in the case of
    // nested hierarchies (lists), there is no 1:1 mapping of rows to values. So src_pos
    // has to take into account the # of values we have to skip in the page to get to the
    // desired logical row.  For flat hierarchies, skipped_leaf_values will always be 0.
    int const src_pos = [&]() {
      if constexpr (has_lists_t) { return thread_pos + skipped_leaf_values; }
      return thread_pos;
    }();

    void* const dst = data_out + (static_cast<size_t>(dst_pos) * dtype_len);

    if (s->setup.col.logical_type.has_value() &&
        s->setup.col.logical_type->type == LogicalType::DECIMAL) {
      switch (dtype) {
        case Type::INT32:
          read_fixed_width_value_fast(s, sb, src_pos, static_cast<uint32_t*>(dst));
          break;
        case Type::INT64:
          read_fixed_width_value_fast(s, sb, src_pos, static_cast<uint2*>(dst));
          break;
        default:
          if (s->output_cvt.dtype_len_in <= sizeof(int32_t)) {
            read_fixed_width_byte_array_as_int(s, sb, src_pos, static_cast<int32_t*>(dst));
          } else if (s->output_cvt.dtype_len_in <= sizeof(int64_t)) {
            read_fixed_width_byte_array_as_int(s, sb, src_pos, static_cast<int64_t*>(dst));
          } else {
            read_fixed_width_byte_array_as_int(s, sb, src_pos, static_cast<__int128_t*>(dst));
          }
          break;
      }
    } else if (dtype == Type::BOOLEAN) {
      read_boolean(sb, src_pos, static_cast<uint8_t*>(dst));
    } else if (dtype == Type::INT96) {
      read_int96_timestamp(s, sb, src_pos, static_cast<int64_t*>(dst));
    } else if (dtype_len == 8) {
      if (s->output_cvt.dtype_len_in == 4) {
        // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
        // TIME_MILLIS is the only duration type stored as int32:
        // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype
        auto const dst_ptr = static_cast<uint32_t*>(dst);
        read_fixed_width_value_fast(s, sb, src_pos, dst_ptr);
        // zero out most significant bytes
        cuda::std::memset(dst_ptr + 1, 0, sizeof(int32_t));
      } else if (s->output_cvt.ts_scale) {
        read_int64_timestamp(s, sb, src_pos, static_cast<int64_t*>(dst));
      } else {
        read_fixed_width_value_fast(s, sb, src_pos, static_cast<uint2*>(dst));
      }
    } else if (dtype_len == 4) {
      read_fixed_width_value_fast(s, sb, src_pos, static_cast<uint32_t*>(dst));
    } else {
      read_nbyte_fixed_width_value(s, sb, src_pos, static_cast<uint8_t*>(dst), dtype_len);
    }

    thread_pos += max_batch_size;
  }
}

template <int block_size, bool has_lists_t, copy_mode copy_mode_t, typename state_buf>
__device__ inline void decode_fixed_width_split_values(
  auto* s, state_buf* const sb, int start, int end, int t)
{
  using cudf::detail::warp_size;
  constexpr int num_warps      = block_size / warp_size;
  constexpr int max_batch_size = num_warps * warp_size;

  // nesting level that is storing actual leaf values
  int const leaf_level_index = s->setup.col.max_nesting_depth - 1;
  auto const data_out        = s->nesting.nesting_info[leaf_level_index].data_out;

  Type const dtype    = s->setup.col.physical_type;
  auto const data_len = cuda::std::distance(s->stream.data_start, s->stream.data_end);

  // Check malformed BYTE_STREAM_SPLIT pages
  if (s->output_cvt.dtype_len_in <= 0 or data_len <= 0) {
    if (t == 0) { s->set_error_code(decode_error::INVALID_BYTE_STREAM_SPLIT_SIZE); }
    return;
  }

  auto const num_values = data_len / s->output_cvt.dtype_len_in;

  int const skipped_leaf_values = s->setup.page.skipped_leaf_values;

  // decode values
  int thread_pos = start + t;
  while (thread_pos < end) {
    // Index from value buffer (doesn't include nulls) to final array (has gaps for nulls)
    int const dst_pos = [&]() {
      if constexpr (copy_mode_t == copy_mode::DIRECT) {
        return thread_pos - s->setup.first_row;
      } else {
        int dst_pos = sb->nz_idx[rolling_index<state_buf::nz_buf_size>(thread_pos)];
        if constexpr (!has_lists_t) { dst_pos -= s->setup.first_row; }
        return dst_pos;
      }
    }();

    // src_pos represents the logical row position we want to read from. But in the case of
    // nested hierarchies (lists), there is no 1:1 mapping of rows to values. So src_pos
    // has to take into account the # of values we have to skip in the page to get to the
    // desired logical row.  For flat hierarchies, skipped_leaf_values will always be 0.
    int const src_pos = [&]() {
      if constexpr (has_lists_t) {
        return thread_pos + skipped_leaf_values;
      } else {
        return thread_pos;
      }
    }();

    uint32_t const dtype_len = s->output_cvt.dtype_len;
    uint8_t const* const src = s->stream.data_start + src_pos;
    uint8_t* const dst       = data_out + static_cast<size_t>(dst_pos) * dtype_len;
    auto const is_decimal    = s->setup.col.logical_type.has_value() and
                            s->setup.col.logical_type->type == LogicalType::DECIMAL;

    // Note: non-decimal FIXED_LEN_BYTE_ARRAY will be handled in the string reader
    if (is_decimal) {
      switch (dtype) {
        case Type::INT32: gpuOutputByteStreamSplit<int32_t>(dst, src, num_values); break;
        case Type::INT64: gpuOutputByteStreamSplit<int64_t>(dst, src, num_values); break;
        case Type::FIXED_LEN_BYTE_ARRAY:
          if (s->output_cvt.dtype_len_in <= sizeof(int32_t)) {
            gpuOutputSplitFixedLenByteArrayAsInt(
              reinterpret_cast<int32_t*>(dst), src, num_values, s->output_cvt.dtype_len_in);
            break;
          } else if (s->output_cvt.dtype_len_in <= sizeof(int64_t)) {
            gpuOutputSplitFixedLenByteArrayAsInt(
              reinterpret_cast<int64_t*>(dst), src, num_values, s->output_cvt.dtype_len_in);
            break;
          } else if (s->output_cvt.dtype_len_in <= sizeof(__int128_t)) {
            gpuOutputSplitFixedLenByteArrayAsInt(
              reinterpret_cast<__int128_t*>(dst), src, num_values, s->output_cvt.dtype_len_in);
            break;
          }
          // unsupported decimal precision
          [[fallthrough]];

        default: s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
      }
    } else if (dtype_len == 8) {
      if (s->output_cvt.dtype_len_in == 4) {
        // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
        // TIME_MILLIS is the only duration type stored as int32:
        // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype
        gpuOutputByteStreamSplit<int32_t>(dst, src, num_values);
        // zero out most significant bytes
        cuda::std::memset(dst + sizeof(int32_t), 0, sizeof(int32_t));
      } else if (s->output_cvt.ts_scale) {
        gpuOutputSplitInt64Timestamp(
          reinterpret_cast<int64_t*>(dst), src, num_values, s->output_cvt.ts_scale);
      } else {
        gpuOutputByteStreamSplit<int64_t>(dst, src, num_values);
      }
    } else if (dtype_len == 4) {
      gpuOutputByteStreamSplit<int32_t>(dst, src, num_values);
    } else {
      s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
    }

    thread_pos += max_batch_size;
  }
}

/**
 * @brief Skip validity and row indices for non-list types
 *
 * @tparam decode_block_size Size of the thread block
 * @tparam level_t Definition level type
 *
 * @param target_value_count The target value count to process
 * @param s Pointer to page state
 * @param def Pointer to the definition levels
 * @param is_nested Whether the column is nested
 * @param t Thread index
 *
 * @return Maximum depth valid count after skipping
 */
template <int decode_block_size, typename level_t>
__device__ int skip_validity_and_row_indices_nonlist(
  int32_t target_value_count, auto* s, level_t const* const def, bool is_nested, int t)
{
  int const max_def_level =
    is_nested ? s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1].max_def_level : 1;

  int max_depth_valid_count = 0;
  int value_count           = 0;
  while (value_count < target_value_count) {
    int const batch_size = min(decode_block_size, target_value_count - value_count);

    // definition level
    int const is_valid = (t >= batch_size) ? 0 : ((def[value_count + t] >= max_def_level) ? 1 : 0);

    // thread and block validity count
    using block_scan = cub::BlockScan<int, decode_block_size>;
    __shared__ typename block_scan::TempStorage scan_storage;
    int thread_valid_count = 0, block_valid_count = 0;
    block_scan(scan_storage).ExclusiveSum(is_valid, thread_valid_count, block_valid_count);

    value_count += batch_size;
    max_depth_valid_count += block_valid_count;
  }  // end loop

  return max_depth_valid_count;
}

/**
 * @brief Update validity and row indices for nested types
 *
 * @tparam decode_block_size Size of the thread block
 * @tparam level_t Definition level type
 * @tparam state_buf State buffer type
 *
 * @param target_value_count The target value count to process
 * @param s Pointer to  page state
 * @param sb Pointer to  state buffer
 * @param def Pointer to the definition levels
 * @param t Thread index
 *
 * @return Maximum depth valid count after processing
 */
template <int decode_block_size, typename level_t, typename state_buf>
__device__ int update_validity_and_row_indices_nested(
  int32_t target_value_count, auto* s, state_buf* sb, level_t const* const def, int t)
{
  constexpr int num_warps      = decode_block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  // how many (input) values we've processed in the page so far
  int value_count = s->progress.input_value_count;

  // cap by last row so that we don't process any rows past what we want to output.
  int const first_row                 = s->setup.first_row;
  int const last_row                  = first_row + s->setup.num_rows;
  int const capped_target_value_count = min(target_value_count, last_row);

  int const max_depth       = s->setup.col.max_nesting_depth - 1;
  auto& max_depth_ni        = s->nesting.nesting_info[max_depth];
  int max_depth_valid_count = max_depth_ni.valid_count;

  __syncthreads();

  while (value_count < capped_target_value_count) {
    int const batch_size = min(max_batch_size, capped_target_value_count - value_count);

    // definition level
    int const def_level = (t >= batch_size) ? -1 : def[value_count + t];

    int const thread_value_count = t;
    int const block_value_count  = batch_size;

    // compute our row index, whether we're in row bounds, and validity
    // This ASSUMES that s->progress.row_index_lower_bound is always -1!
    // Its purpose is to handle rows than span page boundaries, which only happen for lists.
    int const row_index                     = thread_value_count + value_count;
    int const in_row_bounds                 = (row_index < last_row);
    bool const in_write_row_bounds          = in_row_bounds && (row_index >= first_row);
    uint32_t const in_write_row_bounds_mask = ballot(in_write_row_bounds);
    // NOTE: The below CANNOT be std::countr_zero(), because for zero start must be 0 not 32
    int const write_start = __ffs(in_write_row_bounds_mask) - 1;  // first bit in the warp to store

    // iterate by depth
    for (int d_idx = 0; d_idx <= max_depth; d_idx++) {
      auto& ni = s->nesting.nesting_info[d_idx];

      int const is_valid = ((def_level >= ni.max_def_level) && in_row_bounds) ? 1 : 0;

      // thread and block validity count
      using block_scan = cub::BlockScan<int, decode_block_size>;
      __shared__ typename block_scan::TempStorage scan_storage;
      int thread_valid_count, block_valid_count;
      block_scan(scan_storage).ExclusiveSum(is_valid, thread_valid_count, block_valid_count);

      // validity is processed per-warp
      //
      // nested schemas always read and write to the same bounds (that is, read and write
      // positions are already pre-bounded by first_row/num_rows). flat schemas will start reading
      // at the first value, even if that is before first_row, because we cannot trivially jump to
      // the correct position to start reading. since we are about to write the validity vector
      // here we need to adjust our computed mask to take into account the write row bounds.
      if (ni.valid_map != nullptr) {
        uint32_t const warp_validity_mask = ballot(is_valid);
        // lane 0 from each warp writes out validity
        if ((write_start >= 0) && ((t % cudf::detail::warp_size) == 0)) {
          int const valid_map_offset = ni.valid_map_offset;
          int const vindex     = value_count + thread_value_count;  // absolute input value index
          int const bit_offset = (valid_map_offset + vindex + write_start) -
                                 first_row;  // absolute bit offset into the output validity map
          int const write_end = cudf::detail::warp_size -
                                __clz(in_write_row_bounds_mask);  // last bit in the warp to store
          int const bit_count = write_end - write_start;

          store_validity(bit_offset, ni.valid_map, warp_validity_mask >> write_start, bit_count);
        }
      }

      if (t == 0) { ni.null_count += (block_value_count - block_valid_count); }

      // if this is valid and we're at the leaf, output dst_pos
      if (d_idx == max_depth) {
        if (is_valid) {
          int const dst_pos = value_count + thread_value_count;
          int const src_pos = max_depth_valid_count + thread_valid_count;

          sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)] = dst_pos;
        }
        // update stuff
        max_depth_valid_count += block_valid_count;
      }

    }  // end depth loop

    value_count += block_value_count;
  }  // end loop

  if (t == 0) {
    // update valid value count for decoding and total # of values we've processed
    max_depth_ni.valid_count      = max_depth_valid_count;
    max_depth_ni.value_count      = value_count;  // Needed AT LEAST for strings!
    s->progress.nz_count          = max_depth_valid_count;
    s->progress.input_value_count = value_count;
    s->progress.input_row_count   = value_count;
  }

  return max_depth_valid_count;
}

/**
 * @brief Update validity and row indices for flat types
 *
 * @tparam decode_block_size Size of the thread block
 * @tparam level_t Definition level type
 * @tparam state_buf State buffer type
 *
 * @param target_value_count The target value count to process
 * @param s Pointer to  page state
 * @param sb Pointer to  state buffer
 * @param def Pointer to the definition levels
 * @param t Thread index
 *
 * @return Maximum depth valid count after processing
 */
template <int decode_block_size, typename level_t, typename state_buf>
__device__ int update_validity_and_row_indices_flat(
  int32_t target_value_count, auto* s, state_buf* sb, level_t const* const def, int t)
{
  constexpr int num_warps      = decode_block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  auto& ni = s->nesting.nesting_info[0];

  // how many (input) values we've processed in the page so far
  int value_count = s->progress.input_value_count;
  int valid_count = ni.valid_count;

  // cap by last row so that we don't process any rows past what we want to output.
  int const first_row                 = s->setup.first_row;
  int const last_row                  = first_row + s->setup.num_rows;
  int const capped_target_value_count = min(target_value_count, last_row);

  int const valid_map_offset = ni.valid_map_offset;

  __syncthreads();

  while (value_count < capped_target_value_count) {
    int const batch_size = min(max_batch_size, capped_target_value_count - value_count);

    int const thread_value_count = t;
    int const block_value_count  = batch_size;

    // compute our row index, whether we're in row bounds, and validity
    // This ASSUMES that s->progress.row_index_lower_bound is always -1!
    // Its purpose is to handle rows than span page boundaries, which only happen for lists.
    int const row_index     = thread_value_count + value_count;
    int const in_row_bounds = (row_index < last_row);

    // use definition level & row bounds to determine if is valid
    int const is_valid =
      ((t >= batch_size) || !in_row_bounds) ? 0 : ((def[value_count + t] > 0) ? 1 : 0);

    // thread and block validity count
    using block_scan = cub::BlockScan<int, decode_block_size>;
    __shared__ typename block_scan::TempStorage scan_storage;
    int thread_valid_count, block_valid_count;
    block_scan(scan_storage).ExclusiveSum(is_valid, thread_valid_count, block_valid_count);
    uint32_t const warp_validity_mask = ballot(is_valid);

    // validity is processed per-warp
    //
    // nested schemas always read and write to the same bounds (that is, read and write
    // positions are already pre-bounded by first_row/num_rows). flat schemas will start reading
    // at the first value, even if that is before first_row, because we cannot trivially jump to
    // the correct position to start reading. since we are about to write the validity vector
    // here we need to adjust our computed mask to take into account the write row bounds.
    bool const in_write_row_bounds     = in_row_bounds && (row_index >= first_row);
    int const in_write_row_bounds_mask = ballot(in_write_row_bounds);
    // NOTE: The below CANNOT be std::countr_zero(), because for zero start must be 0 not 32
    int const write_start = __ffs(in_write_row_bounds_mask) - 1;  // first bit in the warp to store
    // lane 0 from each warp writes out validity
    if ((write_start >= 0) && ((t % cudf::detail::warp_size) == 0)) {
      int const vindex     = value_count + thread_value_count;  // absolute input value index
      int const bit_offset = (valid_map_offset + vindex + write_start) -
                             first_row;  // absolute bit offset into the output validity map
      int const write_end =
        cudf::detail::warp_size - __clz(in_write_row_bounds_mask);  // last bit in the warp to store
      int const bit_count = write_end - write_start;

      store_validity(bit_offset, ni.valid_map, warp_validity_mask >> write_start, bit_count);
    }

    if (t == 0) { ni.null_count += block_value_count - block_valid_count; }

    // output offset
    if (is_valid) {
      int const dst_pos = value_count + thread_value_count;
      int const src_pos = valid_count + thread_valid_count;

      sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)] = dst_pos;
    }

    // update stuff
    value_count += block_value_count;
    valid_count += block_valid_count;
  }

  if (t == 0) {
    // update valid value count for decoding and total # of values we've processed
    ni.valid_count                = valid_count;
    ni.value_count                = value_count;
    s->progress.nz_count          = valid_count;
    s->progress.input_value_count = value_count;
    s->progress.input_row_count   = value_count;
  }

  return valid_count;
}

/**
 * @brief Update validity and row indices for list types
 *
 * @tparam decode_block_size Size of the thread block
 * @tparam level_t Definition level type
 * @tparam state_buf State buffer type
 *
 * @param target_value_count The target value count to process
 * @param s Pointer to page state
 * @param sb Pointer to state buffer
 * @param def Pointer to the definition levels
 * @param rep Repetition level decoder
 * @param t Thread index
 *
 * @return Maximum depth valid count after processing
 */
template <int decode_block_size, bool nullable, typename level_t, typename state_buf>
__device__ int update_validity_and_row_indices_lists(int32_t target_value_count,
                                                     auto* s,
                                                     state_buf* sb,
                                                     level_t const* const def,
                                                     level_t const* const rep,
                                                     int t)
{
  constexpr int num_warps      = decode_block_size / cudf::detail::warp_size;
  constexpr int max_batch_size = num_warps * cudf::detail::warp_size;

  // how many (input) values we've processed in the page so far, prior to this loop iteration
  int value_count = s->progress.input_value_count;

  // how many rows we've processed in the page so far
  int input_row_count = s->progress.input_row_count;

  // cap by last row so that we don't process any rows past what we want to output.
  int const first_row = s->setup.first_row;
  int const last_row  = first_row + s->setup.num_rows;

  int const row_index_lower_bound = s->progress.row_index_lower_bound;
  int const max_depth             = s->setup.col.max_nesting_depth - 1;
  int max_depth_valid_count       = s->nesting.nesting_info[max_depth].valid_count;

  int const warp_index     = t / cudf::detail::warp_size;
  int const warp_lane      = t % cudf::detail::warp_size;
  bool const is_first_lane = (warp_lane == 0);

  __syncthreads();
  __shared__ block_scan_temp_storage<decode_block_size> temp_storage;

  while (value_count < target_value_count) {
    bool const within_batch = value_count + t < target_value_count;

    // get definition level, use repetition level to get start/end depth
    // different for each thread, as each thread has a different r/d
    auto const [def_level, start_depth, end_depth] = [&]() {
      if (!within_batch) { return cuda::std::make_tuple(-1, -1, -1); }

      auto const rep_level  = rep[value_count + t];
      int const start_depth = s->nesting.nesting_info[rep_level].start_depth;

      if constexpr (!nullable) {
        return cuda::std::make_tuple(-1, start_depth, max_depth);
      } else {
        int const def_level = def[value_count + t];
        return cuda::std::make_tuple(
          def_level, start_depth, s->nesting.nesting_info[def_level].end_depth);
      }
    }();

    // Determine value count & row index
    //  track (page-relative) row index for the thread so we can compare against input bounds
    //  keep track of overall # of rows we've read.
    int const is_new_row = start_depth == 0 ? 1 : 0;
    int num_prior_new_rows, total_num_new_rows;
    {
      block_scan_results new_row_scan_results;
      scan_block_exclusive_sum<decode_block_size>(is_new_row, new_row_scan_results, temp_storage);
      __syncthreads();
      num_prior_new_rows = new_row_scan_results.thread_count_within_block;
      total_num_new_rows = new_row_scan_results.block_count;
    }

    int const row_index = input_row_count + ((num_prior_new_rows + is_new_row) - 1);
    input_row_count += total_num_new_rows;
    int const in_row_bounds = (row_index >= row_index_lower_bound) && (row_index < last_row);

    // VALUE COUNT:
    // in_nesting_bounds: if at a nesting level where we need to add value indices
    // the bounds: from current rep to the rep AT the def depth
    int in_nesting_bounds = ((0 >= start_depth && 0 <= end_depth) && in_row_bounds) ? 1 : 0;
    int thread_value_count_within_warp, warp_value_count, thread_value_count, block_value_count;
    {
      block_scan_results value_count_scan_results;
      scan_block_exclusive_sum<decode_block_size>(
        in_nesting_bounds, value_count_scan_results, temp_storage);
      __syncthreads();

      thread_value_count_within_warp = value_count_scan_results.thread_count_within_warp;
      warp_value_count               = value_count_scan_results.warp_count;
      thread_value_count             = value_count_scan_results.thread_count_within_block;
      block_value_count              = value_count_scan_results.block_count;
    }

    // iterate by depth
    for (int d_idx = 0; d_idx <= max_depth; d_idx++) {
      auto& ni = s->nesting.nesting_info[d_idx];

      // everything up to the max_def_level is a non-null value
      int const is_valid = [&](int input_def_level) {
        if constexpr (nullable) {
          return ((input_def_level >= ni.max_def_level) && in_nesting_bounds) ? 1 : 0;
        } else {
          return in_nesting_bounds;
        }
      }(def_level);

      // VALID COUNT:
      // Not all values visited by this block will represent a value at this nesting level.
      // the validity bit for thread t might actually represent output value t-6.
      // the correct position for thread t's bit is thread_value_count.
      uint32_t const warp_valid_mask = warp_reduce_or<cudf::detail::warp_size>(
        static_cast<uint32_t>(is_valid) << thread_value_count_within_warp);
      int thread_valid_count, block_valid_count;
      {
        auto thread_mask = (uint32_t(1) << thread_value_count_within_warp) - 1;

        block_scan_results valid_count_scan_results;
        scan_block_exclusive_sum<decode_block_size>(warp_valid_mask,
                                                    warp_lane,
                                                    warp_index,
                                                    thread_mask,
                                                    valid_count_scan_results,
                                                    temp_storage);
        __syncthreads();
        thread_valid_count = valid_count_scan_results.thread_count_within_block;
        block_valid_count  = valid_count_scan_results.block_count;
      }

      // compute warp and thread value counts for the -next- nesting level. we need to
      // do this for lists so that we can emit an offset for the -current- nesting level.
      // the offset for the current nesting level == current length of the next nesting level
      int next_thread_value_count_within_warp = 0, next_warp_value_count = 0;
      int next_thread_value_count = 0, next_block_value_count = 0;
      int next_in_nesting_bounds = 0;
      if (d_idx < max_depth) {
        // NEXT DEPTH VALUE COUNT:
        next_in_nesting_bounds =
          ((d_idx + 1 >= start_depth) && (d_idx + 1 <= end_depth) && in_row_bounds) ? 1 : 0;
        {
          block_scan_results next_value_count_scan_results;
          scan_block_exclusive_sum<decode_block_size>(
            next_in_nesting_bounds, next_value_count_scan_results, temp_storage);
          __syncthreads();

          next_thread_value_count_within_warp =
            next_value_count_scan_results.thread_count_within_warp;
          next_warp_value_count   = next_value_count_scan_results.warp_count;
          next_thread_value_count = next_value_count_scan_results.thread_count_within_block;
          next_block_value_count  = next_value_count_scan_results.block_count;
        }

        // STORE OFFSET TO THE LIST LOCATION
        // if we're -not- at a leaf column and we're within nesting/row bounds
        // and we have a valid data_out pointer, it implies this is a list column, so
        // emit an offset.
        if (in_nesting_bounds && ni.data_out != nullptr) {
          auto const& next_ni = s->nesting.nesting_info[d_idx + 1];
          int const idx       = ni.value_count + thread_value_count;
          cudf::size_type const ofs =
            next_ni.value_count + next_thread_value_count + next_ni.page_start_value;

          (reinterpret_cast<cudf::size_type*>(ni.data_out))[idx] = ofs;
        }
      }

      // validity is processed per-warp (on lane 0's)
      // thi is because when atomic writes are needed, they are 32-bit operations
      //
      // lists always read and write to the same bounds
      // (that is, read and write positions are already pre-bounded by first_row/num_rows).
      // since we are about to write the validity vector
      // here we need to adjust our computed mask to take into account the write row bounds.
      if constexpr (nullable) {
        if (is_first_lane && (ni.valid_map != nullptr) && (warp_value_count > 0)) {
          // absolute bit offset into the output validity map
          // is cumulative sum of warp_value_count at the given nesting depth
          // DON'T subtract by first_row: since it's lists it's not 1-row-per-value
          int const bit_offset = ni.valid_map_offset + thread_value_count;

          store_validity(bit_offset, ni.valid_map, warp_valid_mask, warp_value_count);
        }

        if (t == 0) { ni.null_count += block_value_count - block_valid_count; }
      }

      // if this is valid and we're at the leaf, output dst_pos
      // Read value_count before the sync, so that when thread 0 modifies it we've already read its
      // value
      int const current_value_count = ni.value_count;
      __syncthreads();  // guard against modification of ni.value_count below
      if (d_idx == max_depth) {
        if (is_valid) {
          int const dst_pos      = current_value_count + thread_value_count;
          int const src_pos      = max_depth_valid_count + thread_valid_count;
          int const output_index = rolling_index<state_buf::nz_buf_size>(src_pos);

          // Index from rolling buffer of values (which doesn't include nulls) to final array (which
          // includes gaps for nulls)
          sb->nz_idx[output_index] = dst_pos;
        }
        max_depth_valid_count += block_valid_count;
      }

      // update stuff
      if (t == 0) {
        ni.value_count += block_value_count;
        ni.valid_map_offset += block_value_count;
      }
      __syncthreads();  // sync modification of ni.value_count

      // propagate value counts for the next depth level
      block_value_count              = next_block_value_count;
      thread_value_count             = next_thread_value_count;
      in_nesting_bounds              = next_in_nesting_bounds;
      warp_value_count               = next_warp_value_count;
      thread_value_count_within_warp = next_thread_value_count_within_warp;
    }  // END OF DEPTH LOOP

    int const batch_size = min(max_batch_size, target_value_count - value_count);
    value_count += batch_size;
  }

  if (t == 0) {
    // update valid value count for decoding and total # of values we've processed
    s->nesting.nesting_info[max_depth].valid_count = max_depth_valid_count;
    s->progress.nz_count                           = max_depth_valid_count;
    s->progress.input_value_count                  = value_count;

    // If we have lists # rows != # values
    s->progress.input_row_count = input_row_count;
  }

  return max_depth_valid_count;
}

template <typename state_buf, typename thread_group>
inline __device__ void bool_plain_decode(auto* s,
                                         state_buf* sb,
                                         int target_pos,
                                         thread_group const& group)
{
  int const pos = s->progress.dict_pos;
  int const t   = group.thread_rank();
  // Ensure all threads have the dict_pos
  group.sync();

  for (auto bit_pos = pos + t; bit_pos < target_pos; bit_pos += group.size()) {
    int const byte_offset       = bit_pos >> 3;
    int const bit_in_byte_index = bit_pos & 7;

    uint8_t const* const read_from = s->stream.data_start + byte_offset;
    bool const read_bit            = (*read_from) & (1 << bit_in_byte_index);

    int const write_to_index     = rolling_index<state_buf::dict_buf_size>(bit_pos);
    sb->dict_idx[write_to_index] = read_bit;
  }
}

template <int rolling_buf_size, typename stream_type>
__device__ int skip_decode(stream_type& parquet_stream, int num_to_skip, int t)
{
  // it could be that (e.g.) we skip 5000 but starting at row 4000 we have a run of length 2000:
  // in that case skip_decode() only skips 4000, and we have to process the remaining 1000 up front
  // modulo 2 * block_size of course, since that's as many as we process at once
  int num_skipped = parquet_stream.skip_decode(t, num_to_skip);
  while (num_skipped < num_to_skip) {
    // TODO: Instead of decoding, skip within the run to the appropriate location
    auto const to_decode = min(rolling_buf_size, num_to_skip - num_skipped);
    num_skipped += parquet_stream.decode_next(t, to_decode);
    __syncthreads();
  }

  return num_skipped;
}

template <int decode_block_size_t,
          int rolling_buf_size,
          bool has_lists_t,
          bool has_dict_t,
          bool has_bools_t,
          bool has_nesting_t,
          typename level_t,
          typename dict_stream_t,
          typename bool_stream_t>
__device__ void skip_ahead_in_decoding(auto* s,
                                       dict_stream_t& dict_stream,
                                       bool_stream_t& bool_stream,
                                       bool bools_are_rle_stream,
                                       bool process_nulls,
                                       level_t const* const def,
                                       int& processed_count,
                                       int& valid_count)
{
  auto const block = cg::this_thread_block();
  int const t      = block.thread_rank();

  auto skip_bools = [&](int num_to_skip) {
    if (bools_are_rle_stream) {
      skip_decode<rolling_buf_size>(bool_stream, num_to_skip, t);
    } else {
      if (t == 0) { s->progress.dict_pos = num_to_skip; }
      // For non-lists don't sync: we'll sync at the end of the lambda instead
      if constexpr (has_lists_t) { block.sync(); }
    }
  };

  if constexpr (has_lists_t) {
    auto const skipped_leaf_values = s->setup.page.skipped_leaf_values;
    if (skipped_leaf_values > 0) {
      processed_count = skipped_leaf_values;
      if constexpr (has_dict_t) {
        skip_decode<rolling_buf_size>(dict_stream, skipped_leaf_values, t);
      } else if constexpr (has_bools_t) {
        skip_bools(skipped_leaf_values);
      }
    }
    return;
  }

  // Non-lists
  int const first_row = s->setup.first_row;
  if (first_row <= 0) { return; }  // Nothing to skip

  // Count the number of valids we're skipping.
  processed_count = first_row;
  valid_count     = !process_nulls
                      ? first_row
                      : skip_validity_and_row_indices_nonlist<decode_block_size_t, level_t>(
                      first_row, s, def, has_nesting_t, t);

  if constexpr (has_dict_t) {
    skip_decode<rolling_buf_size>(dict_stream, valid_count, t);
  } else if constexpr (has_bools_t) {
    skip_bools(valid_count);
  }

  if (t == 0) {
    int const max_depth = s->setup.col.max_nesting_depth - 1;
    auto& ni            = s->nesting.nesting_info[max_depth];

    // update valid value count for decoding and total # of values we've processed
    ni.valid_count                = valid_count;
    ni.value_count                = processed_count;
    s->progress.nz_count          = valid_count;
    s->progress.input_value_count = processed_count;
    s->progress.input_row_count   = processed_count;
  }
  block.sync();
}

/**
 * @brief Check if the kernel mask decodes dictionary-encoded data (has a dictionary stream).
 *
 * @tparam kernel_mask_t The decode kernel mask to test
 * @return True for fixed-width, string and INT32-index dictionary masks
 */
template <decode_kernel_mask kernel_mask_t>
CUDF_HOST_DEVICE constexpr bool has_dict()
{
  return (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_DICT) ||
         (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_DICT_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_DICT_LIST) ||
         (kernel_mask_t == decode_kernel_mask::STRING_DICT) ||
         (kernel_mask_t == decode_kernel_mask::STRING_DICT_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::STRING_DICT_LIST) ||
         (kernel_mask_t == decode_kernel_mask::DICT_INT32);
}

/**
 * @brief Check whether the kernel mask decodes parquet dictionary indices directly to an INT32
 * column.
 *
 * @tparam kernel_mask_t The decode kernel mask to test
 * @return True for the DICT_INT32 mask
 */
template <decode_kernel_mask kernel_mask_t>
CUDF_HOST_DEVICE constexpr bool is_dict_int32_output()
{
  return (kernel_mask_t == decode_kernel_mask::DICT_INT32);
}

template <decode_kernel_mask kernel_mask_t>
CUDF_HOST_DEVICE constexpr bool has_bools()
{
  return (kernel_mask_t == decode_kernel_mask::BOOLEAN) ||
         (kernel_mask_t == decode_kernel_mask::BOOLEAN_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::BOOLEAN_LIST);
}

/**
 * @brief Check if the kernel mask decodes a (non-list) nested column.
 *
 * @tparam kernel_mask_t The decode kernel mask to test
 * @return True for the `*_NESTED` masks
 */
template <decode_kernel_mask kernel_mask_t>
CUDF_HOST_DEVICE constexpr bool has_nesting()
{
  return (kernel_mask_t == decode_kernel_mask::BOOLEAN_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_DICT_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::STRING_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::STRING_DICT_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::STRING_STREAM_SPLIT_NESTED);
}

/**
 * @brief Check if the kernel mask decodes a list column (has a repetition level).
 *
 * @tparam kernel_mask_t The decode kernel mask to test
 * @return True for the `*_LIST` masks
 */
template <decode_kernel_mask kernel_mask_t>
CUDF_HOST_DEVICE constexpr bool has_lists()
{
  return (kernel_mask_t == decode_kernel_mask::BOOLEAN_LIST) ||
         (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_DICT_LIST) ||
         (kernel_mask_t == decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST) ||
         (kernel_mask_t == decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST) ||
         (kernel_mask_t == decode_kernel_mask::STRING_LIST) ||
         (kernel_mask_t == decode_kernel_mask::STRING_DICT_LIST) ||
         (kernel_mask_t == decode_kernel_mask::STRING_STREAM_SPLIT_LIST);
}

template <decode_kernel_mask kernel_mask_t>
CUDF_HOST_DEVICE constexpr bool is_split_decode()
{
  return (kernel_mask_t == decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT) ||
         (kernel_mask_t == decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST) ||
         (kernel_mask_t == decode_kernel_mask::STRING_STREAM_SPLIT) ||
         (kernel_mask_t == decode_kernel_mask::STRING_STREAM_SPLIT_NESTED) ||
         (kernel_mask_t == decode_kernel_mask::STRING_STREAM_SPLIT_LIST);
}

/**
 * @brief Kernel for computing fixed width column data stored in the pages
 *
 * This function will write the page data and the page data's validity to the
 * output specified in the page's column chunk. If necessary, additional
 * conversion will be performed to translate from the Parquet datatype to
 * desired output datatype.
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 * @param page_mask Boolean vector indicating which pages need to be decoded
 * @param initial_str_offsets Vector to store the initial offsets for large nested string cols
 * @param page_string_offset_indices Device span of offsets, indexed per-page, into the column's
 * string offset buffer
 * @param error_code Error code to set if an error is encountered
 */
template <typename level_t, int decode_block_size_t, decode_kernel_mask kernel_mask_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size_t, 8)
  decode_page_data_generic(PageInfo* pages,
                           device_span<ColumnChunkDesc const> chunks,
                           size_t min_row,
                           size_t num_rows,
                           cudf::device_span<bool const> page_mask,
                           cudf::device_span<size_t> initial_str_offsets,
                           cudf::device_span<size_t const> page_string_offset_indices,
                           kernel_error::pointer error_code)
{
  constexpr bool has_dict_t     = has_dict<kernel_mask_t>();
  constexpr bool has_bools_t    = has_bools<kernel_mask_t>();
  constexpr bool has_nesting_t  = has_nesting<kernel_mask_t>();
  constexpr bool has_lists_t    = has_lists<kernel_mask_t>();
  constexpr bool split_decode_t = is_split_decode<kernel_mask_t>();
  constexpr bool has_strings_t =
    (static_cast<uint32_t>(kernel_mask_t) & STRINGS_MASK_NON_DELTA) != 0;
  constexpr bool is_dict_int32_t = is_dict_int32_output<kernel_mask_t>();

  constexpr int rolling_buf_size    = decode_block_size_t * 2;
  constexpr int rle_run_buffer_size = rle_stream_required_run_buffer_size<decode_block_size_t>();

  __shared__ __align__(16) full_page_decode_state state_g;
  constexpr bool use_dict_buffers = has_dict_t || has_bools_t;
  using state_buf_t               = page_state_buffers_s<rolling_buf_size,  // size of nz_idx buffer
                                           use_dict_buffers ? rolling_buf_size : 1,
                                                         1>;
  __shared__ __align__(16) state_buf_t state_buffers;

  auto const block   = cg::this_thread_block();
  auto* const s      = &state_g;
  auto* const sb     = &state_buffers;
  int const page_idx = cg::this_grid().block_rank();
  int const t        = block.thread_rank();
  PageInfo* pp       = &pages[page_idx];

  if (!(BitAnd(pages[page_idx].kernel_mask, kernel_mask_t))) { return; }

  // Exit early if the page is pruned
  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }

  // must come after the kernel mask check
  [[maybe_unused]] null_count_back_copier _{s, t};

  // Setup local page info
  if (!setup_local_page_info(s,
                             pp,
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{kernel_mask_t},
                             page_processing_stage::DECODE)) {
    return;
  }

  bool const process_nulls = should_process_nulls(s);

  // shared buffer. all shared memory is suballocated out of here
  constexpr int rle_run_buffer_bytes =
    cudf::util::round_up_unsafe(rle_run_buffer_size * sizeof(rle_run), size_t{16});
  constexpr int shared_buf_size = cuda::std::max(
    1, rle_run_buffer_bytes * (static_cast<int>(has_dict_t) + static_cast<int>(has_bools_t)));
  __shared__ __align__(16) uint8_t shared_buf[shared_buf_size];

  // setup all shared memory buffers
  int shared_offset = 0;

  auto dict_runs = reinterpret_cast<rle_run*>(shared_buf + shared_offset);
  if constexpr (has_dict_t) { shared_offset += rle_run_buffer_bytes; }

  auto bool_runs = reinterpret_cast<rle_run*>(shared_buf + shared_offset);

  // get the level data
  auto* const def = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto* const rep = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  rle_stream<uint32_t, decode_block_size_t, rolling_buf_size> dict_stream{dict_runs};
  if constexpr (has_dict_t) {
    dict_stream.init(block,
                     s->stream.dict_bits,
                     s->stream.data_start,
                     s->stream.data_end,
                     sb->dict_idx,
                     s->setup.page.num_input_values);
  }

  // Use dictionary stream memory for bools
  rle_stream<uint32_t, decode_block_size_t, rolling_buf_size> bool_stream{bool_runs};
  bool bools_are_rle_stream = (s->stream.dict_run == 0);
  if constexpr (has_bools_t) {
    if (bools_are_rle_stream) {
      bool_stream.init(block,
                       1,
                       s->stream.data_start,
                       s->stream.data_end,
                       sb->dict_idx,
                       s->setup.page.num_input_values);
    }
  }
  block.sync();

  // We use two counters in the loop below: processed_count and valid_count.
  // - processed_count: number of values out of num_input_values that we have decoded so far.
  //   the definition stream returns the number of total rows it has processed in each call
  //   to decode_next and we accumulate in process_count.
  // - valid_count: number of non-null values we have decoded so far. In each iteration of the
  //   loop below, we look at the number of valid items (which could be all for non-nullable),
  //   and valid_count is that running count.
  int processed_count         = 0;
  int valid_count             = 0;
  size_t string_output_offset = 0;
  int const init_valid_map_offset =
    s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // Skip ahead in the decoding so that we don't repeat work
  skip_ahead_in_decoding<decode_block_size_t,
                         rolling_buf_size,
                         has_lists_t,
                         has_dict_t,
                         has_bools_t,
                         has_nesting_t,
                         level_t>(s,
                                  dict_stream,
                                  bool_stream,
                                  bools_are_rle_stream,
                                  process_nulls,
                                  def,
                                  processed_count,
                                  valid_count);

  // the core loop. decode batches of level stream data using rle_stream objects
  // and pass the results to decode_values
  // For chunked reads we may not process all of the rows on the page; if not stop early
  int const first_row = s->setup.first_row;
  int const last_row  = first_row + s->setup.num_rows;
  while ((s->setup.error == 0) && (processed_count < s->setup.page.num_input_values) &&
         (s->progress.input_row_count <= last_row)) {
    int next_valid_count;
    block.sync();
    processed_count += min(rolling_buf_size, s->setup.page.num_input_values - processed_count);

    // only need to process definition levels if this is a nullable column
    if (process_nulls) {
      if constexpr (has_lists_t) {
        next_valid_count =
          update_validity_and_row_indices_lists<decode_block_size_t, true, level_t>(
            processed_count, s, sb, def, rep, t);
      } else if constexpr (has_nesting_t) {
        next_valid_count = update_validity_and_row_indices_nested<decode_block_size_t, level_t>(
          processed_count, s, sb, def, t);
      } else {
        next_valid_count = update_validity_and_row_indices_flat<decode_block_size_t, level_t>(
          processed_count, s, sb, def, t);
      }
    }
    // if we wanted to split off the skip_rows/num_rows case into a separate kernel, we could skip
    // this function call entirely since all it will ever generate is a mapping of (i -> i) for
    // nz_idx.  decode_fixed_width_values would be the only work that happens.
    else {
      if constexpr (has_lists_t) {
        next_valid_count =
          update_validity_and_row_indices_lists<decode_block_size_t, false, level_t>(
            processed_count, s, sb, nullptr, rep, t);
      } else {
        // direct copy: no nulls, no lists, no need to update validity or row indices
        // This ASSUMES that s->progress.row_index_lower_bound is always -1!
        // Its purpose is to handle rows than span page boundaries, which only happen for lists.
        int const capped_target_value_count = min(processed_count, last_row);
        if (t == 0) { s->progress.input_row_count = capped_target_value_count; }
        next_valid_count = capped_target_value_count;
      }
    }
    block.sync();

    // We want to limit the number of dictionary/bool/string items we decode,
    // that correspond to the rows we have processed in this iteration that are valid.
    // We know the number of valid rows to process with: next_valid_count - valid_count.
    if constexpr (has_dict_t) {
      dict_stream.decode_next(t, next_valid_count - valid_count);
      block.sync();
    } else if constexpr (has_bools_t) {
      if (bools_are_rle_stream) {
        bool_stream.decode_next(t, next_valid_count - valid_count);
      } else {
        auto const target_pos = next_valid_count + s->setup.page.skipped_leaf_values;
        bool_plain_decode(s, sb, target_pos, block);
        if (t == 0) { s->progress.dict_pos = target_pos; }
      }
      block.sync();
    }

    auto decode_values = [&]<copy_mode copy_mode_t>() {
      if constexpr (is_dict_int32_t) {
        decode_dict_indices_as_int32<decode_block_size_t, has_lists_t, copy_mode_t>(
          s, sb, valid_count, next_valid_count, t);
      } else if constexpr (has_strings_t) {
        uint32_t* const str_offsets =
          s->setup.col.column_string_offset_base + page_string_offset_indices[page_idx];
        string_output_offset =
          decode_strings<decode_block_size_t, has_dict_t, has_lists_t, split_decode_t, copy_mode_t>(
            s, sb, valid_count, next_valid_count, t, str_offsets, string_output_offset);
      } else if constexpr (split_decode_t) {
        decode_fixed_width_split_values<decode_block_size_t, has_lists_t, copy_mode_t>(
          s, sb, valid_count, next_valid_count, t);
      } else {
        decode_fixed_width_values<decode_block_size_t, has_lists_t, copy_mode_t>(
          s, sb, valid_count, next_valid_count, t);
      }
    };

    if constexpr (has_lists_t) {
      decode_values.template operator()<copy_mode::INDIRECT>();
    } else {
      if (process_nulls) {
        decode_values.template operator()<copy_mode::INDIRECT>();
      } else {
        decode_values.template operator()<copy_mode::DIRECT>();
      }
    }

    valid_count = next_valid_count;
  }

  // Zero-fill null positions after decoding valid values
  if constexpr (has_strings_t || has_lists_t || is_dict_int32_t) {
    if (process_nulls) {
      uint32_t const dtype_len = [&]() -> uint32_t {
        if constexpr (is_dict_int32_t) { return sizeof(int32_t); }
        if constexpr (has_strings_t) { return sizeof(cudf::size_type); }
        return s->output_cvt.dtype_len;
      }();
      int const num_values = [&]() {
        if constexpr (has_lists_t) {
          auto const& ni = s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1];
          return ni.valid_map_offset - init_valid_map_offset;
        } else {
          return s->setup.num_rows;
        }
      }();
      zero_fill_null_positions_shared<decode_block_size_t>(
        s, dtype_len, init_valid_map_offset, num_values, t);
    }
  }

  if constexpr (has_strings_t) {
    // For large strings, update the initial string buffer offset to be used during large string
    // column construction. Otherwise, convert string sizes to final offsets.

    if constexpr (!has_lists_t) {
      if (!process_nulls) {
        if (t == 0) {
          s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1].value_count =
            s->progress.input_row_count;
        }
        block.sync();
      }
    }

    if (s->setup.col.is_large_string_col) {
      // page.chunk_idx are ordered by input_col_idx and row_group_idx respectively.
      auto const chunks_per_rowgroup = initial_str_offsets.size();
      auto const input_col_idx       = pages[page_idx].chunk_idx % chunks_per_rowgroup;
      compute_initial_large_strings_offset<has_lists_t>(s, initial_str_offsets[input_col_idx]);
    } else {
      convert_small_string_lengths_to_offsets<decode_block_size_t, has_lists_t>(s);
    }
  }

  if (t == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
}

}  // anonymous namespace

template <decode_kernel_mask mask>
using kernel_tag_t = std::integral_constant<decode_kernel_mask, mask>;

template <int value>
using int_tag_t = std::integral_constant<int, value>;

/**
 * @copydoc cudf::io::paruquet::detail::decode_page_data
 */
void decode_page_data(cudf::detail::hostdevice_span<PageInfo> pages,
                      cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                      size_t num_rows,
                      size_t min_row,
                      int level_type_size,
                      decode_kernel_mask kernel_mask,
                      cudf::device_span<bool const> page_mask,
                      cudf::device_span<size_t> initial_str_offsets,
                      cudf::device_span<size_t const> page_string_offset_indices,
                      kernel_error::pointer error_code,
                      cuda::stream_ref stream)
{
  // No template parameters on lambdas until C++20, so use type tags instead
  auto launch_kernel = [&](auto block_size_tag, auto kernel_mask_tag) {
    constexpr int decode_block_size   = decltype(block_size_tag)::value;
    constexpr decode_kernel_mask mask = decltype(kernel_mask_tag)::value;

    dim3 dim_block(decode_block_size, 1);
    dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

    if (level_type_size == 1) {
      decode_page_data_generic<uint8_t, decode_block_size, mask>
        <<<dim_grid, dim_block, 0, stream.get()>>>(pages.device_ptr(),
                                                   chunks,
                                                   min_row,
                                                   num_rows,
                                                   page_mask,
                                                   initial_str_offsets,
                                                   page_string_offset_indices,
                                                   error_code);
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      decode_page_data_generic<uint16_t, decode_block_size, mask>
        <<<dim_grid, dim_block, 0, stream.get()>>>(pages.device_ptr(),
                                                   chunks,
                                                   min_row,
                                                   num_rows,
                                                   page_mask,
                                                   initial_str_offsets,
                                                   page_string_offset_indices,
                                                   error_code);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  };

  switch (kernel_mask) {
    case decode_kernel_mask::FIXED_WIDTH_NO_DICT:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_NO_DICT>{});
      break;
    case decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED:
      launch_kernel(int_tag_t<128>{},
                    kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED>{});
      break;
    case decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST>{});
      break;
    case decode_kernel_mask::FIXED_WIDTH_DICT:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_DICT>{});
      break;
    case decode_kernel_mask::FIXED_WIDTH_DICT_NESTED:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_DICT_NESTED>{});
      break;
    case decode_kernel_mask::FIXED_WIDTH_DICT_LIST:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::FIXED_WIDTH_DICT_LIST>{});
      break;
    case decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT:
      launch_kernel(int_tag_t<128>{},
                    kernel_tag_t<decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT>{});
      break;
    case decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED:
      launch_kernel(int_tag_t<128>{},
                    kernel_tag_t<decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED>{});
      break;
    case decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST:
      launch_kernel(int_tag_t<128>{},
                    kernel_tag_t<decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST>{});
      break;
    case decode_kernel_mask::BOOLEAN:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::BOOLEAN>{});
      break;
    case decode_kernel_mask::BOOLEAN_NESTED:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::BOOLEAN_NESTED>{});
      break;
    case decode_kernel_mask::BOOLEAN_LIST:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::BOOLEAN_LIST>{});
      break;
    case decode_kernel_mask::STRING:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::STRING>{});
      break;
    case decode_kernel_mask::STRING_NESTED:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::STRING_NESTED>{});
      break;
    case decode_kernel_mask::STRING_LIST:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::STRING_LIST>{});
      break;
    case decode_kernel_mask::STRING_DICT:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::STRING_DICT>{});
      break;
    case decode_kernel_mask::STRING_DICT_NESTED:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::STRING_DICT_NESTED>{});
      break;
    case decode_kernel_mask::STRING_DICT_LIST:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::STRING_DICT_LIST>{});
      break;
    case decode_kernel_mask::STRING_STREAM_SPLIT:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::STRING_STREAM_SPLIT>{});
      break;
    case decode_kernel_mask::STRING_STREAM_SPLIT_NESTED:
      launch_kernel(int_tag_t<128>{},
                    kernel_tag_t<decode_kernel_mask::STRING_STREAM_SPLIT_NESTED>{});
      break;
    case decode_kernel_mask::STRING_STREAM_SPLIT_LIST:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::STRING_STREAM_SPLIT_LIST>{});
      break;
    case decode_kernel_mask::DICT_INT32:
      launch_kernel(int_tag_t<128>{}, kernel_tag_t<decode_kernel_mask::DICT_INT32>{});
      break;
    default: CUDF_EXPECTS(false, "Kernel type not handled by this function"); break;
  }
}

}  // namespace cudf::io::parquet::detail
