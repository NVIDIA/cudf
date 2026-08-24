/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "page_decode.cuh"

#include <cudf/strings/detail/gather.cuh>

#include <cuda/atomic>
#include <cuda/std/bit>

namespace cudf::io::parquet::detail {

// stole this from cudf/strings/detail/gather.cuh. modified to run on a single string on one warp.
// copies from src to dst in 16B chunks per thread.
inline __device__ void wideStrcpy(uint8_t* dst, uint8_t const* src, size_t len, uint32_t lane_id)
{
  using cudf::detail::warp_size;
  using cudf::strings::detail::load_uint4;

  constexpr size_t out_datatype_size = sizeof(uint4);
  constexpr size_t in_datatype_size  = sizeof(uint);

  auto const alignment_offset = reinterpret_cast<std::uintptr_t>(dst) % out_datatype_size;
  uint4* out_chars_aligned    = reinterpret_cast<uint4*>(dst - alignment_offset);
  auto const in_start         = src;

  // Both `out_start_aligned` and `out_end_aligned` are indices into `dst`.
  // `out_start_aligned` is the first 16B aligned memory location after `dst + 4`.
  // `out_end_aligned` is the last 16B aligned memory location before `len - 4`. Characters
  // between `[out_start_aligned, out_end_aligned)` will be copied using uint4.
  // `dst + 4` and `len - 4` are used instead of `dst` and `len` to avoid
  // `load_uint4` reading beyond string boundaries.
  // use signed int since out_end_aligned can be negative.
  int64_t const out_start_aligned = (in_datatype_size + alignment_offset + out_datatype_size - 1) /
                                      out_datatype_size * out_datatype_size -
                                    alignment_offset;
  int64_t const out_end_aligned =
    (len - in_datatype_size + alignment_offset) / out_datatype_size * out_datatype_size -
    alignment_offset;

  for (int64_t ichar = out_start_aligned + lane_id * out_datatype_size; ichar < out_end_aligned;
       ichar += warp_size * out_datatype_size) {
    *(out_chars_aligned + (ichar + alignment_offset) / out_datatype_size) =
      load_uint4((char const*)in_start + ichar);
  }

  // Tail logic: copy characters of the current string outside
  // `[out_start_aligned, out_end_aligned)`.
  if (out_end_aligned <= out_start_aligned) {
    // In this case, `[out_start_aligned, out_end_aligned)` is an empty set, and we copy the
    // entire string.
    for (int64_t ichar = lane_id; ichar < len; ichar += warp_size) {
      dst[ichar] = in_start[ichar];
    }
  } else {
    // Copy characters in range `[0, out_start_aligned)`.
    if (lane_id < out_start_aligned) { dst[lane_id] = in_start[lane_id]; }
    // Copy characters in range `[out_end_aligned, len)`.
    int64_t ichar = out_end_aligned + lane_id;
    if (ichar < len) { dst[ichar] = in_start[ichar]; }
  }
}

/**
 * @brief Perform exclusive scan on an array of any length using a single block of threads.
 */
template <int block_size>
__device__ inline void block_excl_sum(size_type* arr, size_type length, size_type initial_value)
{
  using block_scan = cub::BlockScan<size_type, block_size>;
  __shared__ typename block_scan::TempStorage scan_storage;

  // Do a series of block sums, storing results in arr as we go
  auto const block = cooperative_groups::this_thread_block();
  for (int pos = 0; pos < length; pos += block.size()) {
    int const tidx = pos + block.thread_rank();
    size_type tval = tidx < length ? arr[tidx] : 0;

    size_type block_sum;
    size_type new_tval;
    block_scan(scan_storage).ExclusiveSum(tval, new_tval, block_sum);
    block.sync();

    if (tidx < length) { arr[tidx] = new_tval + initial_value; }
    initial_value += block_sum;
  }
}

/**
 * @brief Converts string sizes to offsets if this is not a large string column.
 */
template <int block_size, bool has_lists>
__device__ void convert_small_string_lengths_to_offsets(auto const* const state)
{
  // If this is a large string column. In the
  // latter case, offsets will be computed during string column creation.
  auto& ni        = state->nesting.nesting_info[state->setup.col.max_nesting_depth - 1];
  int value_count = ni.value_count;

  // if no repetition we haven't calculated start/end bounds and instead just skipped
  // values until we reach first_row. account for that here.
  if constexpr (not has_lists) { value_count -= state->setup.first_row; }

  // Convert the array of lengths into offsets
  if (value_count > 0) {
    auto const offptr        = reinterpret_cast<size_type*>(ni.data_out);
    auto const initial_value = state->setup.page.str_offset;
    block_excl_sum<block_size>(offptr, value_count, initial_value);
  }
}

/**
 * @brief Atomically update the initial string offset to be used during large string column
 * construction
 */
template <bool has_lists>
inline __device__ void compute_initial_large_strings_offset(auto const* const state,
                                                            size_t& initial_str_offset)
{
  // Values decoded by this page.
  int value_count = state->nesting.nesting_info[state->setup.col.max_nesting_depth - 1].value_count;

  // if no repetition we haven't calculated start/end bounds and instead just skipped
  // values until we reach first_row. account for that here.
  if constexpr (not has_lists) { value_count -= state->setup.first_row; }

  // Atomically update the initial string offset if this is a large string column. This initial
  // offset will be used to compute (64-bit) offsets during large string column construction.
  if (value_count > 0 and threadIdx.x == 0) {
    auto const initial_value = state->setup.page.str_offset;
    cuda::atomic_ref<size_t, cuda::std::thread_scope_device> initial_str_offsets_ref{
      initial_str_offset};
    initial_str_offsets_ref.fetch_min(initial_value, cuda::std::memory_order_relaxed);
  }
}

template <int value>
CUDF_HOST_DEVICE constexpr int log2_int()
{
  static_assert((value >= 1) && ((value & (value - 1)) == 0), "Only works for powers of 2!");
  return 31 - cuda::std::countl_zero(static_cast<uint32_t>(value));
}

template <int block_size>
__device__ inline int calc_threads_per_string_log2(int avg_string_length)  // returns log2(M)
{
  // From testing, performance is best when copying an average of B = 4 bytes at once.
  // So #-threads-per-string M = avg_string_length / 4
  // Help the compiler make the code fast by keeping everything a power of 2
  // For avg length < 4/8/16/..., length power-of-2 = 2/3/4/.../. Divide by 4: 0/1/...

  // avg - 1: Don't want extra thread at powers of 2 (e.g. 32 (0b100000 -> 0b11111 -> 5)
  int const avg_log2     = 32 - __clz(avg_string_length - 1);
  int const threads_log2 = avg_log2 - 2;  // Target 4 bytes / thread at once (log2(4) = 2)

  // This is the target (log2) for M, but we need to clamp its range
  // First clamp #-strings-at-once (N) btw 1 (all threads (T)) & 32 (cache miss if larger)
  // So, clamp #-threads-per-string M = T / N between: T (all) & T/32 (cache miss)
  // So, clamp log2(#-threads-per-string) between log2(T) & log2(T) - 5 (min 1)
  static constexpr int block_size_log2  = log2_int<block_size>();  // 7 for block_size = 128
  static constexpr int min_threads_log2 = cuda::std::max(block_size_log2 - 5, 1);

  // Clamp log2(M) (between 2 and 7 for block_size = 128)
  return cuda::std::max(min_threads_log2, cuda::std::min(block_size_log2, threads_log2));
}

/**
 * @brief Function for copying strings from the parquet page into column memory.
 *
 * All of the threads in the block will help with memcpy's, but only on a max of
 * 32 strings at once due to memory caching issues.
 * The # strings copied at once (and how many threads per string) is determined
 * from the average string length, with a target memcpy of 4-bytes per thread.
 *
 * @param s Page state
 * @param sb Page state buffers
 * @param start The value index to start copying strings for
 * @param end One past the end value index to stop copying strings for
 * @param t The current thread's index
 * @param str_offsets String offset values
 * @param string_output_offset Starting offset into the output column data for writing
 */
template <int block_size,
          bool has_dict_t,
          bool has_lists_t,
          bool split_decode_t,
          copy_mode copy_mode_t,
          typename state_buf>
__device__ size_t decode_strings(auto* s,
                                 state_buf* const sb,
                                 int start,
                                 int end,
                                 int t,
                                 uint32_t* str_offsets,
                                 size_t string_output_offset)
{
  // nesting level that is storing actual leaf values
  int const leaf_level_index    = s->setup.col.max_nesting_depth - 1;
  int const skipped_leaf_values = s->setup.page.skipped_leaf_values;

  auto const& ni = s->nesting.nesting_info[leaf_level_index];

  // decode values
  int pos = start;
  while (pos < end) {
    int const batch_size = min(block_size, end - pos);

    int const target_pos = pos + batch_size;
    int const thread_pos = pos + t;

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
      if constexpr (has_lists_t) { return thread_pos + skipped_leaf_values; }
      return thread_pos;
    }();

    // lookup input string pointer & length. store length.
    bool const in_range                             = (thread_pos < target_pos);
    auto const [thread_input_string, string_length] = [&]() {
      // target_pos will always be properly bounded by num_rows, but dst_pos may be negative (values
      // before first_row) in the flat hierarchy case.
      if (!in_range) { return string_index_pair{nullptr, 0}; }
      if constexpr (has_dict_t) {
        return gpuGetStringData(s, sb, src_pos);
      } else {
        int input_thread_string_offset;
        int string_length;
        if (s->setup.col.physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
          input_thread_string_offset = src_pos * s->output_cvt.dtype_len_in;
          string_length              = s->output_cvt.dtype_len_in;
        } else {
          input_thread_string_offset = str_offsets[src_pos];
          int const next_offset      = str_offsets[src_pos + 1];
          // The memory is laid out as: 4-byte length, string, 4-byte length, string, ...
          // String length = subtract the offsets and the stored length of the next string
          // Except at the end of the dictionary, where the last string offset is repeated.
          string_length = (next_offset == input_thread_string_offset)
                            ? 0
                            : next_offset - input_thread_string_offset - sizeof(int32_t);
        }
        if (input_thread_string_offset >= static_cast<uint32_t>(s->stream.dict_size)) {
          return string_index_pair{nullptr, 0};
        }
        auto const thread_input_string =
          reinterpret_cast<char const*>(s->stream.data_start + input_thread_string_offset);
        return string_index_pair{thread_input_string, string_length};
      }
    }();
    if (in_range) {
      int32_t* str_len_ptr = reinterpret_cast<int32_t*>(ni.data_out) + dst_pos;
      *str_len_ptr         = string_length;
    }

    // compute output string offsets
    size_t thread_string_offset, block_total_string_length;
    {
      using scanner = cub::BlockScan<size_t, block_size>;
      __shared__ typename scanner::TempStorage scan_storage;
      scanner(scan_storage)
        .ExclusiveSum(string_length, thread_string_offset, block_total_string_length);

      // Make sure all threads have finished using scan_storage before next loop overwrites.
      __syncthreads();
    }

    // adjust for prior offset, get output string pointer
    thread_string_offset += string_output_offset;
    string_output_offset += block_total_string_length;
    auto const thread_output_string = ni.string_out + thread_string_offset;

    if constexpr (split_decode_t) {
      if (in_range) {
        auto const split_string_length = s->output_cvt.dtype_len_in;
        auto const stream_length       = s->setup.page.str_bytes / split_string_length;

        for (int ii = 0; ii < split_string_length; ii++) {
          thread_output_string[ii] = s->stream.data_start[src_pos + ii * stream_length];
        }
      }
    } else {
      // With a dictionary, performance is MUCH better for cache hits (reuse dict entry)
      // On a cache hit, we load a 128-byte L1/L2 cache line
      // If each thread (e.g. 128) copies a string (128/line): 128*128 = 16kb PER BLOCK
      // This likely overflows the L2 cache and results in a lot of cache misses
      // Plus, perf is O(longest_length), so we're bottlenecked for non-uniform data

      // Instead, the (T = block_size) threads cooperatively copy the T strings
      // N strings are copied at a time, with M = T / N threads per string
      __shared__ uint8_t const* inputs[block_size];
      __shared__ uint8_t* outputs[block_size];
      __shared__ int lengths[block_size];

      // Save string pointers & lengths so threads can share the work on them
      outputs[t] = thread_output_string;
      inputs[t]  = reinterpret_cast<uint8_t const*>(thread_input_string);
      lengths[t] = string_length;

      // Choose M, N to be powers of 2 to divide T evenly and allow bit shifts.
      // For T threads: clamp N btw T/32 (1 string per warp) & 32 (cache miss if larger)
      // Per string, each thread copies B = Length / M bytes in one contiguous memcpy

      // Performance: O(N*Avg_Length/M), with floor of at least N memcpy's (of O(Avg_length/M))
      // Note if M is large (vs. Avg_length) then many threads may do nothing: Slow
      // Note if M is small then N is large, increasing the time floor of N memcpy's
      // Determine M and N for each batch of strings based on their average length
      int const avg_string_length = block_total_string_length / batch_size;
      int const threads_per_string_log2 =
        calc_threads_per_string_log2<block_size>(avg_string_length);
      int const threads_per_string = 1 << threads_per_string_log2;  // M

      // For block_size = T = 128:
      // For an avg string length of 16 bytes or less (because N clamped): M = 4, N = 32
      // For an avg length of 65+ bytes (rounded): M = 32, N = 4 (1 string / warp at once)
      int const strings_at_once = block_size >> threads_per_string_log2;  // N
      int const string_lane     = t & (threads_per_string - 1);
      int const start_str_idx   = t >> threads_per_string_log2;

      // Sync writing the string info to shared memory above, prior to using it below.
      __syncthreads();

      // loop over all strings in this batch
      // threads work on consecutive strings so that all bytes are close in memory
      for (int str_idx = start_str_idx; str_idx < batch_size; str_idx += strings_at_once) {
        auto const input_string = inputs[str_idx];
        if (input_string == nullptr) { continue; }

        auto output_string = outputs[str_idx];
        int const length   = lengths[str_idx];

        // Max 8 chars at once per thread, else perf degrades dramatically
        // Loop, copying 8 chars at a time, until <= 8 chars per thread left
        static constexpr int max_chars_at_once = 8;
        int chars_remaining_per_thread =
          (length + threads_per_string - 1) >> threads_per_string_log2;
        int group_offset = 0;
        if (chars_remaining_per_thread > max_chars_at_once) {
          int const max_chars_copied_string = max_chars_at_once * threads_per_string;
          int start_index                   = string_lane * max_chars_at_once;
          do {
            memcpy(&(output_string[start_index]), &(input_string[start_index]), max_chars_at_once);

            chars_remaining_per_thread -= max_chars_at_once;
            start_index += max_chars_copied_string;
            group_offset += max_chars_copied_string;
          } while (chars_remaining_per_thread > max_chars_at_once);
        }

        // Final copy of remaining chars
        int const start_index      = group_offset + string_lane * chars_remaining_per_thread;
        int const substring_length = min(chars_remaining_per_thread, length - start_index);
        if (substring_length > 0) {
          memcpy(&(output_string[start_index]), &(input_string[start_index]), substring_length);
        }
      }

      // Make sure all threads have finished using shared memory before next loop overwrites
      __syncthreads();
    }

    pos += batch_size;
  }

  return string_output_offset;
}

}  // namespace cudf::io::parquet::detail
