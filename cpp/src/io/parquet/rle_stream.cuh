/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda/std/algorithm>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/span>

namespace cudf::io::parquet::detail {

template <int num_threads>
__device__ constexpr int rle_stream_required_run_buffer_size()
{
  constexpr int num_rle_stream_decode_warps = (num_threads / cudf::detail::warp_size) - 1;
  return (num_rle_stream_decode_warps * 2);
}

/**
 * @brief Read a 32-bit varint integer from an arbitrary byte source.
 *
 * The byte source is provided via an accessor `read(int off) -> uint8_t`, so
 * callers can read directly from a global pointer, from a shared-memory
 * staging buffer, or dispatch between the two per-byte (chunked staging).
 *
 * @param[in,out] offset The current byte offset into the stream; advanced past the varint
 * @param[in] end_offset One past the last valid byte offset
 * @param[in] read Byte accessor: `read(int off) -> uint8_t`
 *
 * @return The 32-bit value read
 */
template <typename Accessor>
__device__ inline uint32_t get_vlq32(int& offset, int const end_offset, Accessor read)
{
  uint32_t v = read(offset++);
  if (v >= 0x80 && offset < end_offset) {
    v = (v & 0x7f) | (read(offset++) << 7);
    if (v >= (0x80 << 7) && offset < end_offset) {
      v = (v & ((0x7f << 7) | 0x7f)) | (read(offset++) << 14);
      if (v >= (0x80 << 14) && offset < end_offset) {
        v = (v & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | (read(offset++) << 21);
        if (v >= (0x80 << 21) && offset < end_offset) {
          v = (v & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) | (read(offset++) << 28);
        }
      }
    }
  }
  return v;
}

/**
 * @brief Read a 32-bit varint integer via a raw pointer cursor.
 *
 * Thin adapter over the offset-based `get_vlq32` for callers that already
 * track a moving pointer. The lambda is trivial (`base[off]`) and inlines to
 * the same load sequence as a direct `*cur++` walk.
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The 32-bit value read
 */
inline __device__ uint32_t get_vlq32(uint8_t const*& cur, uint8_t const* end)
{
  uint8_t const* const base = cur;
  int offset                = 0;
  int const end_offset      = static_cast<int>(end - cur);
  uint32_t const v          = get_vlq32(offset, end_offset, [base](int off) { return base[off]; });
  cur += offset;
  return v;
}

/**
 * @brief RLE run decode function per warp.
 *
 * @param output output data buffer
 * @param level_run RLE run header
 * @param run_start beginning of data for RLE run
 * @param end pointer to the end of data for RLE run
 * @param run_output_pos absolute output position for this run
 * @param run_offset offset after run_output_pos this call to decode starts outputting at
 * @param size length that will be decoded in this decode call, truncated to fit output buffer
 * @param level_bits bits needed to encode max values in the run (definition, dictionary)
 * @param lane warp lane that is executing this decode call
 */
template <typename level_t, int max_output_values>
__device__ inline void decode(level_t* const output,
                              int const level_run,
                              uint8_t const* const run_start,
                              uint8_t const* const end,
                              int const run_output_pos,
                              int const run_offset,
                              int const size,
                              int level_bits,
                              int lane)
{
  // local output_pos for this `decode` call.
  int decode_output_pos = 0;
  int remain            = size;

  // for bitpacked/literal runs, total size is always a multiple of 8. so we need to take care if
  // we are not starting/ending exactly on a run boundary
  uint8_t const* cur;
  if (is_literal_run(level_run)) {
    int const effective_offset = cudf::util::round_down_safe(run_offset, 8);
    int const lead_values      = (run_offset - effective_offset);
    decode_output_pos -= lead_values;
    remain += lead_values;
    cur = run_start + ((effective_offset >> 3) * level_bits);
  }

  // if this is a repeated run, compute the repeated value
  int level_val;
  if (is_repeated_run(level_run)) {
    level_val = run_start[0];
    if constexpr (sizeof(level_t) > 1) {
      if (level_bits > 8) {
        level_val |= run_start[1] << 8;
        if constexpr (sizeof(level_t) > 2) {
          if (level_bits > 16) {
            level_val |= run_start[2] << 16;
            if (level_bits > 24) { level_val |= run_start[3] << 24; }
          }
        }
      }
    }
  }

  // process
  while (remain > 0) {
    int const batch_len = min(32, remain);

    // if this is a literal run. each thread computes its own level_val
    if (is_literal_run(level_run)) {
      int const batch_len8 = (batch_len + 7) >> 3;
      if (lane < batch_len) {
        int bitpos                = lane * level_bits;
        uint8_t const* cur_thread = cur + (bitpos >> 3);
        bitpos &= 7;
        level_val = 0;
        if (cur_thread < end) { level_val = cur_thread[0]; }
        cur_thread++;
        if (level_bits > 8 - bitpos && cur_thread < end) {
          level_val |= cur_thread[0] << 8;
          cur_thread++;
          if (level_bits > 16 - bitpos && cur_thread < end) {
            level_val |= cur_thread[0] << 16;
            cur_thread++;
            if (level_bits > 24 - bitpos && cur_thread < end) { level_val |= cur_thread[0] << 24; }
          }
        }
        level_val = (level_val >> bitpos) & ((1 << level_bits) - 1);
      }

      cur += batch_len8 * level_bits;
    }

    // store level_val
    if (lane < batch_len && (lane + decode_output_pos) >= 0) {
      auto const idx = lane + run_output_pos + run_offset + decode_output_pos;
      output[rolling_index<max_output_values>(idx)] = level_val;
    }
    remain -= batch_len;
    decode_output_pos += batch_len;
  }
}

// a single rle run. may be broken up into multiple rle_batches
struct rle_run {
  int size;        // total size of the run
  int output_pos;  // absolute position of this run w.r.t output
  uint8_t const* start;
  int level_run;  // level_run header value
  int remaining;  // number of output items remaining to be decoded
};

// Controls the number of run headers parsed per chunk in the chunked-expand path.
// SMEM cost is (2 * max_runs_per_chunk + 1) * 4 bytes. Increasing max_runs_per_chunk reduces the
// number of outer-loop iterations in decode_next_chunked (each with a serial
// header-parse phase and a __syncthreads() at the end), but competes with occupancy in
// preprocess_levels_kernel.
//
// 1024 was chosen empirically for sm_80+. Sweeps of {256, 512, 1024, 2048,
// 4096} on A100, H100, and B200 all showed 1024 either as the numerical
// optimum or within noise of it. The delta over 512 was small (a few percent)
// in every case. sm_70/sm_75 stay at 512, because 1024 does not fit the
// preprocess_levels_kernel SMEM budget on those older arches.
#if __CUDA_ARCH__ >= 800
static constexpr int max_runs_per_chunk = 1024;
#else
static constexpr int max_runs_per_chunk = 512;
#endif

// a stream of rle_runs
template <typename level_t,
          int decode_threads,
          int max_output_values,
          bool use_chunked_expand   = false,
          bool use_smem_staging     = false,
          int smem_stage_size_bytes = 8 * 1024,
          int chunk_size_bytes      = 4096>
struct rle_stream {
  static_assert(!(use_smem_staging && !use_chunked_expand),
                "use_smem_staging requires use_chunked_expand");
  static constexpr int num_rle_stream_decode_threads = decode_threads;
  // the -1 here is for the look-ahead warp that fills in the list of runs to be decoded
  // in an overlapped manner. so if we had 16 total warps:
  // - warp 0 would be filling in batches of runs to be processed
  // - warps 1-15 would be decoding the previous batch of runs generated
  static constexpr int num_rle_stream_decode_warps =
    (num_rle_stream_decode_threads / cudf::detail::warp_size) - 1;

  static constexpr int run_buffer_size = rle_stream_required_run_buffer_size<decode_threads>();

  int level_bits;
  uint8_t const* s_start;
  uint8_t const* cur;
  uint8_t const* end;

  int total_values;
  int cur_values;

  level_t* output;

  rle_run* runs;

  int output_pos;

  int fill_index;
  int decode_index;

  // Optional shared-memory staging of the encoded byte stream. When init() is
  // given a scratch buffer large enough to hold [start, end), the stream is
  // copied into it once (block-cooperatively) and cur/end are rebased into
  // shared memory. This turns the serial run-header parse that dominates
  // fill_run_batch() from a chain of dependent L2 loads into shared-memory
  // loads. It stages *raw encoded bytes*, so it is level_t- and
  // level_bits-agnostic: definition/repetition levels, dictionary indices, and
  // boolean streams all benefit with identical code. Streams that do not fit
  // the budget transparently fall back to parsing from global.
  //
  // The chunked-expand path instead uses double-buffered SMEM staging via
  // cuda::pipeline in decode_next_chunked, addressed by chunk_size (4096 B).
  static constexpr int chunk_size      = 4096;
  static constexpr int smem_stage_size = smem_stage_size_bytes;

  uint8_t* smem_stage;
  int smem_stage_capacity;
  cuda::pipeline<cuda::thread_scope_block>* smem_pipe;

  // Chunked-expand cross-call partial-run state.
  // When a run straddles a decode_next_chunked boundary we record its meta,
  // total count, and how many values were already emitted so the next call
  // can resume from the correct payload offset.  partial_run_meta == -1
  // means no pending partial run.
  int partial_run_meta;   // meta word for the split run (-1 = none)
  int partial_run_total;  // full value count of that run
  int partial_run_done;   // values already emitted from it

  __device__ rle_stream(rle_run* _runs) : runs(_runs) {}

  __device__ inline bool is_last_decode_warp(int warp_id)
  {
    return warp_id == num_rle_stream_decode_warps;
  }

  template <typename Group>
  __device__ void init(Group const& group,
                       int _level_bits,
                       uint8_t const* _start,
                       uint8_t const* _end,
                       level_t* _output,
                       int _total_values,
                       uint8_t* _smem_stage                                   = nullptr,
                       cuda::barrier<cuda::thread_scope_block>* _copy_barrier = nullptr,
                       int stage_capacity                                     = smem_stage_size)
  {
    level_bits = _level_bits;
    // s_start is set below after any smem-staging rebase, so downstream code
    // that stores offsets relative to s_start (chunked-expand meta) works
    // uniformly whether cur points into global or shared memory.
    cur = _start;
    end = _end;

    output = _output;

    output_pos = 0;

    total_values = _total_values;
    cur_values   = 0;
    fill_index   = 0;
    decode_index = -1;  // signals the first iteration. Nothing to decode.

    cudf_assert(stage_capacity >= 0);
    static_cast<void>(group);
    static_cast<void>(_copy_barrier);
    smem_stage          = _smem_stage;
    smem_stage_capacity = stage_capacity;
    smem_pipe           = nullptr;

    if (_smem_stage != nullptr) {
      auto const len = static_cast<int>(cuda::std::distance(_start, _end));
      if constexpr (!use_chunked_expand) {
        cudf_assert(_copy_barrier != nullptr);
        auto* const smem_stage_ptr =
          static_cast<uint8_t const*>(cuda::std::assume_aligned<16>(_smem_stage));
        if (len > 0 && len <= stage_capacity) {
          cuda::memcpy_async(group, _smem_stage, _start, static_cast<size_t>(len), *_copy_barrier);
          cur = smem_stage_ptr;
          end = smem_stage_ptr + len;
        }
      }
    }
    // Chunked-expand keeps s_start global; non-chunked one-shot staging may rebase it to smem.
    s_start = cur;

    partial_run_meta  = -1;
    partial_run_total = 0;
    partial_run_done  = 0;
  }

  __device__ inline void set_pipeline(cuda::pipeline<cuda::thread_scope_block>& pipe)
  {
    if constexpr (use_smem_staging) { smem_pipe = &pipe; }
  }

  __device__ inline int get_rle_run_info(rle_run& run)
  {
    run.start     = cur;
    run.level_run = get_vlq32(run.start, end);

    // run_bytes includes the header size
    int run_bytes = run.start - cur;
    if (is_literal_run(run.level_run)) {
      // from the parquet spec: literal runs always come in multiples of 8 values.
      run.size = (run.level_run >> 1) * 8;
      run_bytes += util::div_rounding_up_unsafe(run.size * level_bits, 8);
    } else {
      // repeated value run
      run.size = (run.level_run >> 1);
      run_bytes += util::div_rounding_up_unsafe(level_bits, 8);
    }

    return run_bytes;
  }

  __device__ inline void fill_run_batch()
  {
    // decode_index == -1 means we are on the very first decode iteration for this stream.
    // In this first iteration we are filling up to half of the runs array to decode in the next
    // iteration. On subsequent iterations, decode_index >= 0 and we are going to fill as many run
    // slots available as we can, to fill up to the slot before decode_index. We are also always
    // bound by cur < end, making sure we stop decoding once we've reached the end of the stream.
    while (((decode_index == -1 && fill_index < num_rle_stream_decode_warps) ||
            fill_index < decode_index + run_buffer_size) &&
           cur < end) {
      // Encoding::RLE
      // Pass by reference to fill the runs shared memory with the run data
      auto& run           = runs[rolling_index<run_buffer_size>(fill_index)];
      int const run_bytes = get_rle_run_info(run);

      run.remaining  = run.size;
      run.output_pos = output_pos;

      cur += run_bytes;
      output_pos += run.size;
      fill_index++;
    }
  }

  __device__ inline int decode_next_ring(int t, int count)
  {
    int const output_count = min(count, total_values - cur_values);

    // otherwise, full decode.
    int const warp_id        = t / cudf::detail::warp_size;
    int const warp_decode_id = warp_id - 1;
    int const warp_lane      = t % cudf::detail::warp_size;

    __shared__ int values_processed_shared;
    __shared__ int decode_index_shared;
    __shared__ int fill_index_shared;
    if (t == 0) {
      values_processed_shared = 0;
      decode_index_shared     = decode_index;
      fill_index_shared       = fill_index;
    }

    __syncthreads();

    fill_index = fill_index_shared;

    do {
      // protect against threads advancing past the end of this loop
      // and updating shared variables.
      __syncthreads();

      // warp 0 reads ahead and fills `runs` array to be decoded by remaining warps.
      if (warp_id == 0) {
        // fill the next set of runs. fill_runs will generally be the bottleneck for any
        // kernel that uses an rle_stream.
        if (warp_lane == 0) {
          fill_run_batch();
          if (decode_index == -1) {
            // first time, set it to the beginning of the buffer (rolled)
            decode_index        = 0;
            decode_index_shared = decode_index;
          }
          fill_index_shared = fill_index;
        }
      }
      // remaining warps decode the runs, starting on the second iteration of this. the pipeline of
      // runs is also persistent across calls to decode_next, so on the second call to decode_next,
      // this branch will start doing work immediately.
      // do/while loop (decode_index == -1 means "first iteration", so we should skip decoding)
      else if (decode_index >= 0 && decode_index + warp_decode_id < fill_index) {
        int const run_index = decode_index + warp_decode_id;
        auto& run           = runs[rolling_index<run_buffer_size>(run_index)];
        // this is the total amount (absolute) we will write in this invocation
        // of `decode_next`.
        int const max_count = cur_values + output_count;
        // run.output_pos is absolute position, we start decoding
        // if it's supposed to fit in this call to `decode_next`.
        if (max_count > run.output_pos) {
          int remaining        = run.remaining;
          int const run_offset = run.size - remaining;
          // last_run_pos is the absolute position of the run, including
          // what was decoded last time.
          int const last_run_pos = run.output_pos + run_offset;

          // the amount we should process is the smallest of current remaining, or
          // space available in the output buffer (for that last run at the end of
          // a call to decode_next).
          int const batch_len = min(remaining, max_count - last_run_pos);
          decode<level_t, max_output_values>(output,
                                             run.level_run,
                                             run.start,
                                             end,
                                             run.output_pos,
                                             run_offset,
                                             batch_len,
                                             level_bits,
                                             warp_lane);

          __syncwarp();
          if (warp_lane == 0) {
            // after writing this batch, are we at the end of the output buffer?
            auto const at_end = ((last_run_pos + batch_len - cur_values) == output_count);

            // update remaining for my warp
            remaining -= batch_len;
            // this is the last batch we will process this iteration if:
            // - either this run still has remaining values
            // - or it is consumed fully and its last index corresponds to output_count
            if (remaining > 0 || at_end) { values_processed_shared = output_count; }
            if (remaining == 0 && (at_end || is_last_decode_warp(warp_id))) {
              decode_index_shared = run_index + 1;
            }
            run.remaining = remaining;
          }
        }
      }
      __syncthreads();
      decode_index = decode_index_shared;
      fill_index   = fill_index_shared;
    } while (values_processed_shared < output_count);

    cur_values += values_processed_shared;

    // valid for every thread
    return values_processed_shared;
  }

  /* Alternate decode path used when `use_chunked_expand` is true.
   *
   * Instead of the ring-buffer producer/consumer model in decode_next_ring
   * (one warp parses run headers, other warps expand one run each), thread 0
   * parses up to `max_runs_per_chunk` headers up-front into shared-memory tables
   * (chunk_out_off / chunk_meta), and then *all* warps cooperatively expand a
   * slice of the concatenated output range using binary search. This keeps
   * every warp busy even when runs are highly non-uniform in size, at the
   * cost of an extra intra-block sync per chunk.
   *
   * A single RLE run may exceed the requested `count`. In that case we emit
   * as much as fits, stash `partial_run_{meta,total,done}` in class state,
   * and resume it as slot 0 of the next invocation without re-parsing its
   * header (cur has already advanced past the payload).
   */
  __device__ inline int decode_next_chunked(int t, int count)
  {
    int const output_count = min(count, total_values - cur_values);

    // Per-chunk shared-memory scratch. `chunk_out_off[i]` is the exclusive
    // prefix-sum of run lengths within the current chunk, so run `i`
    // occupies output positions [chunk_out_off[i], chunk_out_off[i+1]).
    // `chunk_meta[i]` encodes both the payload offset (into s_start) and,
    // in the top bit, whether the run is literal (1) or RLE (0).
    __shared__ int chunk_out_off[max_runs_per_chunk + 1];
    __shared__ int chunk_meta[max_runs_per_chunk];
    cuda::std::span<int> const chunk_out_off_v{chunk_out_off, max_runs_per_chunk + 1};
    cuda::std::span<int> const chunk_meta_v{chunk_meta, max_runs_per_chunk};
    // Payload offset within run slot 0: non-zero only when continuing a run
    // that was split across two decode_next_chunked calls.
    __shared__ int s_run0_payload_offset;
    __shared__ int s_chunk_runs;   // number of runs parsed in this chunk (n)
    __shared__ int s_chunk_total;  // sum of run lengths in this chunk (co)
    __shared__ int s_base_out;     // absolute output pos where this chunk starts
    __shared__ int s_load_cur_offset;

    int const lane          = t & 31;
    int const warp          = t >> 5;
    int constexpr num_warps = num_rle_stream_decode_threads / cudf::detail::warp_size;
    int const value_width   = (level_bits + 7) >> 3;
    // Bit mask used to extract a single level from a bit-packed literal-run
    // payload word. Invariant across the whole call; hoisted out of the
    // phase-2 expand loop to keep it out of the hot register set.
    uint32_t const level_mask = (level_bits == 32) ? 0xffffffffu : ((1u << level_bits) - 1);
    int out_pos_total         = cur_values;
    int const out_end         = cur_values + output_count;
    int const stream_len      = static_cast<int>(cuda::std::distance(s_start, end));
    bool const use_smem       = use_smem_staging && smem_stage != nullptr && smem_pipe != nullptr &&
                          smem_stage_capacity >= smem_stage_size && stream_len > 0;
    auto const block  = cooperative_groups::this_thread_block();
    int loaded_chunk0 = -1;
    int loaded_chunk1 = -1;

    auto is_loaded   = [&](int chunk) { return chunk == loaded_chunk0 || chunk == loaded_chunk1; };
    auto mark_loaded = [&](int chunk) {
      if ((chunk & 1) == 0) {
        loaded_chunk0 = chunk;
      } else {
        loaded_chunk1 = chunk;
      }
    };
    auto stage_ptr = [&](int chunk) {
      return static_cast<uint8_t*>(cuda::std::assume_aligned<16>(smem_stage)) +
             (chunk & 1) * chunk_size;
    };
    auto load_chunk = [&](int chunk) {
      if constexpr (!use_smem_staging) { return false; }
      if (!use_smem || chunk < 0 || chunk * chunk_size >= stream_len || is_loaded(chunk)) {
        return false;
      }
      int const offset = chunk * chunk_size;
      int const bytes  = min(chunk_size, stream_len - offset);
      smem_pipe->producer_acquire();
      cuda::memcpy_async(
        block, stage_ptr(chunk), s_start + offset, static_cast<size_t>(bytes), *smem_pipe);
      smem_pipe->producer_commit();
      return true;
    };
    auto staged_byte = [&](int offset) {
      if constexpr (!use_smem_staging) { return s_start[offset]; }
      int const chunk = offset / chunk_size;
      return (use_smem && is_loaded(chunk)) ? stage_ptr(chunk)[offset - chunk * chunk_size]
                                            : s_start[offset];
    };

    // Outer loop: process the requested output range in chunks of up to
    // `max_runs_per_chunk` runs at a time until we have emitted `output_count`
    // values or run out of encoded input.
    while (out_pos_total < out_end) {
      if (t == 0) { s_load_cur_offset = static_cast<int>(cuda::std::distance(s_start, cur)); }
      __syncthreads();

      int const cur_offset       = s_load_cur_offset;
      int const current_chunk    = cur_offset / chunk_size;
      bool const current_pending = load_chunk(current_chunk);
      if (current_pending) {
        smem_pipe->consumer_wait();
        mark_loaded(current_chunk);
      }
      bool const next_pending = load_chunk(current_chunk + 1);

      // ----- Phase 1: single-thread run-header parse ------------------
      // Thread 0 walks the encoded stream, decoding VLQ run headers and
      // filling chunk_out_off / chunk_meta. The other threads wait at the
      // __syncthreads() below. This is cheap because it is bounded by
      // max_runs_per_chunk headers and header parsing is inherently serial.
      if (t == 0) {
        int co                = 0;
        int n                 = 0;
        int out_base          = out_pos_total;
        int cur_offset        = static_cast<int>(cuda::std::distance(s_start, cur));
        chunk_out_off_v[0]    = 0;
        s_run0_payload_offset = 0;

        // Slot 0 special case: resume a run that was split by the previous
        // call. `cur` already points past this run's payload (fully
        // consumed last call), so we do NOT re-parse its header - we just
        // reuse the saved meta and continue emitting values.
        if (partial_run_meta != -1) {
          int const remaining   = partial_run_total - partial_run_done;
          int const room        = out_end - out_base;
          int const cnt         = min(remaining, room);
          chunk_meta_v[0]       = partial_run_meta;
          chunk_out_off_v[1]    = cnt;
          s_run0_payload_offset = partial_run_done;
          n                     = 1;
          co                    = cnt;
          if (cnt < remaining) {
            partial_run_done += cnt;
          } else {
            partial_run_meta = -1;
          }
        }

        // Parse up to max_runs_per_chunk headers, stopping early if the output range
        // fills up or the encoded stream is exhausted.
        while (n < max_runs_per_chunk && (out_base + co) < out_end && cur_offset < stream_len) {
          uint32_t const level_run = get_vlq32(cur_offset, stream_len, staged_byte);
          int cnt;
          int meta;

          // Parquet RLE header format: LSB selects the encoding.
          //   bit 0 = 1  -> literal (bit-packed) run of `groups*8` values
          //   bit 0 = 0  -> RLE run of `level_run >> 1` copies of one value
          // The high bit of `meta` distinguishes the two at expand time.
          //
          // Assumption: (cur - s_start) fits in 31 bits, i.e. the encoded
          // level stream for a single Parquet page is < 2 GiB. This is
          // guaranteed by the Parquet format in practice (page payloads are
          // orders of magnitude smaller than 2 GiB) and is independent of
          // max_runs_per_chunk.
          if (level_run & 1u) {
            int const groups = level_run >> 1;
            cnt              = groups * 8;
            meta             = cur_offset | (1u << 31);
            cur_offset += groups * level_bits;
          } else {
            cnt  = level_run >> 1;
            meta = cur_offset;
            cur_offset += value_width;
          }

          // If this run overflows the requested output window, clamp it and
          // stash the remainder as a pending partial run for the next call.
          // Note `cur` has already been advanced past the *full* payload
          // above, which is what we want - the resumed call reads the
          // payload out of s_start via the saved meta offset.
          int const room = out_end - (out_base + co);
          if (cnt > room) {
            partial_run_meta  = meta;
            partial_run_total = cnt;
            partial_run_done  = room;
            cnt               = room;
          }
          co += cnt;
          chunk_meta_v[n]      = meta;
          chunk_out_off_v[++n] = co;
          if (partial_run_meta != -1) { break; }
        }
        cur           = s_start + cur_offset;
        s_chunk_runs  = n;
        s_chunk_total = co;
        s_base_out    = out_base;
      }
      __syncthreads();

      // ----- Phase 2: cooperative expand ------------------------------
      // All warps see the same chunk_out_off / chunk_meta tables. We split
      // the flat output range [0, chunk_total) into `num_warps` equal-ish
      // slices and each warp writes its slice.
      int const chunk_runs          = s_chunk_runs;
      int const chunk_total         = s_chunk_total;
      int const base_out            = s_base_out;
      int const run0_payload_offset = s_run0_payload_offset;

      if (chunk_runs == 0) { break; }

      int const per = (chunk_total + num_warps - 1) / num_warps;
      int const lo  = warp * per;
      int const hi  = min(lo + per, chunk_total);
      if (lo < hi) {
        // Binary-search chunk_out_off to find the first run that intersects
        // this warp's slice, then iterate forward until we pass `hi`.
        int const a = static_cast<int>(
                        cuda::std::upper_bound(
                          chunk_out_off_v.begin(), chunk_out_off_v.begin() + chunk_runs + 1, lo) -
                        chunk_out_off_v.begin()) -
                      1;
        for (int r = a; r < chunk_runs && chunk_out_off_v[r] < hi; ++r) {
          int const r_lo   = chunk_out_off_v[r];
          int const r_hi   = chunk_out_off_v[r + 1];
          int const seg_lo = max(r_lo, lo);
          int const seg_hi = min(r_hi, hi);
          int const meta   = chunk_meta_v[r];
          // For slot 0 of a resumed partial run add the already-emitted offset
          // so we read from the correct position in the payload.
          int const run_payload_off = (r == 0) ? run0_payload_offset : 0;

          // Two expand kernels, selected by the meta top-bit:
          //   literal (bit 31 set) -> each output position needs its own
          //     bit-field extract from the packed payload.
          //   RLE    (bit 31 clear) -> single value read once, broadcast
          //     across [seg_lo, seg_hi) by all lanes.
          // Note that we don't pay any divergence cost here since all threads
          // in the warp are processing the same run (meta), but we do lose
          // some occupancy due to carrying around the local registers needed
          // for both branches.
          if (meta & (1u << 31)) {
            int const payload_off = meta & 0x7fffffff;
            for (int p = seg_lo + lane; p < seg_hi; p += 32) {
              int const local = (p - r_lo) + run_payload_off;
              int bitpos      = local * level_bits;
              int source      = payload_off + (bitpos >> 3);
              bitpos &= 7;
              uint32_t level_val = 0;
              if (source < stream_len) { level_val = staged_byte(source); }
              ++source;
              if (level_bits > 8 - bitpos && source < stream_len) {
                level_val |= static_cast<uint32_t>(staged_byte(source)) << 8;
                ++source;
                if (level_bits > 16 - bitpos && source < stream_len) {
                  level_val |= static_cast<uint32_t>(staged_byte(source)) << 16;
                  ++source;
                  if (level_bits > 24 - bitpos && source < stream_len) {
                    level_val |= static_cast<uint32_t>(staged_byte(source)) << 24;
                  }
                }
              }
              level_val = (level_val >> bitpos) & level_mask;
              output[rolling_index<max_output_values>(base_out + p)] =
                static_cast<level_t>(level_val);
            }
          } else {
            // RLE run: read the single repeated value once from s_start,
            // assembling up to 4 payload bytes into `level_val` based on
            // level_bits, then warp_fill the whole segment.
            int const value_off = meta & 0x7fffffff;
            uint32_t level_val  = staged_byte(value_off);
            if constexpr (sizeof(level_t) > 1) {
              if (level_bits > 8) {
                level_val |= static_cast<uint32_t>(staged_byte(value_off + 1)) << 8;
                if constexpr (sizeof(level_t) > 2) {
                  if (level_bits > 16) {
                    level_val |= static_cast<uint32_t>(staged_byte(value_off + 2)) << 16;
                    if (level_bits > 24) {
                      level_val |= static_cast<uint32_t>(staged_byte(value_off + 3)) << 24;
                    }
                  }
                }
              }
            }
            level_t const fill = static_cast<level_t>(level_val);
            for (int q = base_out + seg_lo + lane; q < base_out + seg_hi; q += 32) {
              output[rolling_index<max_output_values>(q)] = fill;
            }
          }
        }
      }
      // Barrier before rewriting the shared tables on the next iteration.
      __syncthreads();

      // Pipeline consumer_wait/release is the only staging barrier here; adding
      // a per-chunk block-wide sync repeats the BSS staging regression pattern.
      if (current_pending) { smem_pipe->consumer_release(); }
      if (next_pending) {
        smem_pipe->consumer_wait();
        mark_loaded(current_chunk + 1);
        smem_pipe->consumer_release();
      }

      out_pos_total = base_out + chunk_total;
    }

    int const decoded = out_pos_total - cur_values;
    cur_values        = out_pos_total;
    return decoded;
  }

  __device__ inline int decode_next(int t, int count)
  {
    // Fast path: level_bits == 0 means every level is implicitly 0, so no
    // headers or payloads need parsing. This is a very common case: columns
    // with no nulls (especially non-nested ones) have all-zero definition
    // levels. Handled here so both decode_next_ring and decode_next_chunked
    // stay focused on the general RLE path.
    int const output_count = min(count, total_values - cur_values);
    if (level_bits == 0) {
      int written = 0;
      while (written < output_count) {
        int const batch_size = min(num_rle_stream_decode_threads, output_count - written);
        if (t < batch_size) {
          output[rolling_index<max_output_values>(cur_values + written + t)] = 0;
        }
        written += batch_size;
      }
      cur_values += output_count;
      return output_count;
    }
    if constexpr (use_chunked_expand) {
      return decode_next_chunked(t, count);
    } else {
      return decode_next_ring(t, count);
    }
  }

  __device__ inline int skip_runs(int target_count)
  {
    // we want to process all runs UP TO BUT NOT INCLUDING the run that overlaps with the skip
    // amount so threads spin like crazy on fill_run_batch(), skipping writing unnecessary run info.
    // then when it hits the one that matters, we don't process it at all and bail as if we never
    // started basically we're setting up the rle_stream vars necessary to start fill_run_batch for
    // the first time
    while (cur < end) {
      rle_run run;
      int run_bytes = get_rle_run_info(run);

      if ((output_pos + run.size) > target_count) {
        return output_pos;  // bail! we've reached the starting run
      }

      // skip this run
      output_pos += run.size;
      cur += run_bytes;
    }

    return output_pos;  // we skipped everything
  }

  __device__ inline int skip_decode(int t, int count)
  {
    int const output_count = min(count, total_values - cur_values);

    // if level_bits == 0, there's nothing to do
    // a very common case: columns with no nulls, especially if they are non-nested
    cur_values = (level_bits == 0) ? output_count : skip_runs(output_count);
    return cur_values;
  }

  __device__ inline int decode_next(int t) { return decode_next(t, max_output_values); }
};

template <typename level_t, int decode_threads, int max_output_values>
using rle_stream_chunked = rle_stream<level_t, decode_threads, max_output_values, true, false>;

}  // namespace cudf::io::parquet::detail
