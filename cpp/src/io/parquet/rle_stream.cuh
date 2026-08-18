/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/utilities/block_utils.cuh"
#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/std/algorithm>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/span>

namespace cudf::io::parquet::detail {

namespace cg = cooperative_groups;

template <int num_threads>
__device__ constexpr int rle_stream_required_run_buffer_size()
{
  constexpr int num_rle_stream_decode_warps = (num_threads / cudf::detail::warp_size) - 1;
  return (num_rle_stream_decode_warps * 2);
}

/**
 * @brief Read a 32-bit varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The 32-bit value read
 */
inline __device__ uint32_t get_vlq32(uint8_t const*& cur, uint8_t const* end)
{
  uint32_t v = *cur++;
  if (v >= 0x80 && cur < end) {
    v = (v & 0x7f) | ((*cur++) << 7);
    if (v >= (0x80 << 7) && cur < end) {
      v = (v & ((0x7f << 7) | 0x7f)) | ((*cur++) << 14);
      if (v >= (0x80 << 14) && cur < end) {
        v = (v & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 21);
        if (v >= (0x80 << 21) && cur < end) {
          v = (v & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 28);
        }
      }
    }
  }
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
// header-parse phase and a group.sync() at the end), but competes with occupancy in
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
          int smem_stage_size_bytes = 8 * 1024>
struct rle_stream {
  static constexpr int num_rle_stream_decode_threads = decode_threads;
  // the -1 here is for the look-ahead warp that fills in the list of runs to be decoded
  // in an overlapped manner. so if we had 16 total warps:
  // - warp 0 would be filling in batches of runs to be processed
  // - warps 1-15 would be decoding the previous batch of runs generated
  static constexpr int num_rle_stream_decode_warps =
    (num_rle_stream_decode_threads / cudf::detail::warp_size) - 1;

  static constexpr int run_buffer_size = rle_stream_required_run_buffer_size<decode_threads>();

  // Bit packing of `run_desc` (chunked-expand path). The 32-bit descriptor
  // stores a byte offset into the encoded level stream (`cur - s_start`) in
  // its low 31 bits and uses the top bit as a flag: 1 = literal (bit-packed)
  // run, 0 = RLE (repeated-value) run. Keeping both in one word lets Phase 1
  // publish a single 32-bit value per run into shared memory and lets Phase 2
  // dispatch on the flag without a second load.
  //
  // Invariant enforced by `cudf_assert` at parse time: (cur - s_start) fits in
  // 31 bits, i.e. the encoded level stream for a single Parquet page is < 2
  // GiB. Parquet page payloads are orders of magnitude smaller than this in
  // practice.
  static constexpr uint32_t run_desc_literal_flag = 1u << 31;
  static constexpr uint32_t run_desc_offset_mask  = 0x7fffffffu;

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
  static constexpr int smem_stage_size = smem_stage_size_bytes;

  // Ring-mode streams need a shared-memory ring buffer of run headers to
  // coordinate the producer/consumer warps in decode_next_ring. Chunked-expand
  // streams parse run headers directly into per-chunk shared tables and never
  // touch `runs`, so we forbid supplying one to catch accidental waste.
  __device__ rle_stream(rle_run* _runs)
    requires(!use_chunked_expand)
    : runs(_runs)
  {
  }
  __device__ rle_stream()
    requires(use_chunked_expand)
    : runs(nullptr)
  {
  }

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

    cudf_assert(stage_capacity >= 0 and stage_capacity <= smem_stage_size);

    // If smem staging is active, use cuda::memcpy_async for a
    // block-cooperative global-to-shared copy that automatically dispatches to
    // the best copy path (cp.async, cp.async.bulk, or TMA) depending on the
    // hardware. Callers must provide a copy_barrier when using smem staging,
    // and must issue copy_barrier->arrive_and_wait() after init() to complete
    // the async copy.
    if (_smem_stage != nullptr) {
      auto* const smem_stage =
        static_cast<uint8_t const*>(cuda::std::assume_aligned<16>(_smem_stage));
      auto const len = static_cast<int>(cuda::std::distance(_start, _end));
      if (len > 0 && len <= stage_capacity) {
        cuda::memcpy_async(group, _smem_stage, _start, static_cast<size_t>(len), *_copy_barrier);
        // Rebase the parse cursor and end onto the shared copy. All downstream
        // reads (get_rle_run_info, decode, skip_runs) follow cur/end and now hit
        // shared memory with no other changes required.
        cur = smem_stage;
        end = smem_stage + len;
      }
    }
    // Anchor s_start to the (possibly rebased) cur so chunked-expand meta
    // offsets index into the same memory space that the parse cursor uses.
    s_start = cur;
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

  template <typename Group>
  __device__ inline int decode_next_ring(Group const& group, int count)
  {
    int const output_count = min(count, total_values - cur_values);

    auto const warp          = cg::tiled_partition<cudf::detail::warp_size>(group);
    int const warp_id        = warp.meta_group_rank();
    int const warp_decode_id = warp_id - 1;
    int const warp_lane      = warp.thread_rank();

    __shared__ int values_processed_shared;
    __shared__ int decode_index_shared;
    __shared__ int fill_index_shared;
    // Do not use cg::invoke_one here: rle_stream member state is per-thread,
    // so persistent state must be owned by a stable, well-defined thread.
    if (group.thread_rank() == 0) {
      values_processed_shared = 0;
      decode_index_shared     = decode_index;
      fill_index_shared       = fill_index;
    }

    group.sync();

    fill_index = fill_index_shared;

    while (true) {
      // protect against threads advancing past the end of this loop
      // and updating shared variables.
      group.sync();

      // warp 0 reads ahead and fills `runs` array to be decoded by remaining warps.
      if (warp_id == 0) {
        // fill the next set of runs. fill_runs will generally be the bottleneck for any
        // kernel that uses an rle_stream.
        // Do not use cg::invoke_one here: fill_run_batch() advances per-thread
        // stream cursors, so the ring producer must always be lane 0.
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
      // decode_index == -1 means "first iteration", so we should skip decoding.
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

          warp.sync();
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
      group.sync();
      decode_index = decode_index_shared;
      fill_index   = fill_index_shared;
      if (values_processed_shared >= output_count) { break; }
    }

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
   * The current sole caller passes max_output_values = INT_MAX, so a single
   * RLE run can never exceed the requested `count`. A cudf_assert in the
   * header-parse loop enforces this invariant; a future caller that needs to
   * split runs across calls must restore the partial-run resume machinery
   * removed in this commit.
   */
  template <typename Group>
  __device__ inline int decode_next_chunked(Group const& group, int count)
  {
    int const output_count = min(count, total_values - cur_values);

    // Per-chunk shared-memory scratch. `chunk_out_off[i]` is the exclusive
    // prefix-sum of run lengths within the current chunk, so run `i`
    // occupies output positions [chunk_out_off[i], chunk_out_off[i+1]).
    // `chunk_meta[i]` encodes both the payload offset (into s_start) and,
    // in the top bit, whether the run is literal (1) or RLE (0).
    __shared__ int chunk_out_off[max_runs_per_chunk + 1];
    __shared__ uint32_t chunk_meta[max_runs_per_chunk];
    cuda::std::span<int> const chunk_out_off_v{chunk_out_off, max_runs_per_chunk + 1};
    cuda::std::span<uint32_t> const chunk_meta_v{chunk_meta, max_runs_per_chunk};
    __shared__ int s_chunk_runs;   // number of runs parsed in this chunk (num_runs)
    __shared__ int s_chunk_total;  // sum of run lengths in this chunk (run_prefix_end)
    __shared__ int s_base_out;     // absolute output pos where this chunk starts

    auto const warp       = cg::tiled_partition<cudf::detail::warp_size>(group);
    int const lane        = warp.thread_rank();
    int const warp_id     = warp.meta_group_rank();
    int const num_warps   = warp.meta_group_size();
    int const value_width = cudf::util::div_rounding_up_unsafe(level_bits, 8);
    // Bit mask used to extract a single level from a bit-packed literal-run
    // payload word. Invariant across the whole call; hoisted out of the
    // phase-2 expand loop to keep it out of the hot register set.
    uint32_t const level_mask = (level_bits == 32) ? 0xffffffffu : ((1u << level_bits) - 1);
    int out_pos_total         = cur_values;
    int const out_end         = cur_values + output_count;

    // Outer loop: process the requested output range in chunks of up to
    // `max_runs_per_chunk` runs at a time until we have emitted `output_count`
    // values or run out of encoded input.
    while (out_pos_total < out_end) {
      // ----- Phase 1: single-thread run-header parse ------------------
      // Thread 0 walks the encoded stream, decoding VLQ run headers and
      // filling chunk_out_off / chunk_meta. Do not use cg::invoke_one here:
      // it may choose different threads across calls, but `cur` is per-thread
      // rle_stream state that must persist on thread 0.
      // The other threads wait at the group.sync() below. This is cheap because
      // it is bounded by max_runs_per_chunk headers and header parsing is
      // inherently serial.
      if (group.thread_rank() == 0) {
        int run_prefix_end = 0;
        int num_runs       = 0;
        int out_base       = out_pos_total;
        chunk_out_off_v[0] = 0;
        // Parse up to max_runs_per_chunk headers, stopping early if the output range
        // fills up or the encoded stream is exhausted.
        while (num_runs < max_runs_per_chunk && (out_base + run_prefix_end) < out_end &&
               cur < end) {
          uint32_t const level_run = get_vlq32(cur, end);

          // Parquet RLE header format: LSB selects the encoding.
          //   bit 0 = 1  -> literal (bit-packed) run of `groups*8` values
          //   bit 0 = 0  -> RLE run of `level_run >> 1` copies of one value
          // The high bit of `run_desc` distinguishes the two at expand time;
          // see `run_desc_literal_flag` / `run_desc_offset_mask` for the
          // bit layout and the 31-bit offset invariant enforced below.
          cudf_assert(static_cast<uint64_t>(cur - s_start) < (uint64_t{run_desc_offset_mask} + 1));
          int run_len;
          uint32_t run_desc;
          if (level_run & 1u) {
            int const groups = level_run >> 1;
            run_len          = groups * 8;
            run_desc         = static_cast<uint32_t>(cur - s_start) | run_desc_literal_flag;
            cur += groups * level_bits;
          } else {
            run_len  = level_run >> 1;
            run_desc = static_cast<uint32_t>(cur - s_start);
            cur += value_width;
          }
          // Clamp the run to the remaining output window. A run can legally
          // straddle out_end when row-range filtering makes output_count
          // smaller than INT_MAX (e.g. skip_rows / num_rows / bounds-page
          // filtering in preprocess_levels_kernel). Without clamping, Phase 2
          // would write past the output buffer causing silent data corruption.
          // `cur` has already been advanced past the full payload above, which
          // is correct: we emit only `run_len` values but do not need to
          // re-parse the header on any subsequent call (the outer loop exits
          // after this chunk because out_pos_total will equal out_end).
          int const room = out_end - (out_base + run_prefix_end);
          run_len        = min(run_len, room);
          run_prefix_end += run_len;
          chunk_meta_v[num_runs]      = run_desc;
          chunk_out_off_v[++num_runs] = run_prefix_end;
        }
        s_chunk_runs  = num_runs;
        s_chunk_total = run_prefix_end;
        s_base_out    = out_base;
      }
      group.sync();

      // ----- Phase 2: cooperative expand ------------------------------
      // All warps see the same chunk_out_off / chunk_meta tables. We split
      // the flat output range [0, chunk_total) into `num_warps` equal-ish
      // slices and each warp writes its slice.
      int const chunk_runs  = s_chunk_runs;
      int const chunk_total = s_chunk_total;
      int const base_out    = s_base_out;

      if (chunk_runs == 0) { break; }

      int const per = cudf::util::div_rounding_up_safe(chunk_total, num_warps);
      int const lo  = warp_id * per;
      int const hi  = min(lo + per, chunk_total);
      if (lo < hi) {
        // Per-lane expand: each lane owns output positions
        //   out_pos = lo + lane, lo + lane + 32, lo + lane + 64, ...
        // and finds its own run. run_idx starts by binary-search on the
        // lane's first out_pos, then advances forward by linear walk (usually
        // 0 steps when still in the same run). Across all 32 lanes the
        // linear walks amortize to <= (num_runs_in_slice / 32) warp cycles
        // total.
        //
        // This keeps all 32 lanes writing on every iteration, instead of the
        // per-run loop where only lanes 0..(run_len-1) do useful work on
        // short runs.
        int out_pos = lo + lane;
        int run_idx =
          static_cast<int>(cuda::std::upper_bound(chunk_out_off_v.begin(),
                                                  chunk_out_off_v.begin() + chunk_runs + 1,
                                                  out_pos) -
                           chunk_out_off_v.begin()) -
          1;
        while (out_pos < hi) {
          // Linear walk forward: no iterations if we're still in the same
          // run (long run case), 1+ iterations only when out_pos crosses one
          // or more short-run boundaries.
          while (run_idx < chunk_runs && chunk_out_off_v[run_idx + 1] <= out_pos) {
            ++run_idx;
          }
          int const run_start_out = chunk_out_off_v[run_idx];
          uint32_t const run_desc = chunk_meta_v[run_idx];

          if (run_desc & run_desc_literal_flag) {
            // Literal (bit-packed) run: bit-field extract for this lane's
            // out_pos.
            uint32_t const payload_off = run_desc & run_desc_offset_mask;
            uint8_t const* payload     = s_start + payload_off;
            int const local            = out_pos - run_start_out;
            int bitpos                 = local * level_bits;
            uint8_t const* source      = payload + (bitpos >> 3);
            bitpos &= 7;
            uint32_t level_val;
            if (source + sizeof(uint32_t) <= end) {
              // Fast path: whole 32-bit field is in-bounds, so one unaligned
              // load replaces up to four dependent byte reads.
              level_val = cudf::io::unaligned_load<uint32_t>(source);
            } else {
              // Tail path: within the last 4 bytes of the encoded stream, so
              // fall back to per-byte reads and guard each against `end`.
              level_val = 0;
              if (source < end) { level_val = source[0]; }
              if (level_bits > 8 - bitpos && (source + 1) < end) {
                level_val |= static_cast<uint32_t>(source[1]) << 8;
                if (level_bits > 16 - bitpos && (source + 2) < end) {
                  level_val |= static_cast<uint32_t>(source[2]) << 16;
                  if (level_bits > 24 - bitpos && (source + 3) < end) {
                    level_val |= static_cast<uint32_t>(source[3]) << 24;
                  }
                }
              }
            }
            level_val = (level_val >> bitpos) & level_mask;
            output[rolling_index<max_output_values>(base_out + out_pos)] =
              static_cast<level_t>(level_val);
          } else {
            // RLE run: read the single repeated value from s_start.
            // Guard each byte against `end` to match the literal path, since a
            // truncated Parquet page can leave `run_desc`'s payload offset
            // pointing within [s_start, end) while `level_bits > 8` implies
            // vptr[1..3] may lie past `end`.
            uint8_t const* vptr = s_start + (run_desc & run_desc_offset_mask);
            uint32_t level_val  = 0;
            if (vptr < end) { level_val = vptr[0]; }
            if constexpr (sizeof(level_t) > 1) {
              if (level_bits > 8 && (vptr + 1) < end) {
                level_val |= static_cast<uint32_t>(vptr[1]) << 8;
                if constexpr (sizeof(level_t) > 2) {
                  if (level_bits > 16 && (vptr + 2) < end) {
                    level_val |= static_cast<uint32_t>(vptr[2]) << 16;
                    if (level_bits > 24 && (vptr + 3) < end) {
                      level_val |= static_cast<uint32_t>(vptr[3]) << 24;
                    }
                  }
                }
              }
            }
            output[rolling_index<max_output_values>(base_out + out_pos)] =
              static_cast<level_t>(level_val);
          }
          out_pos += warp.size();
        }
      }
      // Barrier before rewriting the shared tables on the next iteration.
      group.sync();

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
    //
    // The write uses `cur_values + written + t` rather than `written + t` so it
    // targets the correct ring slots regardless of how many times decode_next
    // has already been called. No current caller enters this fast path with
    // cur_values > 0 -- the writer floors dict_rle_bits >= 1 in chunk_dict.cu,
    // and the REPETITION/DEFINITION decoders in decode_preprocess.cu are
    // single-call -- so all reachable end-to-end tests still pass with the
    // simpler `written + t` form. This invariant is preserved defensively so
    // any future caller that iterates decode_next with level_bits == 0 stays
    // correct without a silent off-by-one in the ring buffer.
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
      return decode_next_chunked(cg::this_thread_block(), count);
    } else {
      return decode_next_ring(cg::this_thread_block(), count);
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
    static_assert(not use_chunked_expand, "skip_decode is not supported by chunked-expand");
    int const output_count = min(count, total_values - cur_values);

    // if level_bits == 0, there's nothing to do
    // a very common case: columns with no nulls, especially if they are non-nested
    cur_values = (level_bits == 0) ? output_count : skip_runs(output_count);
    return cur_values;
  }

  __device__ inline int decode_next(int t) { return decode_next(t, max_output_values); }
};

template <typename level_t, int decode_threads, int max_output_values>
using rle_stream_chunked = rle_stream<level_t, decode_threads, max_output_values, true>;

}  // namespace cudf::io::parquet::detail
