/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "page_decode.cuh"

#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

#include <climits>

namespace cudf::io::parquet::detail {

// DELTA_XXX encoding support
//
// DELTA_BINARY_PACKED is used for INT32 and INT64 data types. Encoding begins with a header
// containing a block size, number of mini-blocks in each block, total value count, and first
// value. The first three are ULEB128 variable length ints, and the last is a zigzag ULEB128
// varint.
//   -- the block size is a multiple of 128
//   -- the mini-block count is chosen so that each mini-block will contain a multiple of 32 values
//   -- the value count includes the first value stored in the header
//
// It seems most Parquet encoders will stick with a block size of 128, and 4 mini-blocks of 32
// elements each. arrow-rs will use a block size of 256 for 64-bit ints.
//
// Following the header are the data blocks. Each block is further divided into mini-blocks, with
// each mini-block having its own encoding bitwidth. Each block begins with a header containing a
// zigzag ULEB128 encoded minimum delta value, followed by an array of uint8 bitwidths, one entry
// per mini-block. While encoding, the lowest delta value is subtracted from all the deltas in the
// block to ensure that all encoded values are positive. The deltas for each mini-block are bit
// packed using the same encoding as the RLE/Bit-Packing Hybrid encoder.

// The DELTA_BINARY_PACKED spec requires the number of values in a mini-block to be a multiple of
// 32. The decoders rely on the coincidence that this also equals warp size; they produce values
// in warp_size-wide passes, so it must divide every spec-valid mini-block size.
constexpr int delta_mini_block_size_multiple = 32;
static_assert(delta_mini_block_size_multiple % cudf::detail::warp_size == 0,
              "warp_size must divide the DELTA mini-block size multiple; the pass-based decoders "
              "assume warp_size divides every spec-valid mini-block size");

// The decode loops produce up to two (warp_size-wide) passes per iteration: pages whose
// mini-blocks hold at least two passes keep the two-pass batch the loops have always used, and
// running several passes back to back amortizes the per-iteration synchronization.
constexpr int delta_max_batch_size = 2 * cudf::detail::warp_size;

// The rolling buffer must hold two batches in flight (the consumer drains one batch while the
// producer decodes the next), plus one slot for the first value from the block header: it is not
// stored in the buffer, but it still impacts buffer indexing and we need to account for it to
// avoid race conditions.
constexpr int delta_rolling_buf_size = (2 * delta_max_batch_size) + 1;

// Upper bound on the warps that may cooperate in decode_next_passes_wide(). Sized for the
// 128-thread decode block; the method static_asserts against it.
constexpr int delta_max_decode_warps = 4;

/**
 * @brief Read a ULEB128 varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The value read
 */
inline __device__ uleb128_t get_uleb128(uint8_t const*& cur, uint8_t const* end)
{
  uleb128_t v = 0, l = 0, c;
  while (cur < end) {
    c = *cur++;
    v |= (c & 0x7f) << l;
    l += 7;
    if ((c & 0x80) == 0) { return v; }
  }
  return v;
}

/**
 * @brief Read a ULEB128 zig-zag encoded varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The value read
 */
inline __device__ zigzag128_t get_zz128(uint8_t const*& cur, uint8_t const* end)
{
  uleb128_t u = get_uleb128(cur, end);
  return static_cast<zigzag128_t>((u >> 1u) ^ -static_cast<zigzag128_t>(u & 1));
}

struct delta_binary_decoder {
  uint8_t const* block_start;  // start of data, but updated as data is read
  uint8_t const* block_end;    // end of data
  uleb128_t block_size;        // usually 128, must be multiple of 128
  uleb128_t mini_block_count;  // usually 4, chosen such that block_size/mini_block_count is a
                               // multiple of 32
  uleb128_t value_count;       // total values encoded in the block
  zigzag128_t first_value;     // initial value, stored in the header
  zigzag128_t last_value;      // last value decoded

  uint32_t values_per_mb;      // block_size / mini_block_count, must be multiple of 32
  uint32_t current_value_idx;  // current value index, initialized to 0 at start of block
  uint32_t cur_pass;           // current pass within the mini-block

  zigzag128_t cur_min_delta;     // min delta for the block
  uint32_t cur_mb;               // index of the current mini-block within the block
  uint8_t const* cur_mb_start;   // pointer to the start of the current mini-block data
  uint8_t const* cur_bitwidths;  // pointer to the bitwidth array in the block
  bool error;                    // flag to catch malformed headers

  // Per-warp pass sums, used only by decode_next_passes_wide() to carry the running total across
  // warps. A member rather than a function-scope __shared__ array because a kernel may run two
  // decoders concurrently on different warps, and a single static allocation would alias between
  // them. The scan itself stays shuffle-only.
  zigzag128_t warp_pass_total[delta_max_decode_warps];

  zigzag128_t value[delta_rolling_buf_size];  // circular buffer of delta values

  // returns the value stored in the `value` array at index
  // `rolling_index<delta_rolling_buf_size>(idx)`. If `idx` is `0`, then return `first_value`.
  __device__ constexpr zigzag128_t value_at(size_type idx)
  {
    return idx == 0 ? first_value : value[rolling_index<delta_rolling_buf_size>(idx)];
  }

  // returns the number of values encoded in the block data. when all_values is true,
  // account for the first value in the header. otherwise just count the values encoded
  // in the mini-block data.
  __device__ constexpr uint32_t num_encoded_values(bool all_values)
  {
    return value_count == 0 ? 0 : all_values ? value_count : value_count - 1;
  }

  // index just past the values decode_next_pass() has produced so far (0 before the first pass,
  // even though the header value already occupies index 0)
  __device__ uint32_t next_pass_start_idx()
  {
    return current_value_idx + cur_pass * cudf::detail::warp_size;
  }

  // read mini-block header into state object. should only be called from init_binary_block or
  // setup_next_mini_block. header format is:
  //
  // | min delta (int) | bit-width array (1 byte * mini_block_count) |
  //
  // on exit db->cur_mb is 0 and db->cur_mb_start points to the first mini-block of data, or
  // nullptr if out of data.
  // is_decode indicates whether this is being called from initialization code (false) or
  // the actual decoding (true)
  inline __device__ void init_mini_block(bool is_decode)
  {
    cur_mb       = 0;
    cur_mb_start = nullptr;

    if (current_value_idx < num_encoded_values(is_decode)) {
      auto d_start  = block_start;
      cur_min_delta = get_zz128(d_start, block_end);
      cur_bitwidths = d_start;

      d_start += mini_block_count;
      cur_mb_start = d_start;
    }
  }

  // read delta binary header into state object. should be called on thread 0. header format is:
  //
  // | block size (uint) | mini-block count (uint) | value count (uint) | first value (int) |
  //
  // also initializes the first mini-block before exit
  inline __device__ void init_binary_block(uint8_t const* d_start, uint8_t const* d_end)
  {
    block_end        = d_end;
    block_size       = get_uleb128(d_start, d_end);
    mini_block_count = get_uleb128(d_start, d_end);
    value_count      = get_uleb128(d_start, d_end);
    first_value      = get_zz128(d_start, d_end);
    last_value       = first_value;

    current_value_idx = 0;
    cur_pass          = 0;
    error             = false;

    // Validate the header against the DELTA_BINARY_PACKED spec: the mini-block count must evenly
    // divide the block size, and each mini-block must hold a multiple of 32 values. The decoders
    // rely on the latter to advance from one mini-block to the next.
    if (mini_block_count == 0 or block_size == 0 or (block_size % mini_block_count) != 0 or
        ((block_size / mini_block_count) % delta_mini_block_size_multiple) != 0) {
      error         = true;
      value_count   = 0;
      values_per_mb = 1;
      block_start   = d_end;
      cur_mb        = 0;
      cur_mb_start  = d_end;
      cur_bitwidths = d_end;
      return;
    }

    values_per_mb = block_size / mini_block_count;

    // init the first mini-block
    block_start = d_start;

    // only call init if there are actually encoded values
    if (value_count > 1) { init_mini_block(false); }
  }

  // skip to the start of the next mini-block. should only be called on thread 0.
  // calls init_binary_block if currently on the last mini-block in a block.
  // is_decode indicates whether this is being called from initialization code (false) or
  // the actual decoding (true)
  inline __device__ void setup_next_mini_block(bool is_decode)
  {
    if (current_value_idx >= num_encoded_values(is_decode)) { return; }

    current_value_idx += values_per_mb;

    // just set pointer to start of next mini_block
    if (cur_mb < mini_block_count - 1) {
      cur_mb_start += cur_bitwidths[cur_mb] * values_per_mb / CHAR_BIT;
      cur_mb++;
    }
    // out of mini-blocks, start a new block
    else {
      block_start = cur_mb_start + cur_bitwidths[cur_mb] * values_per_mb / CHAR_BIT;
      init_mini_block(is_decode);
    }
  }

  // given start/end pointers in the data, find the end of the binary encoded block. when done,
  // `this` will be initialized with the correct start and end positions. returns the end, which is
  // start of data/next block. should only be called from thread 0.
  inline __device__ uint8_t const* find_end_of_block(uint8_t const* start, uint8_t const* end)
  {
    // read block header
    init_binary_block(start, end);

    // test for no encoded values. a single value will be in the block header.
    if (value_count <= 1) { return block_start; }

    // read mini-block headers and skip over data
    while (current_value_idx < num_encoded_values(false)) {
      setup_next_mini_block(false);
    }
    // calculate the correct end of the block
    auto const* const new_end = cur_mb == 0 ? block_start : cur_mb_start;
    // re-init block with correct end
    init_binary_block(start, new_end);
    return new_end;
  }

  // account for the first value from the block header before the first mini-block is decoded.
  // the first value is not encoded in the mini-block data, but it still occupies index 0 of the
  // value stream. returns true if there are more values to decode after the header value.
  // called by all threads in a single warp `warp`.
  inline __device__ bool advance_past_first_value(
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    if (current_value_idx >= value_count) { return false; }

    if (current_value_idx == 0) {
      // make sure all threads access current_value_idx above before incrementing
      warp.sync();
      if (warp.thread_rank() == 0) { current_value_idx++; }
      warp.sync();
      if (current_value_idx >= value_count) { return false; }
    }
    return true;
  }

  // decode a single warp_size-wide pass (indexed by `pass`) of the current mini-block and convert
  // the deltas to values (see decode_next_pass). called by all threads in a single warp `warp`.
  inline __device__ void calc_mini_block_pass(
    uint32_t pass, cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    using cudf::detail::warp_size;

    auto const lane_id     = static_cast<int>(warp.thread_rank());
    uint32_t const mb_bits = cur_bitwidths[cur_mb];

    // position at the end of this pass's values since the following calculates negative indexes
    auto const d_start = cur_mb_start + (pass + 1) * (warp_size * mb_bits / CHAR_BIT);

    // unpack deltas. modified from version in decode_dictionary_indices(), but
    // that one only unpacks up to bitwidths of 24. simplified some since this
    // will always do batches of 32.
    // NOTE: because this needs to handle up to 64 bits, the branching used in the other
    // implementation has been replaced with a loop. While this uses more registers, the
    // looping version is just as fast and easier to read.
    zigzag128_t delta = 0;
    if (current_value_idx + pass * warp_size + lane_id < value_count) {
      // ofs is non-positive, so the arithmetic shift and mask compute the byte offset and leading
      // bit position as floored division/modulo by CHAR_BIT (a plain / and % would round
      // toward 0)
      int32_t ofs      = (lane_id - warp_size) * mb_bits;
      uint8_t const* p = d_start + (ofs >> 3);
      ofs &= 7;
      if (p < block_end) {
        uint32_t c = CHAR_BIT - ofs;  // 0 - 7 bits
        delta      = (*p++) >> ofs;

        while (c < mb_bits && p < block_end) {
          delta |= static_cast<zigzag128_t>(*p++) << c;
          c += CHAR_BIT;
        }
        delta &= (static_cast<zigzag128_t>(1) << mb_bits) - 1;
      }
    }

    // add min delta to get true delta
    delta += cur_min_delta;

    // do inclusive scan to get value - first_value at each position. cg::inclusive_scan is
    // shuffle-based and carries no shared storage, so any number of delta decoders (e.g. the
    // prefix and suffix decoder warps of the DELTA_BYTE_ARRAY kernels) can run it concurrently,
    // each over its own warp tile, with no risk of aliasing.
    delta = cg::inclusive_scan(warp, delta, cg::plus<int64_t>{});

    // now add first value from header or last value from previous pass to get true value
    delta += last_value;
    int const value_idx =
      rolling_index<delta_rolling_buf_size>(current_value_idx + warp_size * pass + lane_id);
    value[value_idx] = delta;

    // lanes can reach this point at different times (the bit-unpacking loop above has a
    // lane-dependent trip count), so make sure every lane's read of last_value above has
    // happened before lane 31 overwrites it below.
    warp.sync();

    // save value from last lane in warp. this will become the 'first value' added to the
    // deltas calculated in the next pass (or invocation).
    if (lane_id == warp_size - 1) { last_value = delta; }
    warp.sync();
  }

  // decodes and discards values so the decoder resumes at the pass boundary at or just past
  // `skip`. the up to warp_size - 1 values decoded beyond `skip` stay resident in the rolling
  // buffer for the consumer, which resumes reading at `skip`. works for any mini-block size.
  // called by all threads in a thread block (`block`); the decode runs on warp 0 (`warp`).
  inline __device__ void skip_values(
    int skip,
    cg::thread_block const& block,
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    while (next_pass_start_idx() < static_cast<uint32_t>(skip) &&
           current_value_idx < num_encoded_values(true)) {
      // decode_next_pass only runs in warp 0, but advances decoder state everyone reads,
      // so everyone must sync around it
      block.sync();
      if (warp.meta_group_rank() == 0) { decode_next_pass(warp); }
      block.sync();
    }
  }

  // Decodes and skips values until the pass containing `skip` has been decoded, keeping a
  // running sum of the skipped values (indices below `skip`) and returning it. Values decoded
  // beyond `skip` stay resident in the rolling buffer for the consumer. Works for any
  // mini-block size. Called by all threads in warp 0 (`warp`); the result is only valid on
  // thread 0. This is intended for use only by the DELTA_LENGTH_BYTE_ARRAY decoder.
  inline __device__ size_t skip_values_and_sum(
    int skip, cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    using cudf::detail::warp_size;
    // DELTA_LENGTH_BYTE_ARRAY lengths are encoded as INT32 by convention (since the PLAIN encoding
    // uses 4-byte lengths).
    using delta_length_type = int32_t;
    auto const t            = warp.thread_rank();

    // initialize sum with first value, which is stored in the block header. cast to
    // `delta_length_type` to ensure the value is interpreted properly before promoting it
    // back to `size_t`.
    size_t sum = static_cast<delta_length_type>(value_at(0));

    // if only skipping one value, we're done already
    if (skip == 1) { return sum; }

    while (next_pass_start_idx() < static_cast<uint32_t>(skip) &&
           current_value_idx < num_encoded_values(true)) {
      // the pass decoded below produces indices [pass_first, pass_first + warp_size); the
      // header value at index 0 is not part of any pass and is already in `sum`
      auto const pass_first = max(next_pass_start_idx(), 1u);
      decode_next_pass(warp);

      auto const idx      = pass_first + t;
      size_t const val    = idx < static_cast<uint32_t>(skip) && idx < value_count
                              ? static_cast<delta_length_type>(value_at(idx))
                              : 0;
      auto const warp_sum = cg::reduce(warp, val, cg::plus<size_t>{});
      if (t == 0) { sum += warp_sum; }
      warp.sync();
    }

    return sum;
  }

  // decode the next warp_size-wide pass of the current mini-block into db->value, advancing to
  // the next mini-block once all of its passes have been decoded. Decoding a single pass at a
  // time keeps the rolling buffer footprint independent of the mini-block size. Should only be
  // called by a single warp `warp`. NOTE: lane 0's state updates are not synchronized on exit;
  // the caller must synchronize the warp (or block) before the next call so all lanes observe
  // them.
  inline __device__ void decode_next_pass(
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    using cudf::detail::warp_size;

    if (not advance_past_first_value(warp)) { return; }

    // unpack one pass of deltas and save in db->value
    calc_mini_block_pass(cur_pass, warp);

    // advance within the mini-block; move to the next mini-block once all passes are decoded
    if (warp.thread_rank() == 0) {
      if (++cur_pass == values_per_mb / warp_size) {
        cur_pass = 0;
        setup_next_mini_block(true);
      }
    }
  }

  // Decode up to `num_warps_t` warp_size-wide passes of the *current* DELTA block at once, one
  // pass per warp, and return each lane's value in a register rather than writing `value[]`.
  // Called by all threads of a `block` holding exactly `num_warps_t` warps.
  //
  // A single call never crosses into the next DELTA block. That is what makes the parallelism
  // available: within a block the header has already been parsed into `cur_min_delta` and
  // `cur_bitwidths`, so every pass is addressable with O(num_warps_t) pointer arithmetic, whereas
  // finding the next block's first byte requires a serial varint parse of its header. The cap
  // costs nothing in practice -- the spec's `block_size % 128 == 0` means a block holds a multiple
  // of four passes -- but `init_binary_block` does not validate that, so `active` is computed
  // rather than assumed.
  //
  // On return `lane_value` holds the value for stream index `lane_idx` on every lane whose warp
  // was active and whose `lane_idx < value_count`; `active_values` is how many stream indices the
  // call produced, starting at the pre-call `next_pass_start_idx()`. Returns false at end of page.
  // Ends with a block barrier, so all state is published on exit.
  template <int num_warps_t>
  inline __device__ bool decode_next_passes_wide(
    cg::thread_block const& block,
    cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp,
    zigzag128_t& lane_value,
    uint32_t& lane_idx,
    uint32_t& active_values)
  {
    using cudf::detail::warp_size;
    static_assert(num_warps_t <= delta_max_decode_warps,
                  "warp_pass_total is sized for delta_max_decode_warps");

    if (current_value_idx >= value_count) { return false; }

    // The header's first value occupies stream index 0 but is encoded in no mini-block. Step past
    // it once per page, block-scoped (advance_past_first_value is warp-scoped and is left alone
    // for the single-warp callers).
    if (current_value_idx == 0) {
      block.sync();
      if (block.thread_rank() == 0) { current_value_idx++; }
      block.sync();
      if (current_value_idx >= value_count) { return false; }
    }

    auto const w    = static_cast<uint32_t>(warp.meta_group_rank());
    auto const lane = static_cast<int>(warp.thread_rank());

    uint32_t const ppm             = values_per_mb / warp_size;  // passes per mini-block
    uint32_t const passes_in_block = block_size / warp_size;
    uint32_t const ordinal         = cur_mb * ppm + cur_pass;  // pass ordinal within the block
    uint32_t const active = min(static_cast<uint32_t>(num_warps_t), passes_in_block - ordinal);
    bool const is_active  = w < active;

    // Because `active` stops at the block boundary, `mb` is provably < mini_block_count, so
    // cur_bitwidths[mb] never reads past the bitwidth array.
    uint32_t const p   = ordinal + w;
    uint32_t const mb  = is_active ? p / ppm : cur_mb;
    uint32_t const sub = is_active ? p % ppm : 0;
    // Inactive warps must not land on a stream index, or they would write over the values an
    // active warp produced. value_count is past the end of the stream, so every caller's
    // `lane_idx < end` guard rejects them.
    lane_idx = is_active
                 ? current_value_idx + (mb - cur_mb) * values_per_mb + sub * warp_size + lane
                 : static_cast<uint32_t>(value_count);

    uint32_t const mb_bits  = cur_bitwidths[mb];
    uint8_t const* mb_start = cur_mb_start;
    for (uint32_t m = cur_mb; m < mb; ++m) {
      mb_start += cur_bitwidths[m] * values_per_mb / CHAR_BIT;
    }
    // position at the end of this pass's values; the unpack below uses negative indexes
    auto const d_start = mb_start + (sub + 1) * (warp_size * mb_bits / CHAR_BIT);

    // Unpack deltas. This mirrors calc_mini_block_pass() above; the two are kept separate because
    // that one is on the byte-for-byte-stable path used by the legacy kernels.
    // TODO: factor the shared body into a free device function once that constraint lifts.
    zigzag128_t delta = 0;
    if (is_active && lane_idx < value_count) {
      int32_t ofs      = (lane - warp_size) * mb_bits;
      uint8_t const* q = d_start + (ofs >> 3);
      ofs &= 7;
      if (q < block_end) {
        uint32_t c = CHAR_BIT - ofs;
        delta      = (*q++) >> ofs;

        while (c < mb_bits && q < block_end) {
          delta |= static_cast<zigzag128_t>(*q++) << c;
          c += CHAR_BIT;
        }
        delta &= (static_cast<zigzag128_t>(1) << mb_bits) - 1;
      }
    }
    delta += cur_min_delta;

    // Running sum within this warp's pass. Shuffle-based, so concurrent decoders on other warps
    // cannot alias; only the already-reduced per-warp totals below cross warps.
    zigzag128_t const local = cg::inclusive_scan(warp, delta, cg::plus<int64_t>{});
    if (lane == warp_size - 1) { warp_pass_total[w] = is_active ? local : zigzag128_t{0}; }

    // `active` is uniform across the block, so every thread reaches this barrier. It publishes the
    // warp totals and also orders every warp's reads of cur_mb_start / cur_mb / cur_bitwidths
    // above against thread 0's rewrite of them below.
    block.sync();

    zigzag128_t carry = last_value;
    for (uint32_t j = 0; j < w; ++j) {
      carry += warp_pass_total[j];
    }
    lane_value = local + carry;

    zigzag128_t block_total = last_value;
    for (uint32_t j = 0; j < active; ++j) {
      block_total += warp_pass_total[j];
    }

    if (block.thread_rank() == 0) {
      uint32_t next = cur_pass + active;
      // setup_next_mini_block carries the end-of-page freeze and the block transition; reusing it
      // keeps this path's cursor arithmetic identical to the single-warp one.
      while (next >= ppm) {
        next -= ppm;
        setup_next_mini_block(true);
      }
      cur_pass = next;
    }
    active_values = active * warp_size;

    block.sync();
    // `last_value` is the one piece of state read *after* the barrier above it (at `carry` and
    // `block_total`), so publishing it inside that thread-0 block would race a slower warp still
    // computing its carry -- it would pick up the next iteration's base and decode wrong values.
    // Deferring the write past this barrier orders it against those reads without costing another
    // one: every thread already computed the same `block_total`, the next call's leading
    // `block.sync()` orders this write against the next round of reads, and nothing between here
    // and there touches `last_value` (the wide path does not go through calc_mini_block_pass).
    if (block.thread_rank() == 0) { last_value = block_total; }
    return true;
  }
};

}  // namespace cudf::io::parquet::detail
