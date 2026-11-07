/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cuda/std/bit>

#include <cstdint>

namespace cudf::io::parquet::detail {

/**
 * @file
 * @brief Warp-wide parsing of PLAIN BYTE_ARRAY length prefixes
 *
 * A PLAIN BYTE_ARRAY page or dictionary holds `[4-byte little-endian length][bytes]` per entry, so
 * finding the entries is a pointer chase. The functions here let a warp parse a shared-memory
 * window of such data speculatively: lane 0 follows the chain from the known cursor and every
 * other lane from the first candidate prefix of its slice of the window, then pointer doubling over
 * the slices verifies that the chain from lane 0 crosses the window through links that start where
 * the previous one exits. When it does, the entries are exactly those a sequential parse would
 * find. When it does not, in binary data, for lengths of 64 KiB and more, or on a corrupt length,
 * the caller parses the window with one lane. The dictionary index (page_hdr.cu) and the string
 * offsets pre-pass of PLAIN data pages (page_string_decode.cu) share this code.
 */

/**
 * @brief Bytes of the window each lane parses. 516 / 4 == 1 (mod 32) keeps the lock-step reads of
 * the lanes on distinct shared memory banks.
 *
 * With the padding below a warp's window is 16,528 bytes; the dictionary kernel measured 8,192
 * chunks with 32 KiB dictionaries at 0.80 ms per launch against 0.82 ms with a 4 KiB window and
 * one parsing lane for 8-byte entries, 0.36 against 0.41 ms for 32-byte entries, and 0.23 against
 * 0.21 ms for 300-byte entries (RTX PRO 6000); a 260-byte slice lost up to 47% on 1 MiB pages of
 * short entries.
 */
constexpr int32_t length_prefix_slice_size = 516;

/// Bytes of page data a window holds: one slice per lane.
constexpr int32_t length_prefix_window_size = length_prefix_slice_size * cudf::detail::warp_size;

/**
 * @brief Slack after the window: a length prefix in its last bytes is read as two whole words, so
 * 4 bytes would do; 16 keeps every window aligned for int4 loads.
 */
constexpr int32_t length_prefix_window_pad = 16;

/**
 * @brief Mean entry length above which a slice holds too few entries for the speculative parse
 * to pay for its candidate scan; such data is parsed by one lane.
 *
 * Measured on 1 MiB dictionary pages, the two parses cross between mean lengths of 128 and 160
 * bytes (0.48 vs 0.63 ms at 128, 0.62 vs 0.54 ms at 160).
 */
constexpr int32_t length_prefix_max_mean_length_for_speculation = 128;

static_assert(length_prefix_slice_size % 4 == 0 and
                (length_prefix_slice_size / 4) % cudf::detail::warp_size == 1,
              "Slice size must be 4 * (32 * n + 1) bytes");
static_assert((length_prefix_window_size + length_prefix_window_pad) % sizeof(int4) == 0,
              "Each warp's window must stay aligned for int4 loads");

/// A warp of the block
using length_prefix_warp =
  cooperative_groups::thread_block_tile<cudf::detail::warp_size, cooperative_groups::thread_block>;

/**
 * @brief Page-relative offset at which the window holding `cursor` starts: aligned so that
 * `data + start` is a multiple of 16 bytes. Negative for the first window of a misaligned page;
 * the bytes before the page are zero in the window.
 */
__device__ inline int32_t length_prefix_window_start(uint8_t const* data, int32_t cursor)
{
  return cursor - static_cast<int32_t>(reinterpret_cast<uintptr_t>(data + cursor) % sizeof(int4));
}

/**
 * @brief Loads a window of page data into shared memory with the whole warp.
 *
 * Aligned 16-byte loads, eight per lane in flight; the page tail is copied byte-wise and bytes
 * before the page start are zero. The caller synchronizes the warp afterwards.
 *
 * @param warp The warp
 * @param data Page data
 * @param data_size Page data size in bytes
 * @param start Page-relative offset of the window, from length_prefix_window_start()
 * @param window_bytes Bytes to load, at most length_prefix_window_size and data_size - start
 * @param window Shared memory of length_prefix_window_size + length_prefix_window_pad bytes,
 * 16-byte aligned
 */
__device__ inline void load_length_prefix_window(length_prefix_warp const& warp,
                                                 uint8_t const* data,
                                                 int32_t data_size,
                                                 int32_t start,
                                                 int32_t window_bytes,
                                                 uint8_t* window)
{
  constexpr int32_t load_size = sizeof(int4);
  // Vector loads each lane issues before storing any of them, to keep several in flight: eight
  // instead of four took the dictionary kernel's 32-byte-entry case from 0.40 to 0.36 ms per
  // launch.
  constexpr int32_t loads_per_lane = 8;
  constexpr int32_t load_stride    = cudf::detail::warp_size * load_size;
  int32_t const lane               = warp.thread_rank();
  for (int32_t base = lane * load_size; base < window_bytes; base += load_stride * loads_per_lane) {
    int4 values[loads_per_lane];
#pragma unroll
    for (int32_t u = 0; u < loads_per_lane; ++u) {
      int32_t const i = base + u * load_stride;
      // Offset of the load from the start of the page
      int64_t const source = static_cast<int64_t>(start) + i;
      if (source >= 0 and source + load_size <= data_size and i + load_size <= window_bytes) {
        values[u] = *reinterpret_cast<int4 const*>(data + source);
      }
    }
#pragma unroll
    for (int32_t u = 0; u < loads_per_lane; ++u) {
      int32_t const i = base + u * load_stride;
      if (i >= window_bytes) { break; }
      int64_t const source = static_cast<int64_t>(start) + i;
      if (source >= 0 and source + load_size <= data_size and i + load_size <= window_bytes) {
        reinterpret_cast<int4*>(window)[i / load_size] = values[u];
      } else {  // Byte-wise copy for the page tail; zero-pad bytes before the page start
        for (int32_t j = 0; j < load_size and i + j < window_bytes; ++j) {
          window[i + j] = source + j >= 0 ? data[source + j] : 0;
        }
      }
    }
  }
}

/**
 * @brief Reads the little-endian length prefix at window byte offset `offset`.
 *
 * Reads the word after the prefix, up to 3 bytes past it; the window padding covers that.
 */
__device__ inline int32_t read_length_prefix(uint32_t const* window, int32_t offset)
{
  uint32_t const lo = window[offset >> 2];
  uint32_t const hi = window[(offset >> 2) + 1];
  return static_cast<int32_t>(__funnelshift_r(lo, hi, (offset & 3) * 8));
}

/**
 * @brief Returns a mask with bit b set when byte b of `word` is zero.
 *
 * `__vcmpeq4` yields 0xFF for each zero byte; bit 8b of that lands in bit b.
 */
__device__ inline uint32_t zero_byte_mask(uint32_t word)
{
  uint32_t const r = __vcmpeq4(word, 0u);
  return (r & 1u) | ((r >> 7) & 2u) | ((r >> 14) & 4u) | ((r >> 21) & 8u);
}

/**
 * @brief Finds the first candidate length prefix in a slice of the window.
 *
 * A candidate at offset p has zero bytes at p + 2 and p + 3 and a non-zero byte at p + 4 (bytes
 * past the window count as non-zero): a length below 64 KiB that ends a run of zero bytes. Text
 * has no zero bytes outside its length prefixes, so in text the first candidate of a slice is a
 * real prefix with few exceptions; the speculative parse verifies every candidate it uses.
 *
 * @param window The window as 32-bit words
 * @param lo First offset to consider, a multiple of 4
 * @param hi One past the last offset to consider
 * @param window_bytes Active window size in bytes
 * @return Offset of the first candidate with all 4 prefix bytes inside the window, or -1
 */
__device__ inline int32_t first_candidate_prefix(uint32_t const* window,
                                                 int32_t lo,
                                                 int32_t hi,
                                                 int32_t window_bytes)
{
  int32_t const limit = min(hi, window_bytes - 3);
  if (lo >= limit) { return -1; }
  int32_t const first_word = lo >> 2;
  uint32_t zero_flags      = zero_byte_mask(window[first_word]);
  for (int32_t word = first_word; 4 * word - 2 < limit; ++word) {
    uint32_t const next_flags = zero_byte_mask(window[word + 1]);
    // Zero flags of the 8 bytes starting at byte 4 * word
    uint32_t const flags = zero_flags | (next_flags << 4);
    int32_t candidate    = -1;
    // A prefix starting at byte 4 * word + b has its bytes +2 and +3 at flags bits b + 2 and
    // b + 3 and the byte after it at bit b + 4. Starts before the word were covered by the
    // previous iteration, unless this is the first word of the slice.
#pragma unroll
    for (int32_t b = -2; b <= 1; ++b) {
      if (b < 0 && word == first_word) { continue; }
      bool const zero_prefix_tail = ((flags >> (b + 2)) & 3u) == 3u;
      bool const data_follows     = !((flags >> (b + 4)) & 1u) || 4 * word + b + 4 >= window_bytes;
      if (zero_prefix_tail && data_follows) {
        candidate = 4 * word + b;
        break;
      }
    }
    if (candidate >= 0 && candidate < limit) { return candidate; }
    zero_flags = next_flags;
  }
  return -1;
}

/**
 * @brief Per-lane result of a verified speculative parse of one window.
 */
struct window_chain {
  int32_t start;      ///< Window offset of this lane's first entry, or -1 when it has none.
  int32_t count;      ///< Entries of this lane; 0 when its slice is not on the chain.
  int32_t preceding;  ///< Chain entries ahead of this lane's.
  int32_t total;      ///< Entries of the whole chain.
  int32_t chain_end;  ///< Window offset of the first length prefix the window does not hold.
};

/**
 * @brief Parses one window with all lanes at once and verifies the result.
 *
 * Every lane follows the entry chain inside its slice: lane 0 from the known cursor, the others
 * from the first candidate prefix of their slice. A lane's chain is valid when it leaves the
 * window or exits exactly where the slice it lands in started its own chain, so pointer doubling
 * over the slices tells whether the chain from lane 0 is made of valid links all the way to the
 * end of the window. If so, the entries are exactly those a sequential parse would find.
 *
 * @param warp The warp
 * @param window The window as 32-bit words
 * @param window_start Page-relative byte offset of the window
 * @param window_bytes Active window size in bytes
 * @param data_size Page data size in bytes
 * @param cursor_offset Window offset of the next length prefix; its 4 bytes lie in the window and
 * in the page
 * @param slice_starts Shared scratch, one entry per lane
 * @param[out] chain The result, valid when true is returned
 * @return False when the window has to be parsed by one lane: a chain met a negative or
 * oversized length, or a candidate was not a prefix
 */
__device__ inline bool speculative_parse_window(length_prefix_warp const& warp,
                                                uint32_t const* window,
                                                int32_t window_start,
                                                int32_t window_bytes,
                                                int32_t data_size,
                                                int32_t cursor_offset,
                                                int32_t* slice_starts,
                                                window_chain& chain)
{
  constexpr int32_t warp_size = cudf::detail::warp_size;
  int32_t const lane          = warp.thread_rank();
  int32_t const slice_begin   = lane * length_prefix_slice_size;
  int32_t const slice_end     = min(slice_begin + length_prefix_slice_size, window_bytes);
  int32_t start               = -1;
  if (lane == 0) {
    start = cursor_offset;
  } else if (slice_begin < window_bytes) {
    start = first_candidate_prefix(window, slice_begin, slice_end, window_bytes);
  }
  int32_t count     = 0;
  int32_t chain_end = start;
  bool invalid      = false;
  if (start >= 0) {
    int32_t offset = start;
    while (offset < slice_end) {
      if (offset + 4 > window_bytes) { break; }  // The next prefix is not in this window
      int32_t const length = read_length_prefix(window, offset);
      // Negative, or the string runs past the page. The bound is taken from the page-relative
      // position, which lies in [0, data_size - 4], so it cannot overflow.
      if (static_cast<uint32_t>(length) >
          static_cast<uint32_t>(data_size - (window_start + offset) - 4)) {
        invalid = true;
        break;
      }
      offset += 4 + length;
      ++count;
    }
    chain_end = offset;
  }
  if (warp.any(invalid)) { return false; }
  slice_starts[lane] = start;
  warp.sync();
  bool const terminal = start < 0 || chain_end + 4 > window_bytes;
  int32_t const next  = terminal ? warp_size : chain_end / length_prefix_slice_size;
  bool const linked   = terminal || slice_starts[next] == chain_end;

  // Pointer doubling from lane 0 over the slice links: each round doubles the hops a lane knows
  // about, so log2(warp_size) rounds cover a chain through all the slices.
  constexpr int32_t pointer_doubling_rounds =
    cuda::std::bit_width(static_cast<uint32_t>(warp_size - 1));
  uint32_t visited  = 1u << lane;
  int32_t successor = (linked && !terminal) ? next : warp_size;
  bool valid        = linked;
#pragma unroll
  for (int32_t round = 0; round < pointer_doubling_rounds; ++round) {
    int32_t const source           = successor < warp_size ? successor : lane;
    int32_t const next_successor   = warp.shfl(successor, source);
    uint32_t const visited_by_next = warp.shfl(visited, source);
    bool const next_valid          = warp.shfl(valid, source);
    if (successor < warp_size) {
      visited |= visited_by_next;
      valid     = valid && next_valid;
      successor = next_successor;
    }
  }
  uint32_t const on_chain = warp.shfl(visited, 0);
  if (!warp.shfl(valid, 0)) { return false; }

  int32_t const mine = ((on_chain >> lane) & 1u) ? count : 0;
  int32_t const inclusive =
    cooperative_groups::inclusive_scan(warp, mine, cooperative_groups::plus<int32_t>{});
  chain.start     = mine > 0 ? start : -1;
  chain.count     = mine;
  chain.preceding = inclusive - mine;
  chain.total     = warp.shfl(inclusive, warp_size - 1);
  chain.chain_end = warp.shfl(chain_end, warp_size - 1 - __clz(on_chain));
  return true;
}

}  // namespace cudf::io::parquet::detail
