/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpuinflate.hpp"
#include "io/utilities/block_utils.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/cub.cuh>

namespace cudf::io::detail {
constexpr int32_t batch_size    = (1 << 5);
constexpr int32_t batch_count   = (1 << 2);
constexpr int32_t prefetch_size = (1 << 9);  // 512B, in 32B chunks

void __device__ busy_wait(size_t cycles)
{
  clock_t start = clock();
  for (;;) {
    clock_t const now     = clock();
    clock_t const elapsed = now > start ? now - start : now + (0xffff'ffff - start);
    if (elapsed >= cycles) return;
  }
}

/**
 * @brief Describes a single LZ77 symbol (single entry in batch)
 */
struct unsnap_batch_s {
  int32_t len;  // 1..64 = Number of bytes
  uint32_t
    offset;  // copy distance if greater than zero or negative of literal offset in byte stream
};

/**
 * @brief Queue structure used to exchange data between warps
 */
struct unsnap_queue_s {
  unsnap_queue_s() = default;  // required to compile on ctk-12.2 + aarch64

  uint32_t prefetch_wrpos;         ///< Prefetcher write position
  uint32_t prefetch_rdpos;         ///< Prefetch consumer read position
  int32_t prefetch_end;            ///< Prefetch enable flag (nonzero stops prefetcher)
  int32_t batch_len[batch_count];  ///< Length of each batch - <0:end, 0:not ready, >0:symbol count
  unsnap_batch_s batch[batch_count * batch_size];  ///< LZ77 batch data
  uint8_t buf[prefetch_size];                      ///< Prefetch buffer
};

/**
 * @brief snappy decompression state
 */
struct unsnap_state_s {
  CUDF_HOST_DEVICE constexpr unsnap_state_s() noexcept {
  }  // required to compile on ctk-12.2 + aarch64

  uint8_t const* base{};           ///< base ptr of compressed stream
  uint8_t const* end{};            ///< end of compressed stream
  uint32_t uncompressed_size{};    ///< uncompressed stream size
  uint32_t bytes_left{};           ///< remaining bytes to decompress
  int32_t error{};                 ///< current error status
  uint32_t tstart{};               ///< start time for perf logging
  unsnap_queue_s volatile q{};     ///< queue for cross-warp communication
  device_span<uint8_t const> src;  ///< input for current block
  device_span<uint8_t> dst;        ///< output for current block
};

inline __device__ volatile uint8_t& byte_access(unsnap_state_s* s, uint32_t pos)
{
  return s->q.buf[pos & (prefetch_size - 1)];
}

/**
 * @brief prefetches data for the symbol decoding stage
 *
 * @param s decompression state
 * @param t warp lane id
 */
__device__ void snappy_prefetch_bytestream(unsnap_state_s* s, int t)
{
  uint8_t const* base = s->base;
  auto end            = (uint32_t)(s->end - base);
  auto align_bytes    = (uint32_t)(0x20 - (0x1f & reinterpret_cast<uintptr_t>(base)));
  int32_t pos         = min(align_bytes, end);
  int32_t blen;
  // Start by prefetching up to the next a 32B-aligned location
  if (t < pos) { s->q.buf[t] = base[t]; }
  blen = 0;
  do {
    __syncwarp();
    if (!t) {
      uint32_t minrdpos;
      s->q.prefetch_wrpos = pos;
      minrdpos            = pos - min(pos, prefetch_size - 32u);
      blen                = (int)min(32u, end - pos);
      for (;;) {
        uint32_t rdpos = s->q.prefetch_rdpos;
        if (rdpos >= minrdpos) break;
        if (s->q.prefetch_end) {
          blen = 0;
          break;
        }
        busy_wait(20);
      }
    }
    blen = shuffle(blen);
    if (t < blen) { byte_access(s, pos + t) = base[pos + t]; }
    pos += blen;
  } while (blen > 0);
}

/**
 * @brief Lookup table for get_len3_mask()
 *
 * Indexed by a 10-bit pattern, contains the corresponding 4-bit mask of
 * 3-byte code lengths in the lower 4 bits, along with the total number of
 * bytes used for coding the four lengths in the upper 4 bits.
 * The upper 4-bit value could also be obtained by 8+__popc(mask4)
 *
 *   for (uint32_t k = 0; k < 1024; k++)
 *   {
 *       for (uint32_t i = 0, v = 0, b = k, n = 0; i < 4; i++)
 *       {
 *           v |= (b & 1) << i;
 *           n += (b & 1) + 2;
 *           b >>= (b & 1) + 2;
 *       }
 *       k_len3lut[k] = v | (n << 4);
 *   }
 */
static uint8_t const __device__ __constant__ k_len3lut[1 << 10] = {
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xa3, 0x80, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xa3, 0x94, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xa3, 0x94, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xb7, 0x98, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xb7, 0x94, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xb7, 0x98, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xb7, 0x94, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xa3, 0x80, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xa3, 0xac, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xa3, 0xac, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xb7, 0x98, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xb7, 0xac, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xb7, 0x98, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xb7, 0xac, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xbb, 0x80, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xbb, 0x94, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xbb, 0x94, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xb7, 0x98, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xb7, 0x94, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xb7, 0x98, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xb7, 0x94, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xbb, 0x80, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xbb, 0xac, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xbb, 0xac, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xb7, 0x98, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xb7, 0xac, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xb7, 0x98, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xb7, 0xac, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xa3, 0x80, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xa3, 0x94, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xa3, 0x94, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xcf, 0x98, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xcf, 0x94, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xcf, 0x98, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xcf, 0x94, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xa3, 0x80, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xa3, 0xac, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xa3, 0xac, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xcf, 0x98, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xcf, 0xac, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xcf, 0x98, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xcf, 0xac, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xbb, 0x80, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xbb, 0x94, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xbb, 0x94, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xcf, 0x98, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xcf, 0x94, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xcf, 0x98, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xcf, 0x94, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xbb, 0x80, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xbb, 0xac, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xbb, 0xac, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xcf, 0x98, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xcf, 0xac, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xcf, 0x98, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xcf, 0xac, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf};

/**
 * @brief Returns a 32-bit mask where 1 means 3-byte code length and 0 means 2-byte
 * code length, given an input mask of up to 96 bits.
 *
 * Implemented by doing 8 consecutive lookups, building the result 4-bit at a time
 */
inline __device__ uint32_t get_len3_mask(uint32_t v0, uint32_t v1, uint32_t v2)
{
  uint32_t m, v, m4, n;
  v  = v0;
  m4 = k_len3lut[v & 0x3ff];
  m  = m4 & 0xf;
  n  = m4 >> 4;  // 8..12
  v  = v0 >> n;
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 4;
  n += m4 >> 4;  // 16..24
  v  = __funnelshift_r(v0, v1, n);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 8;
  n += m4 >> 4;  // 24..36
  v >>= (m4 >> 4);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 12;
  n  = (n + (m4 >> 4)) & 0x1f;  // (32..48) % 32 = 0..16
  v1 = __funnelshift_r(v1, v2, n);
  v2 >>= n;
  v  = v1;
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 16;
  n  = m4 >> 4;  // 8..12
  v  = v1 >> n;
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 20;
  n += m4 >> 4;  // 16..24
  v  = __funnelshift_r(v1, v2, n);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 24;
  n += m4 >> 4;  // 24..36
  v >>= (m4 >> 4);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 28;
  return m;
}

/**
 * @brief Returns a 32-bit mask where each 2-bit pair contains the symbol length
 * minus 2, given two input masks each containing bit0 or bit1 of the corresponding
 * code length minus 2 for up to 32 bytes
 */
inline __device__ uint32_t get_len5_mask(uint32_t v0, uint32_t v1)
{
  uint32_t m;
  m = (v1 & 1) * 2 + (v0 & 1);
  v0 >>= (m + 2);
  v1 >>= (m + 1);
  for (uint32_t i = 1; i < 16; i++) {
    uint32_t m2 = (v1 & 2) | (v0 & 1);
    uint32_t n  = m2 + 2;
    m |= m2 << (i * 2);
    v0 >>= n;
    v1 >>= n;
  }
  return m;
}

/**
 * @brief decode symbols and output LZ77 batches (single-warp)
 *
 * @param s decompression state
 * @param t warp lane id
 */
__device__ void snappy_decode_symbols(unsnap_state_s* s, uint32_t t)
{
  uint32_t cur        = 0;
  auto end            = static_cast<uint32_t>(s->end - s->base);
  uint32_t bytes_left = s->uncompressed_size;
  uint32_t dst_pos    = 0;
  int32_t batch       = 0;

  for (;;) {
    int32_t batch_len;
    unsnap_batch_s volatile* b;

    // Wait for prefetcher
    if (t == 0) {
      s->q.prefetch_rdpos = cur;
#pragma unroll(1)  // We don't want unrolling here
      while (s->q.prefetch_wrpos < min(cur + 5 * batch_size, end)) {
        busy_wait(10);
      }
      b = &s->q.batch[batch * batch_size];
    }
    // Process small symbols in parallel: for data that does not get good compression,
    // the stream will consist of a large number of short literals (1-byte or 2-byte)
    // followed by short repeat runs. This results in many 2-byte or 3-byte symbols
    // that can all be decoded in parallel once we know the symbol length.
    {
      uint32_t v0, v1, v2, len3_mask, cur_t, is_long_sym, short_sym_mask;
      uint32_t b0;
      cur            = shuffle(cur);
      cur_t          = cur + t;
      b0             = byte_access(s, cur_t);
      v0             = ballot((b0 == 4) || (b0 & 2));
      b0             = byte_access(s, cur_t + 32);
      v1             = ballot((b0 == 4) || (b0 & 2));
      b0             = byte_access(s, cur_t + 64);
      v2             = ballot((b0 == 4) || (b0 & 2));
      len3_mask      = shuffle((t == 0) ? get_len3_mask(v0, v1, v2) : 0);
      cur_t          = cur + 2 * t + __popc(len3_mask & ((1 << t) - 1));
      b0             = byte_access(s, cur_t);
      is_long_sym    = ((b0 & ~4) != 0) && (((b0 + 1) & 2) == 0);
      short_sym_mask = ballot(is_long_sym);
      batch_len      = 0;
      b = reinterpret_cast<unsnap_batch_s volatile*>(shuffle(reinterpret_cast<uintptr_t>(b)));
      if (!(short_sym_mask & 1)) {
        batch_len = shuffle((t == 0) ? (short_sym_mask) ? __ffs(short_sym_mask) - 1 : 32 : 0);
        if (batch_len != 0) {
          uint32_t blen = 0;
          int32_t ofs   = 0;
          // Check for bad copy element (offset == 0).
          bool is_copy_offset_bad = false;
          if (t < batch_len) {
            blen               = (b0 & 1) ? ((b0 >> 2) & 7) + 4 : ((b0 >> 2) + 1);
            ofs                = (b0 & 1) ? ((b0 & 0xe0) << 3) | byte_access(s, cur_t + 1)
                                 : (b0 & 2) ? byte_access(s, cur_t + 1) | (byte_access(s, cur_t + 2) << 8)
                                            : -(int32_t)(cur_t + 1);
            is_copy_offset_bad = (b0 & 3) != 0 && ofs == 0;
            b[t].len           = blen;
            b[t].offset        = ofs;
            ofs += blen;  // for correct out-of-range detection below
          }
          blen       = warp_reduce_pos<cudf::detail::warp_size>(blen, t);
          bytes_left = shuffle(bytes_left);
          dst_pos    = shuffle(dst_pos);
          // If `is_copy_offset_bad`, truncate the batch at the first lane with a malformed copy
          // element (offset == 0) so warp 2 never runs it; the scalar slow path below then re-reads
          // it, breaks, and the kernel reports FAILURE.
          short_sym_mask = __ffs(
            ballot(blen > bytes_left || ofs > (int32_t)(dst_pos + blen) || is_copy_offset_bad));
          if (short_sym_mask != 0) { batch_len = min(batch_len, short_sym_mask - 1); }
          if (batch_len != 0) {
            blen = shuffle(blen, batch_len - 1);
            cur  = shuffle(cur_t, batch_len - 1) + 2 + ((len3_mask >> (batch_len - 1)) & 1);
            if (t == 0) {
              dst_pos += blen;
              bytes_left -= blen;
            }
          }
        }
      }
      // Check if the batch was stopped by a 3-byte or 4-byte literal
      if (batch_len < batch_size - 2 && shuffle(b0 & ~4, batch_len) == 8) {
        // If so, run a slower version of the above that can also handle 3/4-byte literal sequences
        uint32_t batch_add;
        do {
          uint32_t clen, mask_t;
          cur_t     = cur + t;
          b0        = byte_access(s, cur_t);
          clen      = (b0 & 3) ? (b0 & 2) ? 1 : 0 : (b0 >> 2);  // symbol length minus 2
          v0        = ballot(clen & 1);
          v1        = ballot((clen >> 1) & 1);
          len3_mask = shuffle((t == 0) ? get_len5_mask(v0, v1) : 0);
          mask_t    = (1 << (2 * t)) - 1;
          cur_t     = cur + 2 * t + 2 * __popc((len3_mask & 0xaaaa'aaaa) & mask_t) +
                  __popc((len3_mask & 0x5555'5555) & mask_t);
          b0          = byte_access(s, cur_t);
          is_long_sym = ((b0 & 3) ? ((b0 & 3) == 3) : (b0 > 3 * 4)) || (cur_t >= cur + 32) ||
                        (batch_len + t >= batch_size);
          batch_add = __ffs(ballot(is_long_sym)) - 1;
          if (batch_add != 0) {
            uint32_t blen           = 0;
            int32_t ofs             = 0;
            bool is_copy_offset_bad = false;
            if (t < batch_add) {
              blen                    = (b0 & 1) ? ((b0 >> 2) & 7) + 4 : ((b0 >> 2) + 1);
              ofs                     = (b0 & 1) ? ((b0 & 0xe0) << 3) | byte_access(s, cur_t + 1)
                                        : (b0 & 2) ? byte_access(s, cur_t + 1) | (byte_access(s, cur_t + 2) << 8)
                                                   : -(int32_t)(cur_t + 1);
              is_copy_offset_bad      = (b0 & 3) != 0 && ofs == 0;
              b[batch_len + t].len    = blen;
              b[batch_len + t].offset = ofs;
              ofs += blen;  // for correct out-of-range detection below
            }
            blen       = warp_reduce_pos<cudf::detail::warp_size>(blen, t);
            bytes_left = shuffle(bytes_left);
            dst_pos    = shuffle(dst_pos);
            // See comment in the fast-path block above for `is_copy_offset_bad`.
            short_sym_mask = __ffs(
              ballot(blen > bytes_left || ofs > (int32_t)(dst_pos + blen) || is_copy_offset_bad));
            if (short_sym_mask != 0) { batch_add = min(batch_add, short_sym_mask - 1); }
            if (batch_add != 0) {
              blen = shuffle(blen, batch_add - 1);
              cur  = shuffle(cur_t, batch_add - 1) + 2 + ((len3_mask >> ((batch_add - 1) * 2)) & 3);
              if (t == 0) {
                dst_pos += blen;
                bytes_left -= blen;
              }
              batch_len += batch_add;
            }
          }
        } while (batch_add >= 6 && batch_len < batch_size - 2);
      }
    }
    if (t == 0) {
      while (bytes_left > 0 && batch_len < batch_size) {
        uint32_t blen, offset;
        uint8_t b0 = byte_access(s, cur);
        if (b0 & 3) {
          uint8_t b1 = byte_access(s, cur + 1);
          if (!(b0 & 2)) {
            // xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
            offset = ((b0 & 0xe0) << 3) | b1;
            blen   = ((b0 >> 2) & 7) + 4;
            cur += 2;
          } else {
            // xxxxxx1x: copy with 6-bit length, 2-byte or 4-byte offset
            offset = b1 | (byte_access(s, cur + 2) << 8);
            if (b0 & 1)  // 4-byte offset
            {
              offset |= (byte_access(s, cur + 3) << 16) | (byte_access(s, cur + 4) << 24);
              cur += 5;
            } else {
              cur += 3;
            }
            blen = (b0 >> 2) + 1;
          }
          dst_pos += blen;
          if (offset - 1u >= dst_pos || bytes_left < blen) break;
          bytes_left -= blen;
        } else if (b0 < 4 * 4) {
          // 0000xx00: short literal
          blen   = (b0 >> 2) + 1;
          offset = -(int32_t)(cur + 1);
          cur += 1 + blen;
          dst_pos += blen;
          if (bytes_left < blen) break;
          bytes_left -= blen;
        } else {
          // xxxxxx00: literal
          blen = b0 >> 2;
          if (blen >= 60) {
            uint32_t num_bytes = blen - 59;
            blen               = byte_access(s, cur + 1);
            if (num_bytes > 1) {
              blen |= byte_access(s, cur + 2) << 8;
              if (num_bytes > 2) {
                blen |= byte_access(s, cur + 3) << 16;
                if (num_bytes > 3) { blen |= byte_access(s, cur + 4) << 24; }
              }
            }
            cur += num_bytes;
          }
          cur += 1;
          blen += 1;
          offset = -(int32_t)cur;
          cur += blen;
          // Wait for prefetcher
          s->q.prefetch_rdpos = cur;
#pragma unroll(1)  // We don't want unrolling here
          while (s->q.prefetch_wrpos < min(cur + 5 * batch_size, end)) {
            busy_wait(10);
          }
          dst_pos += blen;
          if (bytes_left < blen) break;
          bytes_left -= blen;
        }
        b[batch_len].len    = blen;
        b[batch_len].offset = offset;
        batch_len++;
      }
      if (batch_len != 0) {
        s->q.batch_len[batch] = batch_len;
        batch                 = (batch + 1) & (batch_count - 1);
      }
    }
    batch_len = shuffle(batch_len);
    if (t == 0) {
      while (s->q.batch_len[batch] != 0) {
        busy_wait(20);
      }
    }
    if (batch_len != batch_size) { break; }
  }
  if (!t) {
    s->q.prefetch_end     = 1;
    s->q.batch_len[batch] = -1;
    s->bytes_left         = bytes_left;
    if (bytes_left != 0) { s->error = -2; }
  }
}

/**
 * @brief process LZ77 symbols and output uncompressed stream
 *
 * @param s decompression state
 * @param t thread id within participating group (lane id)
 * @param temp_storage temporary storage used by the algorithm
 *
 * NOTE: No error checks at this stage (WARP0 responsible for not sending offsets and lengths that
 *would result in out-of-bounds accesses)
 */
template <typename Storage>
__device__ void snappy_process_symbols(unsnap_state_s* s, int t, Storage& temp_storage)
{
  auto const literal_base = s->base;
  auto out                = s->dst.data();
  int batch               = 0;

  do {
    unsnap_batch_s volatile* b = &s->q.batch[batch * batch_size];
    int32_t batch_len, blen_t, dist_t;

    if (t == 0) {
      while ((batch_len = s->q.batch_len[batch]) == 0) {
        busy_wait(20);
      }
    } else {
      batch_len = 0;
    }
    batch_len = shuffle(batch_len);
    if (batch_len <= 0) { break; }
    if (t < batch_len) {
      blen_t = b[t].len;
      dist_t = b[t].offset;
    } else {
      blen_t = dist_t = 0;
    }
    // Try to combine as many small entries as possible, but try to avoid doing that
    // if we see a small repeat distance 8 bytes or less
    if (shuffle(min((uint32_t)dist_t, (uint32_t)shuffle_xor(dist_t, 1))) > 8) {
      uint32_t n;
      do {
        uint32_t bofs      = warp_reduce_pos<cudf::detail::warp_size>(blen_t, t);
        uint32_t stop_mask = ballot((uint32_t)dist_t < bofs);
        uint32_t start_mask =
          cub::WarpReduce<uint32_t>(temp_storage).Sum((bofs < 32 && t < batch_len) ? 1 << bofs : 0);
        start_mask = shuffle(start_mask);
        n          = min(min((uint32_t)__popc(start_mask), (uint32_t)(__ffs(stop_mask) - 1u)),
                (uint32_t)batch_len);
        if (n != 0) {
          uint32_t it  = __popc(start_mask & ((2 << t) - 1));
          uint32_t tr  = t - shuffle(bofs - blen_t, it);
          int32_t dist = shuffle(dist_t, it);
          if (it < n) {
            uint8_t const* src = (dist > 0) ? (out + t - dist) : (literal_base + tr - dist);
            out[t]             = *src;
          }
          out += shuffle(bofs, n - 1);
          blen_t = shuffle(blen_t, (n + t) & 0x1f);
          dist_t = shuffle(dist_t, (n + t) & 0x1f);
          batch_len -= n;
        }
      } while (n >= 4);
    }
    for (int i = 0; i < batch_len; i++) {
      int32_t blen  = shuffle(blen_t, i);
      int32_t dist  = shuffle(dist_t, i);
      int32_t blen2 = (i + 1 < batch_len) ? shuffle(blen_t, i + 1) : 32;
      // Try to combine consecutive small entries if they are independent
      if ((uint32_t)dist >= (uint32_t)blen && blen + blen2 <= 32) {
        int32_t dist2 = shuffle(dist_t, i + 1);
        if ((uint32_t)dist2 >= (uint32_t)(blen + blen2)) {
          int32_t d;
          if (t < blen) {
            d = dist;
          } else {
            dist = dist2;
            d    = (dist2 <= 0) ? dist2 + blen : dist2;
          }
          blen += blen2;
          if (t < blen) {
            uint8_t const* src = (dist > 0) ? (out - d) : (literal_base - d);
            out[t]             = src[t];
          }
          out += blen;
          i++;
          continue;
        }
      }
      if (dist > 0) {
        // Copy
        uint8_t b0, b1;
        if (t < blen) {
          uint32_t pos       = t;
          uint8_t const* src = out + ((pos >= dist) ? (pos % dist) : pos) - dist;
          b0                 = *src;
        }
        if (32 + t < blen) {
          uint32_t pos       = 32 + t;
          uint8_t const* src = out + ((pos >= dist) ? (pos % dist) : pos) - dist;
          b1                 = *src;
        }
        if (t < blen) { out[t] = b0; }
        if (32 + t < blen) { out[32 + t] = b1; }
      } else {
        // Literal
        uint8_t b0, b1;
        dist = -dist;
        while (blen >= 64) {
          b0          = literal_base[dist + t];
          b1          = literal_base[dist + 32 + t];
          out[t]      = b0;
          out[32 + t] = b1;
          dist += 64;
          out += 64;
          blen -= 64;
        }
        if (t < blen) { b0 = literal_base[dist + t]; }
        if (32 + t < blen) { b1 = literal_base[dist + 32 + t]; }
        if (t < blen) { out[t] = b0; }
        if (32 + t < blen) { out[32 + t] = b1; }
      }
      out += blen;
    }
    __syncwarp();
    if (t == 0) { s->q.batch_len[batch] = 0; }
    batch = (batch + 1) & (batch_count - 1);
  } while (true);
}

/**
 * @brief Snappy decompression kernel
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * blockDim {128,1,1}
 *
 * @param[in] inputs Source & destination information per block
 * @param[out] outputs Decompression status per block
 */
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  unsnap_kernel_no_racecheck(device_span<device_span<uint8_t const> const> inputs,
                             device_span<device_span<uint8_t> const> outputs,
                             device_span<codec_exec_result> results)
{
  __shared__ __align__(16) unsnap_state_s state_g;
  __shared__ cub::WarpReduce<uint32_t>::TempStorage temp_storage;
  int t             = threadIdx.x;
  unsnap_state_s* s = &state_g;
  int strm_id       = blockIdx.x;

  if (t < batch_count) { s->q.batch_len[t] = 0; }
  __syncthreads();
  if (!t) {
    s->src         = inputs[strm_id];
    s->dst         = outputs[strm_id];
    auto cur       = s->src.data();
    auto const end = s->src.data() + s->src.size();
    s->error       = 0;
    if (cur < end) {
      // Read uncompressed size (varint), limited to 32-bit
      uint32_t uncompressed_size = *cur++;
      if (uncompressed_size > 0x7f) {
        uint32_t c        = (cur < end) ? *cur++ : 0;
        uncompressed_size = (uncompressed_size & 0x7f) | (c << 7);
        if (uncompressed_size >= (0x80 << 7)) {
          c                 = (cur < end) ? *cur++ : 0;
          uncompressed_size = (uncompressed_size & ((0x7f << 7) | 0x7f)) | (c << 14);
          if (uncompressed_size >= (0x80 << 14)) {
            c = (cur < end) ? *cur++ : 0;
            uncompressed_size =
              (uncompressed_size & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 21);
            if (uncompressed_size >= (0x80 << 21)) {
              c = (cur < end) ? *cur++ : 0;
              if (c < 0x8)
                uncompressed_size =
                  (uncompressed_size & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) |
                  (c << 28);
              else
                s->error = -1;
            }
          }
        }
      }
      s->uncompressed_size = uncompressed_size;
      s->bytes_left        = uncompressed_size;
      s->base              = cur;
      s->end               = end;
      if ((cur >= end && uncompressed_size != 0) || (uncompressed_size > s->dst.size())) {
        s->error = -1;
      }
    } else {
      s->error = -1;
    }
    s->q.prefetch_end   = 0;
    s->q.prefetch_wrpos = 0;
    s->q.prefetch_rdpos = 0;
  }
  __syncthreads();
  if (!s->error) {
    if (t < 32) {
      // WARP0: decode lengths and offsets
      snappy_decode_symbols(s, t);
    } else if (t < 64) {
      // WARP1: prefetch byte stream for WARP0
      snappy_prefetch_bytestream(s, t & 0x1f);
    } else if (t < 96) {
      // WARP2: LZ77
      snappy_process_symbols(s, t & 0x1f, temp_storage);
    }
    __syncthreads();
  }
  if (!t) {
    results[strm_id].bytes_written = s->uncompressed_size - s->bytes_left;
    results[strm_id].status = (s->error == 0) ? codec_status::SUCCESS : codec_status::FAILURE;
  }
}

void gpu_unsnap(device_span<device_span<uint8_t const> const> inputs,
                device_span<device_span<uint8_t> const> outputs,
                device_span<codec_exec_result> results,
                rmm::cuda_stream_view stream)
{
  if (inputs.empty()) { return; }

  dim3 dim_block(128, 1);           // 4 warps per stream, 1 stream per block
  dim3 dim_grid(inputs.size(), 1);  // TODO: Check max grid dimensions vs max expected count

  unsnap_kernel_no_racecheck<128>
    <<<dim_grid, dim_block, 0, stream.value()>>>(inputs, outputs, results);
  CUDF_CUDA_TRY(cudaGetLastError());
}

CUDF_KERNEL void get_snappy_uncompressed_size_kernel(
  device_span<device_span<uint8_t const> const> inputs, device_span<size_t> uncompressed_sizes)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= inputs.size()) return;

  auto cur       = inputs[idx].data();
  auto const end = inputs[idx].data() + inputs[idx].size();

  constexpr int payload_bits_per_byte = 7;
  constexpr size_t payload_mask       = (1 << payload_bits_per_byte) - 1;
  size_t uncompressed_size            = 0;
  int shift                           = 0;
  while (cur < end && shift + payload_bits_per_byte <= 8 * sizeof(size_t)) {
    size_t const byte = *cur++;
    uncompressed_size |= (byte & payload_mask) << shift;
    if ((byte & (1 << payload_bits_per_byte)) == 0) {
      uncompressed_sizes[idx] = uncompressed_size;
      return;
    }
    shift += payload_bits_per_byte;
  }
  // Invalid varint
  uncompressed_sizes[idx] = 0;
}

void get_snappy_uncompressed_size(device_span<device_span<uint8_t const> const> inputs,
                                  device_span<size_t> uncompressed_sizes,
                                  rmm::cuda_stream_view stream)
{
  if (inputs.empty()) { return; }

  int threads_per_block = 128;
  auto const num_blocks =
    cudf::util::div_rounding_up_safe<size_t>(inputs.size(), threads_per_block);

  get_snappy_uncompressed_size_kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    inputs, uncompressed_sizes);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::detail
