/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/compression.hpp"
#include "io/utilities/block_utils.cuh"
#include "orc_gpu.hpp"
#include "utilities/time_utils.cuh"

#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/orc_types.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/limits>
#include <thrust/for_each.h>
#include <thrust/transform.h>

namespace cudf::io::orc::detail {

using cudf::detail::device_2dspan;
using cudf::io::detail::codec_exec_result;
using cudf::io::detail::codec_status;

constexpr int scratch_buffer_size        = 512 * 4;
constexpr int compact_streams_block_size = 1024;

// Apache ORC reader does not handle zero-length patch lists for RLEv2 mode2
// Workaround replaces zero-length patch lists by a dummy zero patch
constexpr bool zero_pll_war = true;

struct byte_rle_enc_state_s {
  uint32_t literal_run;
  uint32_t repeat_run;
  uint32_t rpt_map[(512 / 32) + 1];
};

struct intrle_enc_state_s {
  uint32_t literal_run;
  uint32_t delta_run;
  uint32_t literal_mode;
  uint32_t literal_w;
  uint32_t hdr_bytes;
  uint32_t pl_bytes;
  uint32_t delta_map[(512 / 32) + 1];
};

struct strdata_enc_state_s {
  uint32_t char_count;
  uint32_t lengths_red[(512 / 32)];
  char const* str_data[512];
};

struct orcenc_state_s {
  uint32_t cur_row;       // Current row in group
  uint32_t present_rows;  // # of rows in present buffer
  uint32_t present_out;   // # of rows in present buffer that have been flushed
  uint32_t nrows;         // # of rows in current batch
  uint32_t numvals;       // # of non-zero values in current batch (<=nrows)
  uint32_t numlengths;    // # of non-zero values in DATA2 batch
  uint32_t nnz;           // Running count of non-null values
  encoder_chunk_streams stream;
  encoder_chunk chunk;
  uint32_t strm_pos[CI_NUM_STREAMS];
  uint8_t valid_buf[512];  // valid map bits
  union {
    byte_rle_enc_state_s byte_rle;
    intrle_enc_state_s intrle;
    strdata_enc_state_s strenc;
    stripe_dictionary const* dict_stripe;
  } u;
  union {
    uint8_t u8[scratch_buffer_size];  // gblock_vminscratch buffer
    uint32_t u32[scratch_buffer_size / 4];
  } buf;
  union {
    uint8_t u8[2048];
    uint32_t u32[1024];
    int32_t i32[1024];
    uint64_t u64[1024];
    int64_t i64[1024];
  } vals;
  union {
    uint8_t u8[2048];
    uint32_t u32[1024];
    uint64_t u64[1024];
  } lengths;
};

static inline __device__ uint32_t zigzag(uint32_t v) { return v; }
static inline __device__ uint32_t zigzag(int32_t v)
{
  int32_t s = (v >> 31);
  return ((v ^ s) * 2) - s;
}
static inline __device__ uint64_t zigzag(uint64_t v) { return v; }
static inline __device__ uint64_t zigzag(int64_t v)
{
  int64_t s = (v < 0) ? 1 : 0;
  return ((v ^ -s) * 2) + s;
}

static inline __device__ __uint128_t zigzag(__int128_t v)
{
  int64_t s = (v < 0) ? 1 : 0;
  return ((v ^ -s) * 2) + s;
}

static inline __device__ uint32_t count_leading_bytes_32(int32_t v) { return __clz(v) >> 3; }
static inline __device__ uint32_t count_leading_bytes_64(int64_t v) { return __clzll(v) >> 3; }

/**
 * @brief Raw data output
 *
 * @tparam cid stream type (strm_pos[cid] will be updated and output stored at
 * streams[cid]+strm_pos[cid])
 * @tparam inmask input buffer position mask for circular buffers
 * @param[in] s encoder state
 * @param[in] inbuf base input buffer
 * @param[in] inpos position in input buffer
 * @param[in] count number of bytes to encode
 * @param[in] t thread id
 */
template <stream_index_type cid, uint32_t inmask>
static __device__ void store_bytes(
  orcenc_state_s* s, uint8_t const* inbuf, uint32_t inpos, uint32_t count, int t)
{
  uint8_t* dst = s->stream.data_ptrs[cid] + s->strm_pos[cid];
  while (count > 0) {
    uint32_t n = min(count, 512);
    if (t < n) { dst[t] = inbuf[(inpos + t) & inmask]; }
    dst += n;
    inpos += n;
    count -= n;
  }
  __syncthreads();
  if (!t) { s->strm_pos[cid] = static_cast<uint32_t>(dst - s->stream.data_ptrs[cid]); }
}

/**
 * @brief byte_rle encoder
 *
 * @tparam cid stream type (strm_pos[cid] will be updated and output stored at
 * streams[cid]+strm_pos[cid])
 * @tparam inmask input buffer position mask for circular buffers
 * @param[in] s encoder state
 * @param[in] inbuf base input buffer
 * @param[in] inpos position in input buffer
 * @param[in] numvals max number of values to encode
 * @param[in] flush encode all remaining values if nonzero
 * @param[in] t thread id
 *
 * @return number of input values encoded
 */
template <stream_index_type cid, uint32_t inmask>
static __device__ uint32_t byte_rle(
  orcenc_state_s* s, uint8_t const* inbuf, uint32_t inpos, uint32_t numvals, uint32_t flush, int t)
{
  uint8_t* dst     = s->stream.data_ptrs[cid] + s->strm_pos[cid];
  uint32_t out_cnt = 0;

  while (numvals > 0) {
    uint8_t v0       = (t < numvals) ? inbuf[(inpos + t) & inmask] : 0;
    uint8_t v1       = (t + 1 < numvals) ? inbuf[(inpos + t + 1) & inmask] : 0;
    uint32_t rpt_map = ballot(t + 1 < numvals && v0 == v1), literal_run, repeat_run,
             maxvals = min(numvals, 512);
    if (!(t & 0x1f)) s->u.byte_rle.rpt_map[t >> 5] = rpt_map;
    __syncthreads();
    if (t == 0) {
      // Find the start of an identical 3-byte sequence
      // TBD: The two loops below could be eliminated using more ballot+ffs using warp0
      literal_run = 0;
      repeat_run  = 0;
      while (literal_run < maxvals) {
        uint32_t next = s->u.byte_rle.rpt_map[(literal_run >> 5) + 1];
        uint32_t mask = rpt_map & __funnelshift_r(rpt_map, next, 1);
        if (mask) {
          uint32_t literal_run_ofs = __ffs(mask) - 1;
          literal_run += literal_run_ofs;
          repeat_run = __ffs(~((rpt_map >> literal_run_ofs) >> 1));
          if (repeat_run + literal_run_ofs == 32) {
            while (next == ~0) {
              uint32_t next_idx = ((literal_run + repeat_run) >> 5) + 1;
              next              = (next_idx < 512 / 32) ? s->u.byte_rle.rpt_map[next_idx] : 0;
              repeat_run += 32;
            }
            repeat_run += __ffs(~next) - 1;
          }
          repeat_run = min(repeat_run + 1, maxvals - min(literal_run, maxvals));
          if (repeat_run < 3) {
            literal_run += (flush && literal_run + repeat_run >= numvals) ? repeat_run : 0;
            repeat_run = 0;
          }
          break;
        }
        rpt_map = next;
        literal_run += 32;
      }
      if (repeat_run >= 130) {
        // Limit large runs to multiples of 130
        repeat_run = (repeat_run >= 3 * 130) ? 3 * 130 : (repeat_run >= 2 * 130) ? 2 * 130 : 130;
      } else if (literal_run && literal_run + repeat_run == maxvals) {
        repeat_run = 0;  // Try again at next iteration
      }
      s->u.byte_rle.repeat_run  = repeat_run;
      s->u.byte_rle.literal_run = min(literal_run, maxvals);
    }
    __syncthreads();
    literal_run = s->u.byte_rle.literal_run;
    if (!flush && literal_run == numvals) {
      literal_run &= ~0x7f;
      if (!literal_run) break;
    }
    if (literal_run > 0) {
      uint32_t num_runs = (literal_run + 0x7f) >> 7;
      if (t < literal_run) {
        uint32_t run_id = t >> 7;
        uint32_t run    = min(literal_run - run_id * 128, 128);
        if (!(t & 0x7f)) dst[run_id + t] = 0x100 - run;
        dst[run_id + t + 1] = (cid == CI_PRESENT) ? __brev(v0) >> 24 : v0;
      }
      dst += num_runs + literal_run;
      out_cnt += literal_run;
      numvals -= literal_run;
      inpos += literal_run;
    }
    repeat_run = s->u.byte_rle.repeat_run;
    if (repeat_run > 0) {
      while (repeat_run >= 130) {
        if (t == literal_run)  // repeat_run follows literal_run
        {
          dst[0] = 0x7f;
          dst[1] = (cid == CI_PRESENT) ? __brev(v0) >> 24 : v0;
        }
        dst += 2;
        out_cnt += 130;
        numvals -= 130;
        inpos += 130;
        repeat_run -= 130;
      }
      if (!flush && repeat_run == numvals) {
        // Wait for more data in case we can continue the run later
        break;
      }
      if (repeat_run >= 3) {
        if (t == literal_run)  // repeat_run follows literal_run
        {
          dst[0] = repeat_run - 3;
          dst[1] = (cid == CI_PRESENT) ? __brev(v0) >> 24 : v0;
        }
        dst += 2;
        out_cnt += repeat_run;
        numvals -= repeat_run;
        inpos += repeat_run;
      }
    }
  }
  if (!t) { s->strm_pos[cid] = static_cast<uint32_t>(dst - s->stream.data_ptrs[cid]); }
  return out_cnt;
}

/**
 * @brief Maps the symbol size in bytes to RLEv2 5-bit length code
 */
static __device__ const __constant__ uint8_t kByteLengthToRLEv2_W[9] = {
  0, 7, 15, 23, 27, 28, 29, 30, 31};

/**
 * @brief Encode a varint value, return the number of bytes written
 */
static inline __device__ uint32_t store_varint(uint8_t* dst, __uint128_t v)
{
  uint32_t bytecnt = 0;
  for (;;) {
    auto c = static_cast<uint32_t>(v & 0x7f);
    v >>= 7u;
    if (v == 0) {
      dst[bytecnt++] = c;
      break;
    } else {
      dst[bytecnt++] = c + 0x80;
    }
  }
  return bytecnt;
}

template <class T>
static inline __device__ void store_bytes_big_endian(uint8_t* dst, T v, uint32_t w)
{
  for (uint32_t i = 0, b = w * 8; i < w; ++i) {
    b -= 8;
    dst[i] = static_cast<uint8_t>(v >> b);
  }
}

// Combine and store bits for symbol widths less than 8
static inline __device__ void StoreBitsBigEndian(
  uint8_t* dst, uint32_t v, uint32_t w, int num_vals, int t)
{
  if (t <= (num_vals | 0x1f)) {
    uint32_t mask;
    if (w <= 1) {
      v    = (v << 1) | (shuffle_xor(v, 1) & 0x1);
      v    = (v << 2) | (shuffle_xor(v, 2) & 0x3);
      v    = (v << 4) | (shuffle_xor(v, 4) & 0xf);
      mask = 0x7;
    } else if (w <= 2) {
      v    = (v << 2) | (shuffle_xor(v, 1) & 0x3);
      v    = (v << 4) | (shuffle_xor(v, 2) & 0xf);
      mask = 0x3;
    } else  // if (w <= 4)
    {
      v    = (v << 4) | (shuffle_xor(v, 1) & 0xf);
      mask = 0x1;
    }
    if (t < num_vals && !(t & mask)) { dst[(t * w) >> 3] = static_cast<uint8_t>(v); }
  }
}

/**
 * @brief Integer RLEv2 encoder
 *
 * @tparam cid stream type (strm_pos[cid] will be updated and output stored at
 * streams[cid]+strm_pos[cid])
 * @tparam inmask input buffer position mask for circular buffers
 * @param[in] s encoder state
 * @param[in] inbuf base input buffer
 * @param[in] inpos position in input buffer
 * @param[in] numvals max number of values to encode
 * @param[in] t thread id
 * @param[in] temp_storage shared memory storage to perform block reduce
 *
 * @return number of input values encoded
 */
template <stream_index_type cid,
          class T,
          bool is_signed,
          uint32_t inmask,
          int block_size,
          typename Storage>
static __device__ uint32_t IntegerRLE(
  orcenc_state_s* s, T const* inbuf, uint32_t inpos, uint32_t numvals, int t, Storage& temp_storage)
{
  using block_reduce = cub::BlockReduce<T, block_size>;

  uint8_t* dst     = s->stream.data_ptrs[cid] + s->strm_pos[cid];
  uint32_t out_cnt = 0;
  __shared__ uint64_t block_vmin;

  while (numvals > 0) {
    T v0               = (t < numvals) ? inbuf[(inpos + t) & inmask] : 0;
    T v1               = (t + 1 < numvals) ? inbuf[(inpos + t + 1) & inmask] : 0;
    T v2               = (t + 2 < numvals) ? inbuf[(inpos + t + 2) & inmask] : 0;
    uint32_t delta_map = ballot(t + 2 < numvals && v1 - v0 == v2 - v1), maxvals = min(numvals, 512),
             literal_run, delta_run;
    if (!(t & 0x1f)) s->u.intrle.delta_map[t >> 5] = delta_map;
    __syncthreads();
    if (!t) {
      // Find the start of the next delta run (2 consecutive values with the same delta)
      literal_run = delta_run = 0;
      while (literal_run < maxvals) {
        if (delta_map != 0) {
          uint32_t literal_run_ofs = __ffs(delta_map) - 1;
          literal_run += literal_run_ofs;
          delta_run = __ffs(~((delta_map >> literal_run_ofs) >> 1));
          if (literal_run_ofs + delta_run == 32) {
            for (;;) {
              uint32_t delta_idx = (literal_run + delta_run) >> 5;
              delta_map          = (delta_idx < 512 / 32) ? s->u.intrle.delta_map[delta_idx] : 0;
              if (delta_map != ~0) break;
              delta_run += 32;
            }
            delta_run += __ffs(~delta_map) - 1;
          }
          delta_run += 2;
          break;
        }
        literal_run += 32;
        delta_map = s->u.intrle.delta_map[(literal_run >> 5)];
      }
      literal_run             = min(literal_run, maxvals);
      s->u.intrle.literal_run = literal_run;
      s->u.intrle.delta_run   = min(delta_run, maxvals - literal_run);
    }
    __syncthreads();
    literal_run = s->u.intrle.literal_run;
    // Find minimum and maximum values
    if (literal_run > 0) {
      // Find min & max
      T vmin = (t < literal_run) ? v0 : cuda::std::numeric_limits<T>::max();
      T vmax = (t < literal_run) ? v0 : cuda::std::numeric_limits<T>::min();
      uint32_t literal_mode, literal_w;
      vmin = block_reduce(temp_storage).Reduce(vmin, cuda::minimum{});
      __syncthreads();
      vmax = block_reduce(temp_storage).Reduce(vmax, cuda::maximum{});
      if (t == 0) {
        uint32_t mode1_w, mode2_w;
        typename std::make_unsigned<T>::type vrange_mode1, vrange_mode2;
        block_vmin = static_cast<uint64_t>(vmin);
        if constexpr (sizeof(T) > 4) {
          vrange_mode1 = (is_signed) ? max(zigzag(vmin), zigzag(vmax)) : vmax;
          vrange_mode2 = vmax - vmin;
          mode1_w      = 8 - min(count_leading_bytes_64(vrange_mode1), 7);
          mode2_w      = 8 - min(count_leading_bytes_64(vrange_mode2), 7);
        } else {
          vrange_mode1 = (is_signed) ? max(zigzag(vmin), zigzag(vmax)) : vmax;
          vrange_mode2 = vmax - vmin;
          mode1_w      = 4 - min(count_leading_bytes_32(vrange_mode1), 3);
          mode2_w      = 4 - min(count_leading_bytes_32(vrange_mode2), 3);
        }
        // Decide between mode1 & mode2 (also mode3 for length=2 repeat)
        if (vrange_mode2 == 0 && mode1_w > 1) {
          // Should only occur if literal_run==2 (otherwise would have resulted in repeat_run >=
          // 3)
          uint32_t bytecnt = 2;
          dst[0]           = 0xC0 + ((literal_run - 1) >> 8);
          dst[1]           = (literal_run - 1) & 0xff;
          bytecnt += store_varint(dst + 2, vrange_mode1);
          dst[bytecnt++]           = 0;  // Zero delta
          s->u.intrle.literal_mode = 3;
          s->u.intrle.literal_w    = bytecnt;
        } else {
          uint32_t range, w;
          // Mode 2 base value cannot be bigger than max int64_t, i.e. the first bit has to be 0
          if (vmin <= cuda::std::numeric_limits<int64_t>::max() and mode1_w > mode2_w and
              (literal_run - 1) * (mode1_w - mode2_w) > 4) {
            s->u.intrle.literal_mode = 2;
            w                        = mode2_w;
            range                    = (uint32_t)vrange_mode2;
          } else {
            s->u.intrle.literal_mode = 1;
            w                        = mode1_w;
            range                    = (uint32_t)vrange_mode1;
          }
          if (w == 1)
            w = (range >= 16) ? w << 3 : (range >= 4) ? 4 : (range >= 2) ? 2 : 1;
          else
            w <<= 3;  // bytes -> bits
          s->u.intrle.literal_w = w;
        }
      }
      __syncthreads();
      vmin         = static_cast<T>(block_vmin);
      literal_mode = s->u.intrle.literal_mode;
      literal_w    = s->u.intrle.literal_w;
      if (literal_mode == 1) {
        // Direct mode
        if (!t) {
          dst[0] = 0x40 +
                   ((literal_w < 8) ? literal_w - 1 : kByteLengthToRLEv2_W[literal_w >> 3]) * 2 +
                   ((literal_run - 1) >> 8);
          dst[1] = (literal_run - 1) & 0xff;
        }
        dst += 2;

        typename std::make_unsigned<T>::type zzv0 = v0;
        if (t < literal_run) { zzv0 = zigzag(v0); }
        if (literal_w < 8) {
          StoreBitsBigEndian(dst, zzv0, literal_w, literal_run, t);
        } else if (t < literal_run) {
          store_bytes_big_endian(dst + t * (literal_w >> 3), zzv0, (literal_w >> 3));
        }
      } else if (literal_mode == 2) {
        // Patched base mode
        if (!t) {
          uint32_t bw, pw = 1, pll, pgw = 1, bv_scale = (is_signed) ? 0 : 1;
          vmax = (is_signed) ? ((vmin < 0) ? -vmin : vmin) * 2 : vmin;
          bw   = (sizeof(T) > 4) ? (8 - min(count_leading_bytes_64(vmax << bv_scale), 7))
                                 : (4 - min(count_leading_bytes_32(vmax << bv_scale), 3));
          if (zero_pll_war) {
            // Insert a dummy zero patch
            pll                                                    = 1;
            dst[4 + bw + ((literal_run * literal_w + 7) >> 3) + 0] = 0;
            dst[4 + bw + ((literal_run * literal_w + 7) >> 3) + 1] = 0;
          } else {
            pll = 0;
          }
          dst[0] = 0x80 +
                   ((literal_w < 8) ? literal_w - 1 : kByteLengthToRLEv2_W[literal_w >> 3]) * 2 +
                   ((literal_run - 1) >> 8);
          dst[1] = (literal_run - 1) & 0xff;
          dst[2] = ((bw - 1) << 5) | kByteLengthToRLEv2_W[pw];
          dst[3] = ((pgw - 1) << 5) | pll;
          if (is_signed) {
            vmax >>= 1;
            vmax |= vmin & ((T)1 << (bw * 8 - 1));
          }
          store_bytes_big_endian(dst + 4, vmax, bw);
          s->u.intrle.hdr_bytes = 4 + bw;
          s->u.intrle.pl_bytes  = (pll * (pw * 8 + pgw) + 7) >> 3;
        }
        __syncthreads();
        dst += s->u.intrle.hdr_bytes;
        v0 -= (t < literal_run) ? vmin : 0;
        if (literal_w < 8)
          StoreBitsBigEndian(dst, (uint32_t)v0, literal_w, literal_run, t);
        else if (t < literal_run)
          store_bytes_big_endian(dst + t * (literal_w >> 3), v0, (literal_w >> 3));
        dst += s->u.intrle.pl_bytes;
      } else {
        // Delta mode
        dst += literal_w;
        literal_w = 0;
      }
      dst += (literal_run * literal_w + 7) >> 3;
      numvals -= literal_run;
      inpos += literal_run;
      out_cnt += literal_run;
      __syncthreads();
    }
    delta_run = s->u.intrle.delta_run;
    if (delta_run > 0) {
      if (t == literal_run) {
        int64_t delta       = (int64_t)v1 - (int64_t)v0;
        uint64_t delta_base = zigzag(v0);
        if (delta == 0 && delta_run >= 3 && delta_run <= 10) {
          // Short repeat
          uint32_t delta_bw = 8 - min(count_leading_bytes_64(delta_base), 7);
          dst[0]            = ((delta_bw - 1) << 3) + (delta_run - 3);
          for (uint32_t i = 0, b = delta_bw * 8; i < delta_bw; i++) {
            b -= 8;
            dst[1 + i] = static_cast<uint8_t>(delta_base >> b);
          }
          s->u.intrle.hdr_bytes = 1 + delta_bw;
        } else {
          // Delta
          uint64_t delta_u = zigzag(delta);
          uint32_t bytecnt = 2;
          dst[0]           = 0xC0 + ((delta_run - 1) >> 8);
          dst[1]           = (delta_run - 1) & 0xff;
          bytecnt += store_varint(dst + bytecnt, delta_base);
          bytecnt += store_varint(dst + bytecnt, delta_u);
          s->u.intrle.hdr_bytes = bytecnt;
        }
      }
      __syncthreads();
      dst += s->u.intrle.hdr_bytes;
      numvals -= delta_run;
      inpos += delta_run;
      out_cnt += delta_run;
    }
  }
  if (!t) { s->strm_pos[cid] = static_cast<uint32_t>(dst - s->stream.data_ptrs[cid]); }
  __syncthreads();
  return out_cnt;
}

/**
 * @brief Store a group of strings as a single concatenated string
 *
 * @param[in] dst destination buffer
 * @param[in] strenc string encoder state
 * @param[in] len(t) string length (per thread)
 * @param[in] t thread id
 */
static __device__ void store_string_data(uint8_t* dst,
                                         strdata_enc_state_s* strenc,
                                         uint32_t len,
                                         int t)
{
  // Start with summing up all the lengths
  uint32_t pos = len;
  uint32_t wt  = t & 0x1f;
  for (uint32_t n = 1; n < 32; n <<= 1) {
    uint32_t tmp = shuffle(pos, (wt & ~n) | (n - 1));
    pos += (wt & n) ? tmp : 0;
  }
  if (wt == 0x1f) { strenc->lengths_red[t >> 5] = pos; }
  dst += pos - len;
  __syncthreads();
  if (t < 32) {
    uint32_t wlen = (wt < 16) ? strenc->lengths_red[wt] : 0;
    uint32_t wpos = wlen;
    for (uint32_t n = 1; n < 16; n <<= 1) {
      uint32_t tmp = shuffle(wpos, (wt & ~n) | (n - 1));
      wpos += (wt & n) ? tmp : 0;
    }
    if (wt < 16) { strenc->lengths_red[wt] = wpos - wlen; }
    if (wt == 0xf) {
      strenc->char_count = wpos;  // Update stream position
    }
  }
  __syncthreads();
  // TBD: Might be more efficient to loop over 4 strings and copy 8 consecutive character at a time
  // rather than have each thread to a memcpy
  if (len > 0) { memcpy(dst + strenc->lengths_red[t >> 5], strenc->str_data[t], len); }
}

/**
 * @brief In-place conversion from lengths to positions
 *
 * @param[in] vals input values
 * @param[in] numvals number of values
 * @param[in] t thread id
 */
template <class T>
inline __device__ void lengths_to_positions(T* vals, uint32_t numvals, unsigned int t)
{
  for (uint32_t n = 1; n < numvals; n <<= 1) {
    __syncthreads();
    if ((t & n) && (t < numvals)) vals[t] += vals[(t & ~n) | (n - 1)];
  }
}

template <int block_size, typename Storage>
static __device__ void encode_null_mask(orcenc_state_s* s,
                                        bitmask_type const* pushdown_mask,
                                        Storage& scan_storage,
                                        int t)
{
  if (s->stream.ids[CI_PRESENT] < 0) return;

  auto const column = *s->chunk.column;
  while (s->present_rows < s->chunk.null_mask_num_rows or s->numvals > 0) {
    // Number of rows read so far
    auto present_rows = s->present_rows;
    // valid_buf capacity is byte per thread in block
    auto const buf_available_bits = encode_block_size * 8 - s->numvals;
    // Number of rows for the block to process in this iteration
    auto const nrows = min(s->chunk.null_mask_num_rows - present_rows, buf_available_bits);
    // Number of rows for this thread to process in this iteration
    auto const t_nrows = min(max(static_cast<int32_t>(nrows) - t * 8, 0), 8);
    auto const row     = s->chunk.null_mask_start_row + present_rows + t * 8;

    auto get_mask_byte = [&](bitmask_type const* mask, size_type offset) -> uint8_t {
      if (t_nrows == 0) return 0;
      if (mask == nullptr) return 0xff;

      size_type const begin_offset = row + offset;
      auto const end_offset        = min(begin_offset + 8, offset + column.size());
      auto const mask_word = cudf::detail::get_mask_offset_word(mask, 0, begin_offset, end_offset);
      return mask_word & 0xff;
    };

    uint8_t pd_byte     = (1 << t_nrows) - 1;
    uint32_t pd_set_cnt = t_nrows;
    uint32_t offset     = t_nrows != 0 ? t * 8 : nrows;
    if (pushdown_mask != nullptr) {
      pd_byte    = get_mask_byte(pushdown_mask, 0) & ((1 << t_nrows) - 1);
      pd_set_cnt = __popc(pd_byte);
      // Scan the number of valid bits to get dst offset for each thread
      cub::BlockScan<uint32_t, block_size>(scan_storage).ExclusiveSum(pd_set_cnt, offset);
    }

    auto const mask_byte = get_mask_byte(column.null_mask(), column.offset());
    auto dst_offset      = offset + s->nnz;
    auto vbuf_bit_idx    = [](int row) {
      // valid_buf is a circular buffer with validity of 8 rows in each element
      return row % (encode_block_size * 8);
    };
    if (dst_offset % 8 == 0 and pd_set_cnt == 8) {
      s->valid_buf[vbuf_bit_idx(dst_offset) / 8] = mask_byte;
    } else {
      for (auto bit_idx = 0; bit_idx < t_nrows; ++bit_idx) {
        // skip bits where pushdown mask is not set
        if (not(pd_byte & (1 << bit_idx))) continue;
        if (mask_byte & (1 << bit_idx)) {
          set_bit(reinterpret_cast<uint32_t*>(s->valid_buf), vbuf_bit_idx(dst_offset++));
        } else {
          clear_bit(reinterpret_cast<uint32_t*>(s->valid_buf), vbuf_bit_idx(dst_offset++));
        }
      }
    }

    __syncthreads();
    if (t == block_size - 1) {
      // Number of loaded rows, available for encode
      s->numvals += offset + pd_set_cnt;
      // Number of loaded rows (different from present_rows because of pushdown masks)
      s->nnz += offset + pd_set_cnt;
    }
    present_rows += nrows;
    if (!t) { s->present_rows = present_rows; }
    __syncthreads();

    // RLE encode the present stream
    if (s->numvals > ((present_rows < s->chunk.null_mask_num_rows) ? 130 * 8 : 0)) {
      auto const flush      = (present_rows < s->chunk.null_mask_num_rows) ? 0 : 7;
      auto const nbytes_out = (s->numvals + flush) / 8;
      auto const nrows_encoded =
        byte_rle<CI_PRESENT, 0x1ff>(s, s->valid_buf, s->present_out / 8, nbytes_out, flush, t) * 8;

      if (!t) {
        // Number of rows encoded so far
        s->present_out += nrows_encoded;
        s->numvals -= min(s->numvals, nrows_encoded);
      }
      __syncthreads();
    }
  }

  // reset shared state
  if (t == 0) { s->nnz = 0; }
}

/**
 * @brief Encode column data
 *
 * @param[in] chunks encoder chunks device array [column][rowgroup]
 * @param[in, out] streams chunk streams device array [column][rowgroup]
 */
// blockDim {`encode_block_size`,1,1}
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  encode_column_data_kernel(device_2dspan<encoder_chunk const> chunks,
                            device_2dspan<encoder_chunk_streams> streams)
{
  __shared__ __align__(16) orcenc_state_s state_g;
  __shared__ union {
    typename cub::BlockScan<uint32_t, block_size>::TempStorage scan_u32;
    typename cub::BlockReduce<int32_t, block_size>::TempStorage i32;
    typename cub::BlockReduce<int64_t, block_size>::TempStorage i64;
    typename cub::BlockReduce<uint32_t, block_size>::TempStorage u32;
    typename cub::BlockReduce<uint64_t, block_size>::TempStorage u64;
  } temp_storage;

  orcenc_state_s* const s = &state_g;
  uint32_t col_id         = blockIdx.x / chunks.size().second;
  uint32_t group_id       = blockIdx.x % chunks.size().second;
  int t                   = threadIdx.x;
  if (t == 0) {
    s->chunk                = chunks[col_id][group_id];
    s->stream               = streams[col_id][group_id];
    s->cur_row              = 0;
    s->present_rows         = 0;
    s->present_out          = 0;
    s->numvals              = 0;
    s->numlengths           = 0;
    s->nnz                  = 0;
    s->strm_pos[CI_DATA]    = 0;
    s->strm_pos[CI_PRESENT] = 0;
    s->strm_pos[CI_INDEX]   = 0;
    // Dictionary data is encoded in a separate kernel
    s->strm_pos[CI_DATA2] =
      s->chunk.encoding_kind == DICTIONARY_V2 ? s->stream.lengths[CI_DATA2] : 0;
    s->strm_pos[CI_DICTIONARY] =
      s->chunk.encoding_kind == DICTIONARY_V2 ? s->stream.lengths[CI_DICTIONARY] : 0;
  }
  __syncthreads();

  auto const pushdown_mask = [&]() -> cudf::bitmask_type const* {
    auto const parent_index = s->chunk.column->parent_index;
    if (!parent_index.has_value()) return nullptr;
    return chunks[parent_index.value()][0].column->pushdown_mask;
  }();

  encode_null_mask<block_size>(s, pushdown_mask, temp_storage.scan_u32, t);
  __syncthreads();

  auto const column = *s->chunk.column;
  while (s->cur_row < s->chunk.num_rows || s->numvals + s->numlengths != 0) {
    // Fetch non-null values
    auto const length_stream_only = s->chunk.type_kind == LIST or s->chunk.type_kind == MAP;
    if (not length_stream_only && s->stream.data_ptrs[CI_DATA] == nullptr) {
      // Pass-through
      __syncthreads();
      if (!t) {
        s->cur_row           = s->chunk.num_rows;
        s->strm_pos[CI_DATA] = s->chunk.num_rows * s->chunk.dtype_len;
      }
    } else if (s->cur_row < s->chunk.num_rows) {
      uint32_t maxnumvals = (s->chunk.type_kind == BOOLEAN) ? 2048 : 1024;
      uint32_t nrows =
        min(min(s->chunk.num_rows - s->cur_row, maxnumvals - max(s->numvals, s->numlengths)),
            encode_block_size);
      auto const row = s->chunk.start_row + s->cur_row + t;

      auto const is_value_valid = [&]() {
        if (t >= nrows) return false;
        return bit_value_or(pushdown_mask, column.offset() + row, true) and
               bit_value_or(column.null_mask(), column.offset() + row, true);
      }();
      s->buf.u32[t] = is_value_valid ? 1u : 0u;

      // TODO: Could use a faster reduction relying on _popc() for the initial phase
      lengths_to_positions(s->buf.u32, encode_block_size, t);
      __syncthreads();
      if (is_value_valid) {
        int nz_idx = (s->nnz + s->buf.u32[t] - 1) & (maxnumvals - 1);
        switch (s->chunk.type_kind) {
          case INT:
          case DATE:
          case FLOAT: s->vals.u32[nz_idx] = column.element<uint32_t>(row); break;
          case DOUBLE:
          case LONG: s->vals.u64[nz_idx] = column.element<uint64_t>(row); break;
          case SHORT: s->vals.u32[nz_idx] = column.element<uint16_t>(row); break;
          case BOOLEAN:
          case BYTE: s->vals.u8[nz_idx] = column.element<uint8_t>(row); break;
          case TIMESTAMP: {
            int64_t ts          = column.element<int64_t>(row);
            int32_t ts_scale    = cudf::detail::powers_of_ten[9 - min(s->chunk.scale, 9)];
            int64_t seconds     = ts / ts_scale;
            int64_t nanos       = (ts - seconds * ts_scale);
            s->vals.i64[nz_idx] = seconds - orc_utc_epoch;
            if (nanos != 0) {
              // Trailing zeroes are encoded in the lower 3-bits
              uint32_t zeroes = 0;
              nanos *= cudf::detail::powers_of_ten[min(s->chunk.scale, 9)];
              if (!(nanos % 100)) {
                nanos /= 100;
                zeroes = 1;
                while (zeroes < 7 && !(nanos % 10)) {
                  nanos /= 10;
                  zeroes++;
                }
              }
              nanos = (nanos << 3) + zeroes;
            }
            s->lengths.u64[nz_idx] = nanos;
            break;
          }
          case STRING:
            if (s->chunk.encoding_kind == DICTIONARY_V2) {
              uint32_t dict_idx = s->chunk.dict_index[row];
              if (dict_idx > 0x7fff'ffffu) {
                dict_idx = s->chunk.dict_index[dict_idx & 0x7fff'ffffu];
              }
              // translate dictionary index to sorted order, if enabled
              if (s->chunk.dict_data_order != nullptr) {
                dict_idx = s->chunk.dict_data_order[dict_idx];
              }
              s->vals.u32[nz_idx] = dict_idx;
            } else {
              string_view value                       = column.element<string_view>(row);
              s->u.strenc.str_data[s->buf.u32[t] - 1] = value.data();
              s->lengths.u32[nz_idx]                  = value.size_bytes();
            }
            break;
            // Reusing the lengths array for the scale stream
            // Note: can be written in a faster manner, given that all values are equal
          case DECIMAL: s->lengths.u32[nz_idx] = zigzag(s->chunk.scale); break;
          case LIST:
          case MAP: {
            auto const& offsets = column.child(lists_column_view::offsets_column_index);
            // Compute list length from the offsets
            s->lengths.u32[nz_idx] = offsets.element<size_type>(row + 1 + column.offset()) -
                                     offsets.element<size_type>(row + column.offset());
          } break;
          default: break;
        }
      }
      __syncthreads();
      if (s->chunk.type_kind == STRING && s->chunk.encoding_kind != DICTIONARY_V2) {
        // Store string data
        uint32_t nz     = s->buf.u32[511];
        uint32_t nz_idx = (s->nnz + t) & 0x3ff;
        uint32_t len    = (t < nz && s->u.strenc.str_data[t]) ? s->lengths.u32[nz_idx] : 0;
        store_string_data(
          s->stream.data_ptrs[CI_DATA] + s->strm_pos[CI_DATA], &s->u.strenc, len, t);
        if (!t) { s->strm_pos[CI_DATA] += s->u.strenc.char_count; }
        __syncthreads();
      } else if (s->chunk.type_kind == BOOLEAN) {
        // bool8 -> 8x bool1
        uint32_t nz = s->buf.u32[511];
        uint8_t n   = ((s->nnz + nz) - (s->nnz & ~7) + 7) >> 3;
        if (t < n) {
          uint32_t idx8                              = (s->nnz & ~7) + (t << 3);
          s->lengths.u8[((s->nnz >> 3) + t) & 0x1ff] = ((s->vals.u8[(idx8 + 0) & 0x7ff] & 1) << 7) |
                                                       ((s->vals.u8[(idx8 + 1) & 0x7ff] & 1) << 6) |
                                                       ((s->vals.u8[(idx8 + 2) & 0x7ff] & 1) << 5) |
                                                       ((s->vals.u8[(idx8 + 3) & 0x7ff] & 1) << 4) |
                                                       ((s->vals.u8[(idx8 + 4) & 0x7ff] & 1) << 3) |
                                                       ((s->vals.u8[(idx8 + 5) & 0x7ff] & 1) << 2) |
                                                       ((s->vals.u8[(idx8 + 6) & 0x7ff] & 1) << 1) |
                                                       ((s->vals.u8[(idx8 + 7) & 0x7ff] & 1) << 0);
        }
        __syncthreads();
      }
      if (!t) {
        uint32_t nz = s->buf.u32[511];
        s->nnz += nz;
        s->numvals += nz;
        s->numlengths += (s->chunk.type_kind == TIMESTAMP || s->chunk.type_kind == DECIMAL ||
                          s->chunk.type_kind == LIST || s->chunk.type_kind == MAP ||
                          (s->chunk.type_kind == STRING && s->chunk.encoding_kind != DICTIONARY_V2))
                           ? nz
                           : 0;
        s->cur_row += nrows;
      }
      __syncthreads();
      // Encode values
      if (s->numvals > 0) {
        uint32_t flush = (s->cur_row == s->chunk.num_rows) ? 7 : 0, n;
        switch (s->chunk.type_kind) {
          case SHORT:
          case INT:
          case DATE:
            n = IntegerRLE<CI_DATA, int32_t, true, 0x3ff, block_size>(
              s, s->vals.i32, s->nnz - s->numvals, s->numvals, t, temp_storage.i32);
            break;
          case LONG:
          case TIMESTAMP:
            n = IntegerRLE<CI_DATA, int64_t, true, 0x3ff, block_size>(
              s, s->vals.i64, s->nnz - s->numvals, s->numvals, t, temp_storage.i64);
            break;
          case BYTE:
            n = byte_rle<CI_DATA, 0x3ff>(s, s->vals.u8, s->nnz - s->numvals, s->numvals, flush, t);
            break;
          case BOOLEAN:
            n = byte_rle<CI_DATA, 0x1ff>(s,
                                         s->lengths.u8,
                                         (s->nnz - s->numvals + flush) >> 3,
                                         (s->numvals + flush) >> 3,
                                         flush,
                                         t) *
                8;
            break;
          case FLOAT:
            store_bytes<CI_DATA, 0xfff>(
              s, s->vals.u8, (s->nnz - s->numvals) * 4, s->numvals * 4, t);
            n = s->numvals;
            break;
          case DOUBLE:
            store_bytes<CI_DATA, 0x1fff>(
              s, s->vals.u8, (s->nnz - s->numvals) * 8, s->numvals * 8, t);
            n = s->numvals;
            break;
          case STRING:
            if (s->chunk.encoding_kind == DICTIONARY_V2) {
              n = IntegerRLE<CI_DATA, uint32_t, false, 0x3ff, block_size>(
                s, s->vals.u32, s->nnz - s->numvals, s->numvals, t, temp_storage.u32);
            } else {
              n = s->numvals;
            }
            break;
          case DECIMAL: {
            if (is_value_valid) {
              auto const id = column.type().id();
              __uint128_t const zz_val =
                id == type_id::DECIMAL32   ? zigzag(column.element<int32_t>(row))
                : id == type_id::DECIMAL64 ? zigzag(column.element<int64_t>(row))
                                           : zigzag(column.element<__int128_t>(row));
              auto const offset =
                (row == s->chunk.start_row) ? 0 : s->chunk.decimal_offsets[row - 1];
              store_varint(s->stream.data_ptrs[CI_DATA] + offset, zz_val);
            }
            n = s->numvals;
          } break;
          default: n = s->numvals; break;
        }
        __syncthreads();
        if (!t) { s->numvals -= min(n, s->numvals); }
      }
      // Encode secondary stream values
      if (s->numlengths > 0) {
        uint32_t n;
        switch (s->chunk.type_kind) {
          case TIMESTAMP:
            n = IntegerRLE<CI_DATA2, uint64_t, false, 0x3ff, block_size>(
              s, s->lengths.u64, s->nnz - s->numlengths, s->numlengths, t, temp_storage.u64);
            break;
          case DECIMAL:
          case LIST:
          case MAP:
          case STRING:
            n = IntegerRLE<CI_DATA2, uint32_t, false, 0x3ff, block_size>(
              s, s->lengths.u32, s->nnz - s->numlengths, s->numlengths, t, temp_storage.u32);
            break;
          default: n = s->numlengths; break;
        }
        __syncthreads();
        if (!t) { s->numlengths -= min(n, s->numlengths); }
      }
    }
    __syncthreads();
  }
  __syncthreads();
  if (t <= CI_PRESENT && s->stream.ids[t] >= 0) {
    // Update actual compressed length
    // (not needed for decimal data, whose exact size is known before encode)
    if (!(t == CI_DATA && s->chunk.type_kind == DECIMAL))
      streams[col_id][group_id].lengths[t] = s->strm_pos[t];
    if (!s->stream.data_ptrs[t]) {
      streams[col_id][group_id].data_ptrs[t] =
        static_cast<uint8_t*>(const_cast<void*>(column.head())) +
        (column.offset() + s->chunk.start_row) * s->chunk.dtype_len;
    }
  }
}

/**
 * @brief Encode column dictionaries
 *
 * @param[in] stripes Stripe dictionaries device array
 * @param[in] columns Pre-order flattened device array of ORC column views
 * @param[in] chunks encoder_chunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 */
// blockDim {512,1,1}
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  encode_string_dictionaries_kernel(stripe_dictionary const* stripes,
                                    device_span<orc_column_device_view const> columns,
                                    device_2dspan<encoder_chunk const> chunks,
                                    device_2dspan<encoder_chunk_streams> streams)
{
  __shared__ __align__(16) orcenc_state_s state_g;
  __shared__ typename cub::BlockReduce<uint32_t, block_size>::TempStorage temp_storage;

  orcenc_state_s* const s = &state_g;
  uint32_t stripe_id      = blockIdx.x;
  uint32_t cid            = (blockIdx.y) ? CI_DICTIONARY : CI_DATA2;
  int t                   = threadIdx.x;

  if (t == 0) s->u.dict_stripe = &stripes[stripe_id];

  __syncthreads();
  auto const strm_ptr = &streams[s->u.dict_stripe->column_idx][s->u.dict_stripe->start_rowgroup];
  if (t == 0) {
    s->chunk         = chunks[s->u.dict_stripe->column_idx][s->u.dict_stripe->start_rowgroup];
    s->stream        = *strm_ptr;
    s->strm_pos[cid] = 0;
    s->numlengths    = 0;
    s->nrows         = s->u.dict_stripe->entry_count;
    s->cur_row       = 0;
  }
  auto const string_column = columns[s->u.dict_stripe->column_idx];
  auto const dict_data     = s->u.dict_stripe->data;
  __syncthreads();
  if (s->chunk.encoding_kind != DICTIONARY_V2) {
    return;  // This column isn't using dictionary encoding -> bail out
  }

  while (s->cur_row < s->nrows || s->numlengths != 0) {
    uint32_t numvals    = min(s->nrows - s->cur_row, min(1024 - s->numlengths, 512));
    uint32_t string_idx = (t < numvals) ? dict_data[s->cur_row + t] : 0;
    if (cid == CI_DICTIONARY) {
      // Encoding string contents
      char const* ptr = nullptr;
      uint32_t count  = 0;
      if (t < numvals) {
        auto string_val = string_column.element<string_view>(string_idx);
        ptr             = string_val.data();
        count           = string_val.size_bytes();
      }
      s->u.strenc.str_data[t] = ptr;
      store_string_data(s->stream.data_ptrs[CI_DICTIONARY] + s->strm_pos[CI_DICTIONARY],
                        &s->u.strenc,
                        (ptr) ? count : 0,
                        t);
      if (!t) { s->strm_pos[CI_DICTIONARY] += s->u.strenc.char_count; }
    } else {
      // Encoding string lengths
      uint32_t count =
        (t < numvals)
          ? static_cast<uint32_t>(string_column.element<string_view>(string_idx).size_bytes())
          : 0;
      uint32_t nz_idx = (s->cur_row + t) & 0x3ff;
      if (t < numvals) s->lengths.u32[nz_idx] = count;
      __syncthreads();
      if (s->numlengths + numvals > 0) {
        uint32_t n = IntegerRLE<CI_DATA2, uint32_t, false, 0x3ff, block_size>(
          s, s->lengths.u32, s->cur_row, s->numlengths + numvals, t, temp_storage);
        __syncthreads();
        if (!t) {
          s->numlengths += numvals;
          s->numlengths -= min(n, s->numlengths);
        }
      }
    }
    if (t == 0) { s->cur_row += numvals; }
    __syncthreads();
  }
  if (t == 0) { strm_ptr->lengths[cid] = s->strm_pos[cid]; }
}

/**
 * @brief Merge chunked column data into a single contiguous stream
 *
 * @param[in] strm_desc stripe_stream device array [stripe][stream]
 * @param[in] streams List of encoder chunk streams [column][rowgroup]
 * @param[out] srcs  List of source encoder chunk stream data addresses
 * @param[out] dsts List of destination stripe_stream data addresses
 * @param[out] sizes List of stream sizes in bytes
 */
// blockDim {compact_streams_block_size,1,1}
CUDF_KERNEL void __launch_bounds__(compact_streams_block_size)
  init_batched_memcpy_kernel(device_2dspan<stripe_stream const> strm_desc,
                             device_2dspan<encoder_chunk_streams> streams,
                             device_span<uint8_t*> srcs,
                             device_span<uint8_t*> dsts,
                             device_span<size_t> sizes)
{
  auto const stripe_id = (blockIdx.x / strm_desc.size().second) * blockDim.x + threadIdx.x;
  auto const stream_id = blockIdx.x % strm_desc.size().second;
  if (stripe_id >= strm_desc.size().first) { return; }

  auto const out_id = stream_id * strm_desc.size().first + stripe_id;
  stripe_stream ss  = strm_desc[stripe_id][stream_id];

  if (ss.data_ptr == nullptr) { return; }

  auto const cid = ss.stream_type;
  auto dst_ptr   = ss.data_ptr;
  for (auto group = ss.first_chunk_id; group < ss.first_chunk_id + ss.num_chunks; ++group) {
    auto const out_id = stream_id * streams.size().second + group;
    srcs[out_id]      = streams[ss.column_id][group].data_ptrs[cid];
    dsts[out_id]      = dst_ptr;

    // Also update the stream here, data will be copied in a separate kernel
    streams[ss.column_id][group].data_ptrs[cid] = dst_ptr;

    auto const len = streams[ss.column_id][group].lengths[cid];
    // len is the size (in bytes) of the current stream.
    sizes[out_id] = len;
    dst_ptr += len;
  }
}

/**
 * @brief Initializes compression input/output structures
 *
 * @param[in] strm_desc stripe_stream device array [stripe][stream]
 * @param[in] chunks encoder_chunk device array [rowgroup][column]
 * @param[out] inputs Per-block compression input buffers
 * @param[out] outputs Per-block compression output buffers
 * @param[out] results Per-block compression status
 * @param[in] compressed_bfr Compression output buffer
 * @param[in] comp_blk_size Compression block size
 * @param[in] max_comp_blk_size Max size of any block after compression
 * @param[in] comp_block_align Required alignment for compressed blocks
 */
// blockDim {256,1,1}
CUDF_KERNEL void __launch_bounds__(256)
  init_compression_blocks_kernel(device_2dspan<stripe_stream const> strm_desc,
                                 device_2dspan<encoder_chunk_streams const> streams,
                                 device_span<device_span<uint8_t const>> inputs,
                                 device_span<device_span<uint8_t>> outputs,
                                 device_span<codec_exec_result> results,
                                 device_span<uint8_t> compressed_bfr,
                                 uint32_t comp_blk_size,
                                 uint32_t max_comp_blk_size,
                                 uint32_t comp_block_align)
{
  __shared__ __align__(16) stripe_stream ss;
  __shared__ uint8_t* uncomp_base_g;

  auto const padded_block_header_size = util::round_up_unsafe(block_header_size, comp_block_align);
  auto const padded_comp_block_size   = util::round_up_unsafe(max_comp_blk_size, comp_block_align);

  auto const stripe_id = blockIdx.x / strm_desc.size().second;
  auto const stream_id = blockIdx.x % strm_desc.size().second;
  uint32_t t           = threadIdx.x;
  uint32_t num_blocks;
  uint8_t *src, *dst;

  if (t == 0) {
    ss            = strm_desc[stripe_id][stream_id];
    uncomp_base_g = streams[ss.column_id][ss.first_chunk_id].data_ptrs[ss.stream_type];
  }
  __syncthreads();
  src        = uncomp_base_g;
  dst        = compressed_bfr.data() + ss.bfr_offset;
  num_blocks = (ss.stream_size > 0) ? (ss.stream_size - 1) / comp_blk_size + 1 : 1;
  for (uint32_t b = t; b < num_blocks; b += 256) {
    uint32_t blk_size = min(comp_blk_size, ss.stream_size - min(b * comp_blk_size, ss.stream_size));
    inputs[ss.first_block + b] = {src + b * comp_blk_size, blk_size};
    auto const dst_offset =
      padded_block_header_size + b * (padded_block_header_size + padded_comp_block_size);
    outputs[ss.first_block + b] = {dst + dst_offset, max_comp_blk_size};
    results[ss.first_block + b] = {0, codec_status::FAILURE};
  }
}

/**
 * @brief Compacts compressed blocks in a single contiguous stream, and update 3-byte block length
 *fields
 *
 * @param[in,out] strm_desc stripe_stream device array [stripe][stream]
 * @param[in] chunks encoder_chunk device array [rowgroup][column]
 * @param[in] inputs Per-block compression input buffers
 * @param[out] outputs Per-block compression output buffers
 * @param[out] results Per-block compression status
 * @param[in] compressed_bfr Compression output buffer
 * @param[in] comp_blk_size Compression block size
 * @param[in] max_comp_blk_size Max size of any block after compression
 */
// blockDim {1024,1,1}
CUDF_KERNEL void __launch_bounds__(1024)
  compact_compressed_blocks_kernel(device_2dspan<stripe_stream> strm_desc,
                                   device_span<device_span<uint8_t const> const> inputs,
                                   device_span<device_span<uint8_t> const> outputs,
                                   device_span<codec_exec_result> results,
                                   device_span<uint8_t> compressed_bfr,
                                   uint32_t comp_blk_size,
                                   uint32_t max_comp_blk_size)
{
  __shared__ __align__(16) stripe_stream ss;
  __shared__ uint8_t const* comp_src_g;
  __shared__ uint32_t comp_len_g;

  auto const stripe_id = blockIdx.x / strm_desc.size().second;
  auto const stream_id = blockIdx.x % strm_desc.size().second;
  uint32_t t           = threadIdx.x;
  uint32_t num_blocks, b, blk_size;
  uint8_t const* src;
  uint8_t* dst;

  if (t == 0) ss = strm_desc[stripe_id][stream_id];
  __syncthreads();

  num_blocks = (ss.stream_size > 0) ? (ss.stream_size - 1) / comp_blk_size + 1 : 0;
  dst        = compressed_bfr.data() + ss.bfr_offset;
  b          = 0;
  do {
    if (t == 0) {
      auto const src_len =
        min(comp_blk_size, ss.stream_size - min(b * comp_blk_size, ss.stream_size));
      auto dst_len = (results[ss.first_block + b].status == codec_status::SUCCESS)
                       ? results[ss.first_block + b].bytes_written
                       : src_len;
      uint32_t blk_size24{};
      // Only use the compressed block if it's smaller than the uncompressed
      // If compression failed, dst_len == src_len, so the uncompressed block will be used
      if (src_len < dst_len) {
        // Copy from uncompressed source
        src                                       = inputs[ss.first_block + b].data();
        results[ss.first_block + b].bytes_written = src_len;
        dst_len                                   = src_len;
        blk_size24                                = dst_len * 2 + 1;
      } else {
        // Compressed block
        src        = outputs[ss.first_block + b].data();
        blk_size24 = dst_len * 2 + 0;
      }
      dst[0]     = static_cast<uint8_t>(blk_size24 >> 0);
      dst[1]     = static_cast<uint8_t>(blk_size24 >> 8);
      dst[2]     = static_cast<uint8_t>(blk_size24 >> 16);
      comp_src_g = src;
      comp_len_g = dst_len;
    }
    __syncthreads();
    src      = comp_src_g;
    blk_size = comp_len_g;
    dst += 3;  // skip over length written by thread0
    if (src != dst) {
      for (uint32_t i = 0; i < blk_size; i += 1024) {
        uint8_t v = (i + t < blk_size) ? src[i + t] : 0;
        __syncthreads();
        if (i + t < blk_size) { dst[i + t] = v; }
      }
    }
    dst += blk_size;
    __syncthreads();
  } while (++b < num_blocks);
  // Update stripe stream with the compressed size
  if (t == 0) {
    strm_desc[stripe_id][stream_id].stream_size =
      static_cast<uint32_t>(dst - (compressed_bfr.data() + ss.bfr_offset));
  }
}

// Holds a non-owning view of a decimal column's element sizes
struct decimal_column_element_sizes {
  uint32_t col_idx;
  device_span<uint32_t> sizes;
};

// Converts sizes of individual decimal elements to offsets within each row group
// Conversion is done in-place
template <int block_size>
CUDF_KERNEL void decimal_sizes_to_offsets_kernel(device_2dspan<rowgroup_rows const> rg_bounds,
                                                 device_span<decimal_column_element_sizes> sizes)
{
  using block_scan = cub::BlockScan<uint32_t, block_size>;
  __shared__ typename block_scan::TempStorage scan_storage;
  int const t = threadIdx.x;

  auto const col_id          = blockIdx.x / rg_bounds.size().first;
  auto const rg_id           = blockIdx.x % rg_bounds.size().first;
  auto const& col_elem_sizes = sizes[col_id];
  auto const& row_group      = rg_bounds[rg_id][col_elem_sizes.col_idx];
  auto const elem_sizes      = col_elem_sizes.sizes.data() + row_group.begin;

  uint32_t initial_value = 0;
  // Do a series of block sums, storing results in the array as we go
  for (int64_t pos = 0; pos < row_group.size(); pos += block_size) {
    auto const tidx    = pos + t;
    auto tval          = tidx < row_group.size() ? elem_sizes[tidx] : 0u;
    uint32_t block_sum = 0;
    block_scan(scan_storage).InclusiveSum(tval, tval, block_sum);
    if (tidx < row_group.size()) { elem_sizes[tidx] = tval + initial_value; }
    initial_value += block_sum;
  }
}

void encode_orc_column_data(device_2dspan<encoder_chunk const> chunks,
                            device_2dspan<encoder_chunk_streams> streams,
                            rmm::cuda_stream_view stream)
{
  auto const num_blocks = chunks.size().first * chunks.size().second;
  encode_column_data_kernel<encode_block_size>
    <<<num_blocks, encode_block_size, 0, stream.value()>>>(chunks, streams);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void encode_stripe_dictionaries(stripe_dictionary const* stripes,
                                device_span<orc_column_device_view const> columns,
                                device_2dspan<encoder_chunk const> chunks,
                                size_type num_string_columns,
                                size_type num_stripes,
                                device_2dspan<encoder_chunk_streams> enc_streams,
                                rmm::cuda_stream_view stream)
{
  constexpr int block_size = 512;  // 512 threads per dictionary
  dim3 dim_grid(num_string_columns * num_stripes, 2);
  encode_string_dictionaries_kernel<block_size>
    <<<dim_grid, block_size, 0, stream.value()>>>(stripes, columns, chunks, enc_streams);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void compact_orc_data_streams(device_2dspan<stripe_stream> strm_desc,
                              device_2dspan<encoder_chunk_streams> enc_streams,
                              rmm::cuda_stream_view stream)
{
  auto const num_rowgroups = enc_streams.size().second;
  auto const num_streams   = strm_desc.size().second;
  auto const num_stripes   = strm_desc.size().first;
  auto const num_chunks    = num_rowgroups * num_streams;
  auto srcs                = cudf::detail::make_zeroed_device_uvector_async<uint8_t*>(
    num_chunks, stream, cudf::get_current_device_resource_ref());
  auto dsts = cudf::detail::make_zeroed_device_uvector_async<uint8_t*>(
    num_chunks, stream, cudf::get_current_device_resource_ref());
  auto lengths = cudf::detail::make_zeroed_device_uvector_async<size_t>(
    num_chunks, stream, cudf::get_current_device_resource_ref());

  auto const num_blocks =
    cudf::util::div_rounding_up_unsafe(num_stripes, compact_streams_block_size) *
    strm_desc.size().second;
  init_batched_memcpy_kernel<<<num_blocks, compact_streams_block_size, 0, stream.value()>>>(
    strm_desc, enc_streams, srcs, dsts, lengths);
  CUDF_CUDA_TRY(cudaGetLastError());

  // Copy streams in a batched manner.
  cudf::detail::batched_memcpy_async(
    srcs.begin(), dsts.begin(), lengths.begin(), lengths.size(), stream);
}

std::optional<writer_compression_statistics> compress_orc_data_streams(
  device_span<uint8_t> compressed_data,
  uint32_t num_compressed_blocks,
  compression_type compression,
  uint32_t comp_blk_size,
  uint32_t max_comp_blk_size,
  uint32_t comp_block_align,
  bool collect_statistics,
  device_2dspan<stripe_stream> strm_desc,
  device_2dspan<encoder_chunk_streams> enc_streams,
  device_span<codec_exec_result> comp_res,
  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<device_span<uint8_t const>> comp_in(num_compressed_blocks, stream);
  rmm::device_uvector<device_span<uint8_t>> comp_out(num_compressed_blocks, stream);

  size_t const num_blocks = strm_desc.size().first * strm_desc.size().second;
  init_compression_blocks_kernel<<<num_blocks, 256, 0, stream.value()>>>(strm_desc,
                                                                         enc_streams,
                                                                         comp_in,
                                                                         comp_out,
                                                                         comp_res,
                                                                         compressed_data,
                                                                         comp_blk_size,
                                                                         max_comp_blk_size,
                                                                         comp_block_align);
  CUDF_CUDA_TRY(cudaGetLastError());

  cudf::io::detail::compress(compression, comp_in, comp_out, comp_res, stream);

  compact_compressed_blocks_kernel<<<num_blocks, 1024, 0, stream.value()>>>(
    strm_desc, comp_in, comp_out, comp_res, compressed_data, comp_blk_size, max_comp_blk_size);
  CUDF_CUDA_TRY(cudaGetLastError());

  if (collect_statistics) {
    return cudf::io::detail::collect_compression_statistics(comp_in, comp_res, stream);
  } else {
    return std::nullopt;
  }
}

void decimal_sizes_to_offsets(device_2dspan<rowgroup_rows const> rg_bounds,
                              std::map<uint32_t, rmm::device_uvector<uint32_t>>& elem_sizes,
                              rmm::cuda_stream_view stream)
{
  if (rg_bounds.count() == 0) return;

  // Convert map to a vector of views of the `elem_sizes` device buffers
  auto h_sizes =
    cudf::detail::make_empty_host_vector<decimal_column_element_sizes>(elem_sizes.size(), stream);
  std::transform(elem_sizes.begin(), elem_sizes.end(), std::back_inserter(h_sizes), [](auto& p) {
    return decimal_column_element_sizes{p.first, p.second};
  });

  // Copy the vector of views to the device so that we can pass it to the kernel
  auto d_sizes = cudf::detail::make_device_uvector_async<decimal_column_element_sizes>(
    h_sizes, stream, cudf::get_current_device_resource_ref());

  constexpr int block_size = 256;
  // number of rowgroups * number of decimal columns
  auto const num_blocks = elem_sizes.size() * rg_bounds.size().first;
  decimal_sizes_to_offsets_kernel<block_size>
    <<<num_blocks, block_size, 0, stream.value()>>>(rg_bounds, d_sizes);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::orc::detail
