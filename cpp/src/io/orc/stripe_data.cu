/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/block_utils.cuh"
#include "io/utilities/column_buffer.hpp"
#include "orc_gpu.hpp"

#include <cudf/io/orc_types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>

namespace cudf::io::orc::detail {

using cudf::io::detail::string_index_pair;

// Must be able to handle 512x 8-byte values. These values are base 128 encoded
// so 8 byte value is expanded to 10 bytes.
constexpr int bytestream_buffer_size = 512 * 8 * 2;
constexpr int bytestream_buffer_mask = (bytestream_buffer_size - 1) >> 2;

// TODO: Should be more efficient with 512 threads per block and circular queue for values
constexpr int num_warps  = 32;
constexpr int block_size = 32 * num_warps;
// Add some margin to look ahead to future rows in case there are many zeroes
constexpr int row_decoder_buffer_size = block_size + 128;
inline __device__ uint8_t is_rlev1(uint8_t encoding_mode) { return encoding_mode < DIRECT_V2; }

inline __device__ uint8_t is_dictionary(uint8_t encoding_mode) { return encoding_mode & 1; }

struct orc_bytestream_s {
  uint8_t const* base;
  uint32_t pos;
  uint32_t len;
  uint32_t fill_pos;
  uint32_t fill_count;
  union {
    uint8_t u8[bytestream_buffer_size];
    uint32_t u32[bytestream_buffer_size >> 2];
    uint2 u64[bytestream_buffer_size >> 3];
  } buf;
};

struct orc_rlev1_state_s {
  uint32_t num_runs;
  uint32_t num_vals;
  int32_t run_data[num_warps * 12];  // (delta << 24) | (count << 16) | (first_val)
};

struct orc_rlev2_state_s {
  uint32_t num_runs;
  uint32_t num_vals;
  union {
    uint32_t u32[num_warps];
    uint64_t u64[num_warps];
  } baseval;
  uint16_t m2_pw_byte3[num_warps];
  int64_t delta[num_warps];
  uint16_t runs_loc[block_size];
};

struct orc_byte_rle_state_s {
  uint32_t num_runs;
  uint32_t num_vals;
  uint32_t runs_loc[num_warps];
  uint32_t runs_pos[num_warps];
};

struct orc_rowdec_state_s {
  uint32_t nz_count;
  uint32_t row[row_decoder_buffer_size];  // 0=skip, >0: row position relative to cur_row
};

struct orc_strdict_state_s {
  dictionary_entry* local_dict;
  uint32_t dict_pos;
  uint32_t dict_len;
};

struct orc_datadec_state_s {
  int64_t cur_row;          // starting row of current batch
  int64_t end_row;          // ending row of this chunk (start_row + num_rows)
  uint32_t max_vals;        // max # of non-zero values to decode in this batch
  uint32_t nrows;           // # of rows in current batch (up to block_size)
  uint32_t buffered_count;  // number of buffered values in the secondary data stream
  duration_s tz_epoch;      // orc_ut_epoch - ut_offset
  row_group index;
};

struct orcdec_state_s {
  column_desc chunk;
  orc_bytestream_s bs;
  orc_bytestream_s bs2;
  int is_string;
  int64_t num_child_rows;
  union {
    orc_strdict_state_s dict;
    uint32_t nulls_desc_row;  // number of rows processed for nulls.
    orc_datadec_state_s data;
  } top;
  union {
    orc_rlev1_state_s rlev1;
    orc_rlev2_state_s rlev2;
    orc_byte_rle_state_s rle8;
    orc_rowdec_state_s rowdec;
  } u;
  union values {
    uint8_t u8[block_size * 16];
    uint32_t u32[block_size * 4];
    int32_t i32[block_size * 4];
    uint64_t u64[block_size * 2];
    int64_t i64[block_size * 2];
    double f64[block_size * 2];
    __int128_t i128[block_size];
    __uint128_t u128[block_size];
  } vals;
};

/**
 * @brief Manage caching of the first run of TIMESTAMP's DATA stream for a row group.
 *
 * This class is used to address a special case, where the first run of the DATA stream spans two
 * adjacent row groups and its length is greater than the maximum length allowed to be consumed.
 * This limit is imposed by the decoder when processing the SECONDARY stream. This class shall be
 * instantiated in the shared memory, and be used to cache the DATA stream with a decoded data type
 * of `int64_t`. As an optimization, the actual cache is implemented in the cache_helper class as a
 * local variable and does not reside in the shared memory.
 */
class run_cache_manager {
 private:
  enum class status : uint8_t {
    DISABLED,  ///< Run cache manager is disabled. No caching will be performed. If the special case
               ///< happens, the run cache manager will be set to this status after the cache read
               ///< is completed. This status also applies when the special case does not happen.
    CAN_WRITE_TO_CACHE,  ///< Run cache manager is ready for write. If the special case happens, the
                         ///< run cache manager will be set to this status.
    CAN_READ_FROM_CACHE,  ///< Run cache manager is ready for read. If the special case happens, the
                          ///< run cache manager will be set to this status after the cache write is
                          ///< completed.
  };

 public:
  /**
   * @brief Initialize the run cache manager.
   *
   * @param[in] s ORC decoder state.
   */
  __device__ void initialize(orcdec_state_s* s)
  {
    _status          = (s->top.data.index.run_pos[CI_DATA2] > 0 and s->chunk.type_kind == TIMESTAMP)
                         ? status::CAN_WRITE_TO_CACHE
                         : status::DISABLED;
    _reusable_length = 0;
    _run_length      = 0;
  }

 private:
  status _status;  ///< The status of the run cache manager.
  uint32_t
    _reusable_length;  ///< The number of data to be cached and reused later. For example, if a run
                       ///< has a length of 512 but the maximum length allowed to be consumed is
                       ///< capped at 162, then 350 (512-162) data will be cached.
  uint32_t _run_length;  ///< The length of the run, 512 in the above example.
  friend class cache_helper;
};

/**
 * @brief Helper class to help run_cache_manager cache the first run of TIMESTAMP's DATA stream for
 * a row group.
 *
 * The run_cache_manager is intended to be stored in the shared memory, whereas the actual cache is
 * in the local storage (as an optimization). If a function is to use run_cache_manager, both the
 * manager and the cache objects need to be passed. This class is introduced to simplify the
 * function call, so that only a single cache_helper object needs to be passed. To that end, public
 * methods originally belonging to run_cache_manager have been moved to this class.
 */
class cache_helper {
 public:
  /**
   * @brief Constructor.
   *
   * @param[in] run_cache_manager_inst An instance of run_cache_manager.
   */
  __device__ explicit cache_helper(run_cache_manager& run_cache_manager_inst)
    : _manager(run_cache_manager_inst)
  {
  }

  /**
   * @brief Set the reusable length object.
   *
   * @param[in] run_length The length of the first run (spanning two adjacent row groups) of the
   * DATA stream.
   * @param[in] max_length The maximum length allowed to be consumed. This limit is imposed
   * by the decoder when processing the SECONDARY stream.
   */
  __device__ void set_reusable_length(uint32_t run_length, uint32_t max_length)
  {
    if (_manager._status == run_cache_manager::status::CAN_WRITE_TO_CACHE) {
      _manager._run_length = run_length;
      _manager._reusable_length =
        (_manager._run_length > max_length) ? (_manager._run_length - max_length) : 0;
    }
  }

  /**
   * @brief Adjust the maximum length allowed to be consumed when the length of the first run is
   * greater than it.
   *
   * @param[in] max_length The maximum length allowed to be consumed for the DATA stream.
   * @return A new maximum length.
   */
  [[nodiscard]] __device__ uint32_t adjust_max_length(uint32_t max_length)
  {
    auto new_max_length{max_length};
    if (_manager._status == run_cache_manager::status::CAN_READ_FROM_CACHE) {
      new_max_length -= _manager._reusable_length;
    }
    return new_max_length;
  }

  /**
   * @brief Copy the excess data from the intermediate buffer for the DATA stream to the cache.
   *
   * @param[in] src Intermediate buffer for the DATA stream.
   */
  __device__ void write_to_cache(int64_t* src)
  {
    if (_manager._status != run_cache_manager::status::CAN_WRITE_TO_CACHE) { return; }

    auto const tid = threadIdx.x;

    __syncthreads();

    // All threads in the block always take a uniform code path for the following branches.
    // _reusable_length ranges between [0, 512].
    if (_manager._reusable_length > 0) {
      auto const length_to_skip = _manager._run_length - _manager._reusable_length;
      if (tid < _manager._reusable_length) {
        auto const src_idx = tid + length_to_skip;
        _storage           = src[src_idx];
      }
      if (tid == 0) { _manager._status = run_cache_manager::status::CAN_READ_FROM_CACHE; }
    } else {
      if (tid == 0) { _manager._status = run_cache_manager::status::DISABLED; }
    }

    __syncthreads();
  }

  /**
   * @brief Copy the cached data to the intermediate buffer for the DATA stream.
   *
   * @param[in,out] dst Intermediate buffer for the DATA stream.
   * @param[in,out] rle Run length decoder state object.
   */
  __device__ void read_from_cache(int64_t* dst, orc_rlev2_state_s* rle)
  {
    if (_manager._status != run_cache_manager::status::CAN_READ_FROM_CACHE) { return; }

    auto const tid = threadIdx.x;

    // First, shift the data up
    auto const dst_idx = tid + _manager._reusable_length;
    auto const v       = (dst_idx < rle->num_vals + _manager._reusable_length) ? dst[tid] : 0;
    __syncthreads();

    if (dst_idx < rle->num_vals + _manager._reusable_length) { dst[dst_idx] = v; }
    __syncthreads();

    // Second, insert the cached data
    if (tid < _manager._reusable_length) { dst[tid] = _storage; }
    __syncthreads();

    if (tid == 0) {
      // Disable the run cache manager, since cache write-and-read happens at most once per row
      // group.
      _manager._status = run_cache_manager::status::DISABLED;
      rle->num_vals += _manager._reusable_length;
    }

    __syncthreads();
  }

 private:
  run_cache_manager& _manager;  ///< An instance of run_cache_manager.
  int64_t _storage;             ///< Per-thread cache storage.
};

/**
 * @brief Initializes byte stream, modifying length and start position to keep the read pointer
 * 8-byte aligned.
 *
 * Assumes that the address range [start_address & ~7, (start_address + len - 1) | 7]
 * is valid.
 *
 * @param[in,out] bs Byte stream input
 * @param[in] base Pointer to raw byte stream data
 * @param[in] len Stream length in bytes
 */
static __device__ void bytestream_init(orc_bytestream_s* bs, uint8_t const* base, uint32_t len)
{
  uint32_t pos   = (len > 0) ? static_cast<uint32_t>(7 & reinterpret_cast<size_t>(base)) : 0;
  bs->base       = base - pos;
  bs->pos        = pos;
  bs->len        = (len + pos + 7) & ~7;
  bs->fill_pos   = 0;
  bs->fill_count = min(bs->len, bytestream_buffer_size) >> 3;
}

/**
 * @brief Increment the read position, returns number of 64-bit slots to fill
 *
 * @param[in] bs Byte stream input
 * @param[in] bytes_consumed Number of bytes that were consumed
 */
static __device__ void bytestream_flush_bytes(orc_bytestream_s* bs, uint32_t bytes_consumed)
{
  uint32_t pos     = bs->pos;
  uint32_t len     = bs->len;
  uint32_t pos_new = min(pos + bytes_consumed, len);
  bs->pos          = pos_new;
  pos              = min(pos + bytestream_buffer_size, len);
  pos_new          = min(pos_new + bytestream_buffer_size, len);
  bs->fill_pos     = pos;
  bs->fill_count   = (pos_new >> 3) - (pos >> 3);
}

/**
 * @brief Refill the byte stream buffer
 *
 * @param[in] bs Byte stream input
 * @param[in] t thread id
 */
static __device__ void bytestream_fill(orc_bytestream_s* bs, int t)
{
  auto const count = bs->fill_count;
  if (t < count) {
    auto const pos8 = (bs->fill_pos >> 3) + t;
    memcpy(&bs->buf.u64[pos8 & ((bytestream_buffer_size >> 3) - 1)],
           &bs->base[pos8 * sizeof(uint2)],
           sizeof(uint2));
  }
}

/**
 * @brief Read a byte from the byte stream (byte aligned)
 *
 * @param[in] bs Byte stream input
 * @param[in] pos Position in byte stream
 * @return byte
 */
inline __device__ uint8_t bytestream_readbyte(orc_bytestream_s* bs, int pos)
{
  return bs->buf.u8[pos & (bytestream_buffer_size - 1)];
}

/**
 * @brief Read 32 bits from a byte stream (little endian, byte aligned)
 *
 * @param[in] bs Byte stream input
 * @param[in] pos Position in byte stream
 * @result bits
 */
inline __device__ uint32_t bytestream_readu32(orc_bytestream_s* bs, int pos)
{
  uint32_t a = bs->buf.u32[(pos & (bytestream_buffer_size - 1)) >> 2];
  uint32_t b = bs->buf.u32[((pos + 4) & (bytestream_buffer_size - 1)) >> 2];
  return __funnelshift_r(a, b, (pos & 3) * 8);
}

/**
 * @brief Read 64 bits from a byte stream (little endian, byte aligned)
 *
 * @param[in] bs Byte stream input
 * @param[in] pos Position in byte stream
 * @return bits
 */
inline __device__ uint64_t bytestream_readu64(orc_bytestream_s* bs, int pos)
{
  uint32_t a    = bs->buf.u32[(pos & (bytestream_buffer_size - 1)) >> 2];
  uint32_t b    = bs->buf.u32[((pos + 4) & (bytestream_buffer_size - 1)) >> 2];
  uint32_t c    = bs->buf.u32[((pos + 8) & (bytestream_buffer_size - 1)) >> 2];
  uint32_t lo32 = __funnelshift_r(a, b, (pos & 3) * 8);
  uint32_t hi32 = __funnelshift_r(b, c, (pos & 3) * 8);
  uint64_t v    = hi32;
  v <<= 32;
  v |= lo32;
  return v;
}

/**
 * @brief Read up to 32-bits from a byte stream (big endian)
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @return decoded value
 */
inline __device__ uint32_t bytestream_readbits(orc_bytestream_s* bs, int bitpos, uint32_t numbits)
{
  int idx    = bitpos >> 5;
  uint32_t a = __byte_perm(bs->buf.u32[(idx + 0) & bytestream_buffer_mask], 0, 0x0123);
  uint32_t b = __byte_perm(bs->buf.u32[(idx + 1) & bytestream_buffer_mask], 0, 0x0123);
  return __funnelshift_l(b, a, bitpos & 0x1f) >> (32 - numbits);
}

/**
 * @brief Read up to 64-bits from a byte stream (big endian)
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @return decoded value
 */
inline __device__ uint64_t bytestream_readbits64(orc_bytestream_s* bs, int bitpos, uint32_t numbits)
{
  int idx       = bitpos >> 5;
  uint32_t a    = __byte_perm(bs->buf.u32[(idx + 0) & bytestream_buffer_mask], 0, 0x0123);
  uint32_t b    = __byte_perm(bs->buf.u32[(idx + 1) & bytestream_buffer_mask], 0, 0x0123);
  uint32_t c    = __byte_perm(bs->buf.u32[(idx + 2) & bytestream_buffer_mask], 0, 0x0123);
  uint32_t hi32 = __funnelshift_l(b, a, bitpos & 0x1f);
  uint32_t lo32 = __funnelshift_l(c, b, bitpos & 0x1f);
  uint64_t v    = hi32;
  v <<= 32;
  v |= lo32;
  v >>= (64 - numbits);
  return v;
}

/**
 * @brief Decode a big-endian unsigned 32-bit value
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @param[out] result decoded value
 */
inline __device__ void bytestream_readbe(orc_bytestream_s* bs,
                                         int bitpos,
                                         uint32_t numbits,
                                         uint32_t& result)
{
  result = bytestream_readbits(bs, bitpos, numbits);
}

/**
 * @brief Decode a big-endian signed 32-bit value
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @param[out] result decoded value
 */
inline __device__ void bytestream_readbe(orc_bytestream_s* bs,
                                         int bitpos,
                                         uint32_t numbits,
                                         int32_t& result)
{
  uint32_t u = bytestream_readbits(bs, bitpos, numbits);
  result     = (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
}

/**
 * @brief Decode a big-endian unsigned 64-bit value
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @param[out] result decoded value
 */
inline __device__ void bytestream_readbe(orc_bytestream_s* bs,
                                         int bitpos,
                                         uint32_t numbits,
                                         uint64_t& result)
{
  result = bytestream_readbits64(bs, bitpos, numbits);
}

/**
 * @brief Decode a big-endian signed 64-bit value
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @param[out] result decoded value
 */
inline __device__ void bytestream_readbe(orc_bytestream_s* bs,
                                         int bitpos,
                                         uint32_t numbits,
                                         int64_t& result)
{
  uint64_t u = bytestream_readbits64(bs, bitpos, numbits);
  result     = (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
}

/**
 * @brief Return the length of a base-128 varint
 *
 * @param[in] bs Byte stream input
 * @param[in] pos Position in circular byte stream buffer
 * @return length of varint in bytes
 */
template <class T>
inline __device__ uint32_t varint_length(orc_bytestream_s* bs, int pos)
{
  if (bytestream_readbyte(bs, pos) > 0x7f) {
    uint32_t next32 = bytestream_readu32(bs, pos + 1);
    uint32_t zbit   = __ffs((~next32) & 0x8080'8080);
    if (sizeof(T) <= 4 || zbit) {
      return 1 + (zbit >> 3);  // up to 5x7 bits
    } else {
      next32 = bytestream_readu32(bs, pos + 5);
      zbit   = __ffs((~next32) & 0x8080'8080);
      if (zbit) {
        return 5 + (zbit >> 3);  // up to 9x7 bits
      } else if ((sizeof(T) <= 8) || (bytestream_readbyte(bs, pos + 9) <= 0x7f)) {
        return 10;  // up to 70 bits
      } else {
        uint64_t next64 = bytestream_readu64(bs, pos + 10);
        zbit            = __ffsll((~next64) & 0x8080'8080'8080'8080ull);
        if (zbit) {
          return 10 + (zbit >> 3);  // Up to 18x7 bits (126)
        } else {
          return 19;  // Up to 19x7 bits (133)
        }
      }
    }
  } else {
    return 1;
  }
}

/**
 * @brief Decodes a base-128 varint
 *
 * @param[in] bs Byte stream input
 * @param[in] pos Position in circular byte stream buffer
 * @param[in] result Unpacked value
 * @return new position in byte stream buffer
 */
template <class T>
inline __device__ int decode_base128_varint(orc_bytestream_s* bs, int pos, T& result)
{
  uint32_t v = bytestream_readbyte(bs, pos++);
  if (v > 0x7f) {
    uint32_t b = bytestream_readbyte(bs, pos++);
    v          = (v & 0x7f) | (b << 7);
    if (b > 0x7f) {
      b = bytestream_readbyte(bs, pos++);
      v = (v & 0x3fff) | (b << 14);
      if (b > 0x7f) {
        b = bytestream_readbyte(bs, pos++);
        v = (v & 0x1f'ffff) | (b << 21);
        if (b > 0x7f) {
          b = bytestream_readbyte(bs, pos++);
          v = (v & 0x0fff'ffff) | (b << 28);
          if constexpr (sizeof(T) > 4) {
            uint32_t lo = v;
            uint64_t hi;
            v = b >> 4;
            if (b > 0x7f) {
              b = bytestream_readbyte(bs, pos++);
              v = (v & 7) | (b << 3);
              if (b > 0x7f) {
                b = bytestream_readbyte(bs, pos++);
                v = (v & 0x3ff) | (b << 10);
                if (b > 0x7f) {
                  b = bytestream_readbyte(bs, pos++);
                  v = (v & 0x1'ffff) | (b << 17);
                  if (b > 0x7f) {
                    b = bytestream_readbyte(bs, pos++);
                    v = (v & 0xff'ffff) | (b << 24);
                    if (b > 0x7f) {
                      pos++;  // last bit is redundant (extra byte implies bit63 is 1)
                    }
                  }
                }
              }
            }
            hi = v;
            hi <<= 32;
            result = hi | lo;
            return pos;
          }
        }
      }
    }
  }
  result = v;
  return pos;
}

/**
 * @brief Decodes a signed int128 encoded as base-128 varint (used for decimals)
 */
inline __device__ __int128_t decode_varint128(orc_bytestream_s* bs, int pos)
{
  auto byte                  = bytestream_readbyte(bs, pos++);
  __int128_t const sign_mask = -(int32_t)(byte & 1);
  __int128_t value           = (byte >> 1) & 0x3f;
  uint32_t bitpos            = 6;
  while (byte & 0x80 && bitpos < 128) {
    byte = bytestream_readbyte(bs, pos++);
    value |= ((__uint128_t)(byte & 0x7f)) << bitpos;
    bitpos += 7;
  }
  return value ^ sign_mask;
}

/**
 * @brief Decodes an unsigned 32-bit varint
 */
inline __device__ int decode_varint(orc_bytestream_s* bs, int pos, uint32_t& result)
{
  uint32_t u;
  pos    = decode_base128_varint<uint32_t>(bs, pos, u);
  result = u;
  return pos;
}

/**
 * @brief Decodes an unsigned 64-bit varint
 */
inline __device__ int decode_varint(orc_bytestream_s* bs, int pos, uint64_t& result)
{
  uint64_t u;
  pos    = decode_base128_varint<uint64_t>(bs, pos, u);
  result = u;
  return pos;
}

/**
 * @brief Signed version of 32-bit decode_varint
 */
inline __device__ int decode_varint(orc_bytestream_s* bs, int pos, int32_t& result)
{
  uint32_t u;
  pos    = decode_base128_varint<uint32_t>(bs, pos, u);
  result = (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
  return pos;
}

/**
 * @brief Signed version of 64-bit decode_varint
 */
inline __device__ int decode_varint(orc_bytestream_s* bs, int pos, int64_t& result)
{
  uint64_t u;
  pos    = decode_base128_varint<uint64_t>(bs, pos, u);
  result = (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
  return pos;
}

/**
 * @brief In-place conversion from lengths to positions
 *
 * @param[in] vals input values
 * @param[in] numvals number of values
 * @param[in] t thread id
 *
 * @return number of values decoded
 */
template <class T>
inline __device__ void lengths_to_positions(T* vals, uint32_t numvals, unsigned int t)
{
  for (uint32_t n = 1; n < numvals; n <<= 1) {
    __syncthreads();
    if ((t & n) && (t < numvals)) vals[t] += vals[(t & ~n) | (n - 1)];
  }
}

/**
 * @brief ORC Integer RLEv1 decoding
 *
 * @param[in] bs input byte stream
 * @param[in] rle RLE state
 * @param[in] vals buffer for output values (uint32_t, int32_t, uint64_t or int64_t)
 * @param[in] maxvals maximum number of values to decode
 * @param[in] t thread id
 *
 * @return number of values decoded
 */
template <class T>
static __device__ uint32_t
integer_rlev1(orc_bytestream_s* bs, orc_rlev1_state_s* rle, T* vals, uint32_t maxvals, int t)
{
  uint32_t numvals, numruns;
  if (t == 0) {
    uint32_t maxpos  = min(bs->len, bs->pos + (bytestream_buffer_size - 8u));
    uint32_t lastpos = bs->pos;
    numvals = numruns = 0;
    // Find the length and start location of each run
    while (numvals < maxvals && numruns < num_warps * 12) {
      uint32_t pos = lastpos;
      uint32_t n   = bytestream_readbyte(bs, pos++);
      if (n <= 0x7f) {
        // Run
        int32_t delta;
        n = n + 3;
        if (numvals + n > maxvals) break;
        delta         = bytestream_readbyte(bs, pos++);
        vals[numvals] = pos & 0xffff;
        pos += varint_length<T>(bs, pos);
        if (pos > maxpos) break;
        rle->run_data[numruns++] = (delta << 24) | (n << 16) | numvals;
        numvals += n;
      } else {
        // Literals
        uint32_t i;
        n = 0x100 - n;
        if (numvals + n > maxvals) break;
        i = 0;
        do {
          vals[numvals + i] = pos & 0xffff;
          pos += varint_length<T>(bs, pos);
        } while (++i < n);
        if (pos > maxpos) break;
        numvals += n;
      }
      lastpos = pos;
    }
    rle->num_runs = numruns;
    rle->num_vals = numvals;
    bytestream_flush_bytes(bs, lastpos - bs->pos);
  }
  __syncthreads();
  // Expand the runs
  numruns = rle->num_runs;
  if (numruns > 0) {
    int r  = t >> 5;
    int tr = t & 0x1f;
    for (uint32_t run = r; run < numruns; run += num_warps) {
      int32_t run_data = rle->run_data[run];
      int n            = (run_data >> 16) & 0xff;
      int delta        = run_data >> 24;
      uint32_t base    = run_data & 0x3ff;
      uint32_t pos     = vals[base] & 0xffff;
      for (int i = 1 + tr; i < n; i += 32) {
        vals[base + i] = ((delta * i) << 16) | pos;
      }
    }
    __syncthreads();
  }
  numvals = rle->num_vals;
  // Decode individual 32-bit varints
  if (t < numvals) {
    int32_t pos   = vals[t];
    int32_t delta = pos >> 16;
    T v;
    decode_varint(bs, pos, v);
    vals[t] = v + delta;
  }
  __syncthreads();
  return numvals;
}

/**
 * @brief Maps the RLEv2 5-bit length code to 6-bit length
 */
static __device__ const __constant__ uint8_t kRLEv2_W[32] = {
  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
  17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 30, 32, 40, 48, 56, 64};

/**
 * @brief Maps the RLEv2 patch size (pw + pgw) to number of bits
 *
 * Patch size (in bits) is only allowed to be from the below set. If `pw + pgw == 34` then the size
 * of the patch in the file is the smallest size in the set that can fit 34 bits i.e.
 * `ClosestFixedBitsMap[34] == 40`
 *
 * @see https://github.com/apache/orc/commit/9faf7f5147a7bc69
 */
static __device__ const __constant__ uint8_t ClosestFixedBitsMap[65] = {
  1,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
  22, 23, 24, 26, 26, 28, 28, 30, 30, 32, 32, 40, 40, 40, 40, 40, 40, 40, 40, 48, 48, 48,
  48, 48, 48, 48, 48, 56, 56, 56, 56, 56, 56, 56, 56, 64, 64, 64, 64, 64, 64, 64, 64};

/**
 * @brief ORC Integer RLEv2 decoding
 *
 * @param[in] bs input byte stream
 * @param[in] rle RLE state
 * @param[in] vals buffer for output values (uint32_t, int32_t, uint64_t or int64_t)
 * @param[in] maxvals maximum number of values to decode
 * @param[in] t thread id
 * @param[in] has_buffered_values If true, means there are already buffered values
 * @param[in] cache_helper_inst If non-null, the run cache manager will be used to manage
 * caching of the first run of the DATA stream.
 *
 * @return number of values decoded
 */
template <class T>
static __device__ uint32_t integer_rlev2(orc_bytestream_s* bs,
                                         orc_rlev2_state_s* rle,
                                         T* vals,
                                         uint32_t maxvals,
                                         int t,
                                         bool has_buffered_values        = false,
                                         cache_helper* cache_helper_inst = nullptr)
{
  if (t == 0) {
    if (cache_helper_inst != nullptr) { maxvals = cache_helper_inst->adjust_max_length(maxvals); }
    uint32_t maxpos  = min(bs->len, bs->pos + (bytestream_buffer_size - 8u));
    uint32_t lastpos = bs->pos;
    auto numvals     = 0;
    auto numruns     = 0;
    // Find the length and start location of each run
    while (numvals < maxvals) {
      uint32_t pos   = lastpos;
      uint32_t byte0 = bytestream_readbyte(bs, pos++);
      uint32_t n, l;
      int mode               = byte0 >> 6;
      rle->runs_loc[numruns] = numvals;
      vals[numvals]          = lastpos;
      if (mode == 0) {
        // 00lllnnn: short repeat encoding
        l = 1 + ((byte0 >> 3) & 7);  // 1 to 8 bytes
        n = 3 + (byte0 & 7);         // 3 to 10 values
      } else {
        l = kRLEv2_W[(byte0 >> 1) & 0x1f];
        n = 1 + ((byte0 & 1) << 8) + bytestream_readbyte(bs, pos++);
        if (mode == 1) {
          // 01wwwwwn.nnnnnnnn: direct encoding
          l = (l * n + 7) >> 3;
        } else if (mode == 2) {
          // 10wwwwwn.nnnnnnnn.xxxxxxxx.yyyyyyyy: patched base encoding
          uint32_t byte2      = bytestream_readbyte(bs, pos++);
          uint32_t byte3      = bytestream_readbyte(bs, pos++);
          uint32_t bw         = 1 + (byte2 >> 5);        // base value width, 1 to 8 bytes
          uint32_t pw         = kRLEv2_W[byte2 & 0x1f];  // patch width, 1 to 64 bits
          uint32_t pgw        = 1 + (byte3 >> 5);        // patch gap width, 1 to 8 bits
          uint32_t pgw_pw_len = ClosestFixedBitsMap[min(pw + pgw, 64u)];  // ceiled patch width
          uint32_t pll        = byte3 & 0x1f;                             // patch list length
          l                   = (l * n + 7) >> 3;
          l += bw;
          l += (pll * (pgw_pw_len) + 7) >> 3;
        } else {
          // 11wwwwwn.nnnnnnnn.<base>.<delta>: delta encoding
          uint32_t deltapos = varint_length<T>(bs, pos);
          deltapos += varint_length<T>(bs, pos + deltapos);
          l = (l > 1 && n > 2) ? (l * (n - 2) + 7) >> 3 : 0;
          l += deltapos;
        }
      }

      if (cache_helper_inst != nullptr) { cache_helper_inst->set_reusable_length(n, maxvals); }

      if ((numvals != 0) and (numvals + n > maxvals)) break;
      // case where there are buffered values and can't consume a whole chunk
      // from decoded values, so skip adding any more to buffer, work on buffered values and then
      // start fresh in next iteration with empty buffer.
      if ((numvals == 0) and (n > maxvals) and (has_buffered_values)) break;

      pos += l;
      if (pos > maxpos) break;
      ((numvals == 0) and (n > maxvals)) ? numvals = maxvals : numvals += n;
      lastpos = pos;
      numruns++;
    }
    rle->num_vals = numvals;
    rle->num_runs = numruns;
    bytestream_flush_bytes(bs, lastpos - bs->pos);
  }
  __syncthreads();
  // Process the runs, 1 warp per run
  auto const numruns = rle->num_runs;
  auto const r       = t >> 5;
  auto const tr      = t & 0x1f;
  for (uint32_t run = r; run < numruns; run += num_warps) {
    uint32_t base, pos, w, n;
    int mode;
    if (tr == 0) {
      uint32_t byte0;
      base  = rle->runs_loc[run];
      pos   = vals[base];
      byte0 = bytestream_readbyte(bs, pos++);
      mode  = byte0 >> 6;
      if (mode == 0) {
        T baseval;
        // 00lllnnn: short repeat encoding
        w = 8 + (byte0 & 0x38);  // 8 to 64 bits
        n = 3 + (byte0 & 7);     // 3 to 10 values
        bytestream_readbe(bs, pos * 8, w, baseval);
        if constexpr (sizeof(T) <= 4) {
          rle->baseval.u32[r] = baseval;
        } else {
          rle->baseval.u64[r] = baseval;
        }
      } else {
        w = kRLEv2_W[(byte0 >> 1) & 0x1f];
        n = 1 + ((byte0 & 1) << 8) + bytestream_readbyte(bs, pos++);
        if (mode > 1) {
          if (mode == 2) {
            // Patched base
            uint32_t byte2 = bytestream_readbyte(bs, pos++);
            uint32_t byte3 = bytestream_readbyte(bs, pos++);
            uint32_t bw    = 1 + (byte2 >> 5);        // base value width, 1 to 8 bytes
            uint32_t pw    = kRLEv2_W[byte2 & 0x1f];  // patch width, 1 to 64 bits
            if constexpr (sizeof(T) <= 4) {
              uint32_t baseval;
              bytestream_readbe(bs, pos * 8, bw * 8, baseval);
              uint32_t const mask = (1u << (bw * 8 - 1)) - 1;
              // Negative values are represented with the highest bit set to 1
              rle->baseval.u32[r] = (std::is_signed_v<T> and baseval > mask)
                                      ? -static_cast<int32_t>(baseval & mask)
                                      : baseval;
            } else {
              uint64_t baseval;
              bytestream_readbe(bs, pos * 8, bw * 8, baseval);
              uint64_t const mask = (1ul << (bw * 8 - 1)) - 1;
              // Negative values are represented with the highest bit set to 1
              rle->baseval.u64[r] = (std::is_signed_v<T> and baseval > mask)
                                      ? -static_cast<int64_t>(baseval & mask)
                                      : baseval;
            }
            rle->m2_pw_byte3[r] = (pw << 8) | byte3;
            pos += bw;
          } else {
            T baseval;
            int64_t delta;
            // Delta
            pos = decode_varint(bs, pos, baseval);
            if constexpr (sizeof(T) <= 4) {
              rle->baseval.u32[r] = baseval;
            } else {
              rle->baseval.u64[r] = baseval;
            }
            pos           = decode_varint(bs, pos, delta);
            rle->delta[r] = delta;
          }
        }
      }
    }
    base = shuffle(base);
    mode = shuffle(mode);
    pos  = shuffle(pos);
    n    = shuffle(n);
    w    = shuffle(w);
    __syncwarp();  // Not required, included to fix the racecheck warning
    for (uint32_t i = tr; i < n; i += 32) {
      if constexpr (sizeof(T) <= 4) {
        if (mode == 0) {
          vals[base + i] = rle->baseval.u32[r];
        } else if (mode == 1) {
          T v;
          bytestream_readbe(bs, pos * 8 + i * w, w, v);
          vals[base + i] = v;
        } else if (mode == 2) {
          uint32_t ofs   = bytestream_readbits(bs, pos * 8 + i * w, w);
          vals[base + i] = rle->baseval.u32[r] + ofs;
        } else {
          int64_t delta = rle->delta[r];
          if (w > 1 && i > 1) {
            int32_t delta_s = (delta < 0) ? -1 : 0;
            vals[base + i] =
              (bytestream_readbits(bs, pos * 8 + (i - 2) * w, w) ^ delta_s) - delta_s;
          } else {
            vals[base + i] = (i == 0) ? 0 : static_cast<uint32_t>(delta);
          }
        }
      } else {
        if (mode == 0) {
          vals[base + i] = rle->baseval.u64[r];
        } else if (mode == 1) {
          T v;
          bytestream_readbe(bs, pos * 8 + i * w, w, v);
          vals[base + i] = v;
        } else if (mode == 2) {
          uint64_t ofs   = bytestream_readbits64(bs, pos * 8 + i * w, w);
          vals[base + i] = rle->baseval.u64[r] + ofs;
        } else {
          int64_t delta = rle->delta[r], ofs;
          if (w > 1 && i > 1) {
            int64_t delta_s = (delta < 0) ? -1 : 0;
            ofs = (bytestream_readbits64(bs, pos * 8 + (i - 2) * w, w) ^ delta_s) - delta_s;
          } else {
            ofs = (i == 0) ? 0 : delta;
          }
          vals[base + i] = ofs;
        }
      }
    }
    __syncwarp();
    // Patch values
    if (mode == 2) {
      uint32_t pw_byte3 = rle->m2_pw_byte3[r];
      uint32_t pw       = pw_byte3 >> 8;
      uint32_t pgw      = 1 + ((pw_byte3 >> 5) & 7);  // patch gap width, 1 to 8 bits
      uint32_t pll      = pw_byte3 & 0x1f;            // patch list length
      if (pll != 0) {
        uint32_t pgw_pw_len = ClosestFixedBitsMap[min(pw + pgw, 64u)];
        uint64_t patch_pos64 =
          (tr < pll) ? bytestream_readbits64(
                         bs, pos * 8 + ((n * w + 7) & ~7) + tr * (pgw_pw_len), pgw_pw_len)
                     : 0;
        uint32_t patch_pos;
        T patch = 1;
        patch <<= pw;
        patch = (patch - 1) & (T)patch_pos64;
        patch <<= w;
        patch_pos = (uint32_t)(patch_pos64 >> pw);
        for (uint32_t k = 1; k < pll; k <<= 1) {
          uint32_t tmp = shuffle(patch_pos, (tr & ~k) | (k - 1));
          patch_pos += (tr & k) ? tmp : 0;
        }
        if (tr < pll && patch_pos < n) { vals[base + patch_pos] += patch; }
      }
    }
    __syncwarp();
    if (mode == 3) {
      T baseval;
      for (uint32_t i = 1; i < n; i <<= 1) {
        __syncwarp();
        for (uint32_t j = tr; j < n; j += 32) {
          if (j & i) vals[base + j] += vals[base + ((j & ~i) | (i - 1))];
        }
      }
      if constexpr (sizeof(T) <= 4)
        baseval = rle->baseval.u32[r];
      else
        baseval = rle->baseval.u64[r];
      for (uint32_t j = tr; j < n; j += 32) {
        vals[base + j] += baseval;
      }
    }
    __syncwarp();
  }
  __syncthreads();
  // Currently run_cache_manager is only designed to fix the TIMESTAMP's DATA stream bug where the
  // data type is int64_t.
  if constexpr (cuda::std::is_same_v<T, int64_t>) {
    if (cache_helper_inst != nullptr) {
      // Run cache is read from during the 2nd iteration of the top-level while loop in
      // decode_column_data_kernel().
      cache_helper_inst->read_from_cache(vals, rle);
      // Run cache is written to during the 1st iteration of the loop.
      cache_helper_inst->write_to_cache(vals);
    }
  }
  return rle->num_vals;
}

/**
 * @brief Reads 32 booleans as a packed 32-bit value
 *
 * @param[in] vals 32-bit array of values (little-endian)
 * @param[in] bitpos bit position
 *
 * @return 32-bit value
 */
inline __device__ uint32_t rle8_read_bool32(uint32_t* vals, uint32_t bitpos)
{
  uint32_t a = vals[(bitpos >> 5) + 0];
  uint32_t b = vals[(bitpos >> 5) + 1];
  a          = __byte_perm(a, 0, 0x0123);
  b          = __byte_perm(b, 0, 0x0123);
  return __brev(__funnelshift_l(b, a, bitpos));
}

/**
 * @brief ORC Byte RLE decoding
 *
 * @param[in] bs Input byte stream
 * @param[in] rle RLE state
 * @param[in] vals output buffer for decoded 8-bit values
 * @param[in] maxvals Maximum number of values to decode
 * @param[in] t thread id
 *
 * @return number of values decoded
 */
static __device__ uint32_t
byte_rle(orc_bytestream_s* bs, orc_byte_rle_state_s* rle, uint8_t* vals, uint32_t maxvals, int t)
{
  uint32_t numvals, numruns;
  int r, tr;
  if (t == 0) {
    uint32_t maxpos  = min(bs->len, bs->pos + (bytestream_buffer_size - 8u));
    uint32_t lastpos = bs->pos;
    numvals = numruns = 0;
    // Find the length and start location of each run
    while (numvals < maxvals && numruns < num_warps) {
      uint32_t pos           = lastpos, n;
      rle->runs_pos[numruns] = pos;
      rle->runs_loc[numruns] = numvals;
      n                      = bytestream_readbyte(bs, pos++);
      if (n <= 0x7f) {
        // Run
        n = n + 3;
        pos++;
      } else {
        // Literals
        n = 0x100 - n;
        pos += n;
      }
      if ((numvals != 0) and (numvals + n > maxvals)) break;
      if (pos > maxpos) break;
      numruns++;
      ((numvals == 0) and (n > maxvals)) ? numvals = maxvals : numvals += n;
      lastpos = pos;
    }
    rle->num_runs = numruns;
    rle->num_vals = numvals;
    bytestream_flush_bytes(bs, lastpos - bs->pos);
  }
  __syncthreads();
  numruns = rle->num_runs;
  r       = t >> 5;
  tr      = t & 0x1f;
  for (int run = r; run < numruns; run += num_warps) {
    uint32_t pos = rle->runs_pos[run];
    uint32_t loc = rle->runs_loc[run];
    uint32_t n   = bytestream_readbyte(bs, pos++);
    uint32_t literal_mask;
    if (n <= 0x7f) {
      literal_mask = 0;
      n += 3;
    } else {
      literal_mask = ~0;
      n            = 0x100 - n;
    }
    for (uint32_t i = tr; i < n; i += 32) {
      vals[loc + i] = bytestream_readbyte(bs, pos + (i & literal_mask));
    }
  }
  __syncthreads();
  return rle->num_vals;
}

static __device__ const __constant__ int64_t kPow5i[28] = {1,
                                                           5,
                                                           25,
                                                           125,
                                                           625,
                                                           3125,
                                                           15625,
                                                           78125,
                                                           390625,
                                                           1953125,
                                                           9765625,
                                                           48828125,
                                                           244140625,
                                                           1220703125,
                                                           6103515625ll,
                                                           30517578125ll,
                                                           152587890625ll,
                                                           762939453125ll,
                                                           3814697265625ll,
                                                           19073486328125ll,
                                                           95367431640625ll,
                                                           476837158203125ll,
                                                           2384185791015625ll,
                                                           11920928955078125ll,
                                                           59604644775390625ll,
                                                           298023223876953125ll,
                                                           1490116119384765625ll,
                                                           7450580596923828125ll};

/**
 * @brief ORC Decimal decoding (unbounded base-128 varints)
 *
 * @param[in] bs Input byte stream
 * @param[in] scratch Scratch buffer for intermediate results
 * @param[in,out] vals on input: scale from secondary stream, on output: value
 * @param[in] val_scale Scale of each value
 * @param[in] numvals Number of values to decode
 * @param[in] dtype_id Data type identifier for the decimal column
 * @param[in] col_scale Scale from schema to which value will be adjusted
 * @param[in] t thread id
 *
 * @return number of values decoded
 */
static __device__ int decode_decimals(orc_bytestream_s* bs,
                                      orc_byte_rle_state_s* scratch,
                                      orcdec_state_s::values& vals,
                                      int val_scale,
                                      int numvals,
                                      type_id dtype_id,
                                      int col_scale,
                                      int t)
{
  uint32_t num_vals_read = 0;
  // Iterates till `numvals` are read or there is nothing to read once the
  // stream has reached its end, and can't read anything more.
  while (num_vals_read != numvals) {
    if (t == 0) {
      uint32_t maxpos  = min(bs->len, bs->pos + (bytestream_buffer_size - 8u));
      uint32_t lastpos = bs->pos;
      uint32_t n;
      for (n = num_vals_read; n < numvals; n++) {
        uint32_t pos = lastpos;
        pos += varint_length<uint4>(bs, pos);
        if (pos > maxpos) break;
        vals.i64[2 * n] = lastpos;
        lastpos         = pos;
      }
      scratch->num_vals = n;
      bytestream_flush_bytes(bs, lastpos - bs->pos);
    }
    __syncthreads();
    uint32_t num_vals_to_read = scratch->num_vals;

    __int128_t v = 0;
    if (t >= num_vals_read and t < num_vals_to_read) {
      auto const pos = static_cast<int>(vals.i64[2 * t]);
      v              = decode_varint128(bs, pos);
    }
    __syncthreads();
    if (t >= num_vals_read and t < num_vals_to_read) {
      auto const scaled_value = [&]() {
        // Since cuDF column stores just one scale, value needs to be adjusted to col_scale from
        // val_scale. So the difference of them will be used to add 0s or remove digits.
        int32_t const scale = (t < numvals) ? col_scale - val_scale : 0;
        if (scale >= 0) {
          auto const abs_scale = min(scale, 27);
          return (v * kPow5i[abs_scale]) << abs_scale;
        } else {
          auto const abs_scale = min(-scale, 27);
          return (v / kPow5i[abs_scale]) >> abs_scale;
        }
      }();

      if (dtype_id == type_id::DECIMAL32) {
        vals.i32[t] = scaled_value;
      } else if (dtype_id == type_id::DECIMAL64) {
        vals.i64[t] = scaled_value;
      } else {
        vals.i128[t] = scaled_value;
      }
    }
    // There is nothing to read, so break
    if (num_vals_read == num_vals_to_read) break;

    // Update number of values read (This contains values of previous iteration)
    num_vals_read = num_vals_to_read;

    // Have to wait till all threads have copied data
    __syncthreads();
    if (num_vals_read != numvals) {
      bytestream_fill(bs, t);
      __syncthreads();
      if (t == 0) {
        // Needs to be reset since bytestream has been filled
        bs->fill_count = 0;
      }
    }
    // Adding to get all threads in sync before next read
    __syncthreads();
  }
  return num_vals_read;
}

/**
 * @brief Decoding NULLs and builds string dictionary index tables
 *
 * @param[in] chunks column_desc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] max_num_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 */
// blockDim {block_size,1,1}
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  decode_nulls_and_string_dictionaries_kernel(column_desc* chunks,
                                              dictionary_entry* global_dictionary,
                                              size_type num_columns,
                                              size_type num_stripes,
                                              int64_t first_row)
{
  __shared__ __align__(16) orcdec_state_s state_g;
  using warp_reduce  = cub::WarpReduce<uint32_t>;
  using block_reduce = cub::BlockReduce<uint32_t, block_size>;
  __shared__ union {
    typename warp_reduce::TempStorage wr_storage[block_size / 32];
    typename block_reduce::TempStorage bk_storage;
  } temp_storage;

  orcdec_state_s* const s = &state_g;
  // Need the modulo because we have twice as many threads as columns*stripes
  uint32_t const column   = blockIdx.x / num_stripes;
  uint32_t const stripe   = blockIdx.x % num_stripes;
  uint32_t const chunk_id = stripe * num_columns + column;
  int t                   = threadIdx.x;

  if (t == 0) s->chunk = chunks[chunk_id];
  __syncthreads();
  size_t const max_num_rows = s->chunk.column_num_rows - s->chunk.parent_validity_info.null_count;

  bool const is_nulldec = (blockIdx.y == 0);
  if (is_nulldec) {
    uint32_t null_count = 0;
    // Decode NULLs
    if (t == 0) {
      s->chunk.skip_count   = 0;
      s->top.nulls_desc_row = 0;
      bytestream_init(&s->bs, s->chunk.streams[CI_PRESENT], s->chunk.strm_len[CI_PRESENT]);
    }
    __syncthreads();
    if (s->chunk.strm_len[CI_PRESENT] == 0) {
      // No present stream: all rows are valid
      s->vals.u32[t] = ~0;
    }
    auto const prev_parent_null_count =
      (s->chunk.parent_null_count_prefix_sums != nullptr && stripe > 0)
        ? s->chunk.parent_null_count_prefix_sums[stripe - 1]
        : 0;
    auto const parent_null_count =
      (s->chunk.parent_null_count_prefix_sums != nullptr)
        ? s->chunk.parent_null_count_prefix_sums[stripe] - prev_parent_null_count
        : 0;
    auto const num_elems = s->chunk.num_rows - parent_null_count;
    while (s->top.nulls_desc_row < num_elems) {
      auto const nrows_max =
        static_cast<uint32_t>(min(num_elems - s->top.nulls_desc_row, blockDim.x * 32ul));

      bytestream_fill(&s->bs, t);
      __syncthreads();

      uint32_t nrows;
      if (s->chunk.strm_len[CI_PRESENT] > 0) {
        uint32_t nbytes = byte_rle(&s->bs, &s->u.rle8, s->vals.u8, (nrows_max + 7) >> 3, t);
        nrows           = min(nrows_max, nbytes * 8u);
        if (!nrows) {
          // Error: mark all remaining rows as null
          nrows = nrows_max;
          if (t * 32 < nrows) { s->vals.u32[t] = 0; }
        }
      } else {
        nrows = nrows_max;
      }
      __syncthreads();

      auto const row_in = s->chunk.start_row + s->top.nulls_desc_row - prev_parent_null_count;
      if (row_in + nrows > first_row && row_in < first_row + max_num_rows &&
          s->chunk.valid_map_base != nullptr) {
        int64_t dst_row   = row_in - first_row;
        int64_t dst_pos   = max(dst_row, (int64_t)0);
        uint32_t startbit = -static_cast<int32_t>(min(dst_row, (int64_t)0));
        uint32_t nbits    = nrows - min(startbit, nrows);
        uint32_t* valid   = s->chunk.valid_map_base + (dst_pos >> 5);
        uint32_t bitpos   = static_cast<uint32_t>(dst_pos) & 0x1f;
        if ((size_t)(dst_pos + nbits) > max_num_rows) {
          nbits = static_cast<uint32_t>(max_num_rows - min((size_t)dst_pos, max_num_rows));
        }
        // Store bits up to the next 32-bit aligned boundary
        if (bitpos != 0) {
          uint32_t n = min(32u - bitpos, nbits);
          if (t == 0) {
            uint32_t mask = ((1 << n) - 1) << bitpos;
            uint32_t bits = (rle8_read_bool32(s->vals.u32, startbit) << bitpos) & mask;
            atomicAnd(valid, ~mask);
            atomicOr(valid, bits);
            null_count += __popc((~bits) & mask);
          }
          nbits -= n;
          startbit += n;
          valid++;
        }
        // Store bits aligned
        if (t * 32 + 32 <= nbits) {
          uint32_t bits = rle8_read_bool32(s->vals.u32, startbit + t * 32);
          valid[t]      = bits;
          null_count += __popc(~bits);
        } else if (t * 32 < nbits) {
          uint32_t n    = nbits - t * 32;
          uint32_t mask = (1 << n) - 1;
          uint32_t bits = rle8_read_bool32(s->vals.u32, startbit + t * 32) & mask;
          atomicAnd(valid + t, ~mask);
          atomicOr(valid + t, bits);
          null_count += __popc((~bits) & mask);
        }
        __syncthreads();
      }
      // We may have some valid values that are not decoded below first_row -> count these in
      // skip_count, so that subsequent kernel can infer the correct row position
      if (row_in < first_row && t < 32) {
        uint32_t skippedrows = min(static_cast<uint32_t>(first_row - row_in), nrows);
        uint32_t skip_count  = 0;
        for (thread_index_type i = t * 32; i < skippedrows; i += 32 * 32) {
          // Need to arrange the bytes to apply mask properly.
          uint32_t bits = (i + 32 <= skippedrows) ? s->vals.u32[i >> 5]
                                                  : (__byte_perm(s->vals.u32[i >> 5], 0, 0x0123) &
                                                     (0xffff'ffffu << (0x20 - skippedrows + i)));
          skip_count += __popc(bits);
        }
        skip_count = warp_reduce(temp_storage.wr_storage[t / 32]).Sum(skip_count);
        if (t == 0) { s->chunk.skip_count += skip_count; }
      }
      __syncthreads();
      if (t == 0) { s->top.nulls_desc_row += nrows; }
      __syncthreads();
    }
    __syncthreads();
    // Sum up the valid counts and infer null_count
    null_count = block_reduce(temp_storage.bk_storage).Sum(null_count);
    if (t == 0) {
      chunks[chunk_id].null_count = parent_null_count + null_count;
      chunks[chunk_id].skip_count = s->chunk.skip_count;
    }
  } else {
    // Decode string dictionary
    int encoding_kind = s->chunk.encoding_kind;
    if ((encoding_kind == DICTIONARY || encoding_kind == DICTIONARY_V2) &&
        (s->chunk.dict_len > 0)) {
      if (t == 0) {
        s->top.dict.dict_len   = s->chunk.dict_len;
        s->top.dict.local_dict = global_dictionary + s->chunk.dictionary_start;  // Local dictionary
        s->top.dict.dict_pos   = 0;
        // CI_DATA2 contains the LENGTH stream coding the length of individual dictionary entries
        bytestream_init(&s->bs, s->chunk.streams[CI_DATA2], s->chunk.strm_len[CI_DATA2]);
      }
      __syncthreads();
      while (s->top.dict.dict_len > 0) {
        uint32_t numvals = min(s->top.dict.dict_len, blockDim.x), len;
        uint32_t* vals   = s->vals.u32;
        bytestream_fill(&s->bs, t);
        __syncthreads();
        if (is_rlev1(s->chunk.encoding_kind)) {
          numvals = integer_rlev1(&s->bs, &s->u.rlev1, vals, numvals, t);
        } else  // RLEv2
        {
          numvals = integer_rlev2(&s->bs, &s->u.rlev2, vals, numvals, t);
        }
        __syncthreads();
        len = (t < numvals) ? vals[t] : 0;
        lengths_to_positions(vals, numvals, t);
        __syncthreads();
        if (numvals == 0) {
          // This is an error (ran out of data)
          numvals = min(s->top.dict.dict_len, blockDim.x);
          vals[t] = 0;
        }
        if (t < numvals) {
          s->top.dict.local_dict[t] = {s->top.dict.dict_pos + vals[t] - len, len};
        }
        __syncthreads();
        if (t == 0) {
          s->top.dict.dict_pos += vals[numvals - 1];
          s->top.dict.dict_len -= numvals;
          s->top.dict.local_dict += numvals;
        }
        __syncthreads();
      }
    }
  }
}

/**
 * @brief Decode row positions from valid bits
 *
 * @param[in,out] s Column chunk decoder state
 * @param[in] first_row crop all rows below first rows
 * @param[in] t thread id
 * @param[in] temp_storage shared memory storage to perform block reduce
 */
template <typename Storage>
static __device__ void DecodeRowPositions(orcdec_state_s* s,
                                          size_t first_row,
                                          int t,
                                          Storage& temp_storage)
{
  using block_reduce = cub::BlockReduce<uint32_t, block_size>;

  if (t == 0) {
    if (s->chunk.skip_count != 0) {
      s->u.rowdec.nz_count =
        min(static_cast<uint32_t>(
              min(s->chunk.skip_count, static_cast<uint64_t>(s->top.data.max_vals))),
            blockDim.x);
      s->chunk.skip_count -= s->u.rowdec.nz_count;
      s->top.data.nrows = s->u.rowdec.nz_count;
    } else {
      s->u.rowdec.nz_count = 0;
    }
  }
  __syncthreads();
  if (t < s->u.rowdec.nz_count) {
    s->u.rowdec.row[t] = 0;  // Skipped values (below first_row)
  }
  while (s->u.rowdec.nz_count < s->top.data.max_vals &&
         s->top.data.cur_row + s->top.data.nrows < s->top.data.end_row) {
    uint32_t const remaining_rows = s->top.data.end_row - (s->top.data.cur_row + s->top.data.nrows);
    uint32_t nrows =
      min(remaining_rows, min((row_decoder_buffer_size - s->u.rowdec.nz_count) * 2, blockDim.x));
    if (s->chunk.valid_map_base != nullptr) {
      // We have a present stream
      uint32_t rmax       = s->top.data.end_row - min(first_row, s->top.data.end_row);
      auto r              = (uint32_t)(s->top.data.cur_row + s->top.data.nrows + t - first_row);
      uint32_t valid      = (t < nrows && r < rmax)
                              ? (((uint8_t const*)s->chunk.valid_map_base)[r >> 3] >> (r & 7)) & 1
                              : 0;
      auto* row_ofs_plus1 = (uint16_t*)&s->u.rowdec.row[s->u.rowdec.nz_count];
      uint32_t nz_pos, row_plus1, nz_count = s->u.rowdec.nz_count, last_row;
      if (t < nrows) { row_ofs_plus1[t] = valid; }
      lengths_to_positions<uint16_t>(row_ofs_plus1, nrows, t);
      if (t < nrows) {
        nz_count += row_ofs_plus1[t];
        row_plus1 = s->top.data.nrows + t + 1;
      } else {
        row_plus1 = 0;
      }
      __syncthreads();
      if (t == nrows - 1) { s->u.rowdec.nz_count = min(nz_count, s->top.data.max_vals); }
      __syncthreads();

      // TBD: Brute-forcing this, there might be a more efficient way to find the thread with the
      // last row
      last_row = (nz_count == s->u.rowdec.nz_count) ? row_plus1 : 0;
      last_row = block_reduce(temp_storage).Reduce(last_row, cuda::maximum{});
      nz_pos   = (valid) ? nz_count : 0;
      if (t == 0) { s->top.data.nrows = last_row; }
      if (valid && nz_pos - 1 < s->u.rowdec.nz_count) { s->u.rowdec.row[nz_pos - 1] = row_plus1; }
    } else {
      // All values are valid
      nrows = min(nrows, s->top.data.max_vals - s->u.rowdec.nz_count);
      if (t < nrows) { s->u.rowdec.row[s->u.rowdec.nz_count + t] = s->top.data.nrows + t + 1; }
      __syncthreads();
      if (t == 0) {
        s->top.data.nrows += nrows;
        s->u.rowdec.nz_count += nrows;
      }
    }
    __syncthreads();
  }
}

/**
 * @brief Trailing zeroes for decoding timestamp nanoseconds
 */
static __device__ const __constant__ uint32_t kTimestampNanoScale[8] = {
  1, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};

/**
 * @brief Decodes column data
 *
 * @param[in] chunks column_desc device array
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] tz_table Timezone translation table
 * @param[in] row_groups Optional row index data
 * @param[in] first_row Crop all rows below first_row
 * @param[in] rowidx_stride Row index stride
 * @param[in] level nesting level being processed
 */
// blockDim {block_size,1,1}
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  decode_column_data_kernel(column_desc* chunks,
                            dictionary_entry* global_dictionary,
                            table_device_view tz_table,
                            device_2dspan<row_group> row_groups,
                            int64_t first_row,
                            size_type rowidx_stride,
                            size_t level,
                            size_type* error_count)
{
  __shared__ __align__(16) orcdec_state_s state_g;
  using block_reduce = cub::BlockReduce<uint64_t, block_size>;
  __shared__ union {
    typename cub::BlockReduce<uint32_t, block_size>::TempStorage blk_uint32;
    typename cub::BlockReduce<uint64_t, block_size>::TempStorage blk_uint64;
  } temp_storage;

  orcdec_state_s* const s = &state_g;
  uint32_t chunk_id;
  int t              = threadIdx.x;
  auto num_rowgroups = row_groups.size().first;

  if (num_rowgroups > 0) {
    if (t == 0) {
      auto const rg_idx  = blockIdx.x % num_rowgroups;
      auto const col_idx = blockIdx.x / num_rowgroups;
      s->top.data.index  = row_groups[rg_idx][col_idx];
    }
    __syncthreads();
    chunk_id = s->top.data.index.chunk_id;
  } else {
    chunk_id = blockIdx.x;
  }
  if (t == 0) {
    s->chunk          = chunks[chunk_id];
    s->num_child_rows = 0;
  }
  __syncthreads();
  // Struct doesn't have any data in itself, so skip
  bool const is_valid       = s->chunk.type_kind != STRUCT;
  size_t const max_num_rows = s->chunk.column_num_rows;
  __shared__ run_cache_manager run_cache_manager_inst;
  cache_helper cache_helper_inst(run_cache_manager_inst);

  __syncthreads();
  if (t == 0 and is_valid) {
    // If we have an index, seek to the initial run and update row positions
    if (num_rowgroups > 0) {
      if (s->top.data.index.strm_offset[0] > s->chunk.strm_len[CI_DATA]) {
        atomicAdd(error_count, 1);
      }
      if (s->top.data.index.strm_offset[1] > s->chunk.strm_len[CI_DATA2]) {
        atomicAdd(error_count, 1);
      }
      auto const ofs0       = min(s->top.data.index.strm_offset[0], s->chunk.strm_len[CI_DATA]);
      auto const ofs1       = min(s->top.data.index.strm_offset[1], s->chunk.strm_len[CI_DATA2]);
      uint32_t const rg_idx = blockIdx.x % num_rowgroups;
      uint32_t rowgroup_rowofs =
        (level == 0) ? (rg_idx - cuda::std::min(s->chunk.rowgroup_id, rg_idx)) * rowidx_stride
                     : s->top.data.index.start_row;
      ;
      s->chunk.streams[CI_DATA] += ofs0;
      s->chunk.strm_len[CI_DATA] -= ofs0;
      s->chunk.streams[CI_DATA2] += ofs1;
      s->chunk.strm_len[CI_DATA2] -= ofs1;
      rowgroup_rowofs = min(static_cast<uint64_t>(rowgroup_rowofs), s->chunk.num_rows);
      s->chunk.start_row += rowgroup_rowofs;
      s->chunk.num_rows -= rowgroup_rowofs;
    }
    s->is_string               = (s->chunk.type_kind == STRING || s->chunk.type_kind == BINARY ||
                    s->chunk.type_kind == VARCHAR || s->chunk.type_kind == CHAR);
    s->top.data.cur_row        = max(s->chunk.start_row, max(first_row - s->chunk.skip_count, 0ul));
    s->top.data.end_row        = s->chunk.start_row + s->chunk.num_rows;
    s->top.data.buffered_count = 0;
    if (s->top.data.end_row > first_row + max_num_rows) {
      s->top.data.end_row = first_row + max_num_rows;
    }
    if (num_rowgroups > 0) {
      s->top.data.end_row =
        min(s->top.data.end_row, s->chunk.start_row + s->top.data.index.num_rows);
    }
    if (!is_dictionary(s->chunk.encoding_kind)) { s->chunk.dictionary_start = 0; }

    static constexpr duration_s d_orc_utc_epoch = duration_s{orc_utc_epoch};
    s->top.data.tz_epoch = d_orc_utc_epoch - get_ut_offset(tz_table, timestamp_s{d_orc_utc_epoch});

    bytestream_init(&s->bs, s->chunk.streams[CI_DATA], s->chunk.strm_len[CI_DATA]);
    bytestream_init(&s->bs2, s->chunk.streams[CI_DATA2], s->chunk.strm_len[CI_DATA2]);

    run_cache_manager_inst.initialize(s);
  }
  __syncthreads();

  while (is_valid && (s->top.data.cur_row < s->top.data.end_row)) {
    uint32_t list_child_elements = 0;
    bytestream_fill(&s->bs, t);
    bytestream_fill(&s->bs2, t);
    __syncthreads();
    uint32_t num_remaining_rows = s->chunk.start_row + s->chunk.num_rows - s->top.data.cur_row;
    if (num_rowgroups > 0 && (s->is_string || s->chunk.type_kind == TIMESTAMP)) {
      num_remaining_rows +=
        s->top.data.index.run_pos[is_dictionary(s->chunk.encoding_kind) ? CI_DATA : CI_DATA2];
    }
    if (t == 0) {
      s->bs.fill_count  = 0;
      s->bs2.fill_count = 0;
      s->top.data.nrows = 0;
      s->top.data.max_vals =
        min(num_remaining_rows, (s->chunk.type_kind == BOOLEAN) ? blockDim.x * 2 : blockDim.x);
    }
    __syncthreads();
    // Decode data streams
    {
      uint32_t numvals       = s->top.data.max_vals;
      uint64_t secondary_val = 0;
      uint32_t vals_skipped  = 0;
      if (s->is_string || s->chunk.type_kind == TIMESTAMP) {
        // For these data types, we have a secondary unsigned 32-bit data stream
        orc_bytestream_s* bs = (is_dictionary(s->chunk.encoding_kind)) ? &s->bs : &s->bs2;
        uint32_t ofs         = 0;
        if (s->chunk.type_kind == TIMESTAMP) {
          // Restore buffered secondary stream values, if any
          ofs = s->top.data.buffered_count;
          if (ofs > 0) {
            __syncthreads();
            if (t == 0) { s->top.data.buffered_count = 0; }
          }
        }
        if (numvals > ofs) {
          if (is_rlev1(s->chunk.encoding_kind)) {
            if (s->chunk.type_kind == TIMESTAMP)
              numvals = ofs + integer_rlev1(bs, &s->u.rlev1, &s->vals.u64[ofs], numvals - ofs, t);
            else
              numvals = ofs + integer_rlev1(bs, &s->u.rlev1, &s->vals.u32[ofs], numvals - ofs, t);
          } else {
            if (s->chunk.type_kind == TIMESTAMP)
              numvals =
                ofs + integer_rlev2(bs, &s->u.rlev2, &s->vals.u64[ofs], numvals - ofs, t, ofs > 0);
            else
              numvals =
                ofs + integer_rlev2(bs, &s->u.rlev2, &s->vals.u32[ofs], numvals - ofs, t, ofs > 0);
          }
          __syncthreads();
          if (numvals <= ofs && t >= ofs && t < s->top.data.max_vals) { s->vals.u32[t] = 0; }
        }
        // If we're using an index, we may have to drop values from the initial run
        if (num_rowgroups > 0) {
          int cid          = is_dictionary(s->chunk.encoding_kind) ? CI_DATA : CI_DATA2;
          uint32_t run_pos = s->top.data.index.run_pos[cid];
          if (run_pos) {
            vals_skipped = min(numvals, run_pos);
            __syncthreads();
            if (t == 0) { s->top.data.index.run_pos[cid] = 0; }
            numvals -= vals_skipped;
            if (t < numvals) {
              secondary_val = (s->chunk.type_kind == TIMESTAMP) ? s->vals.u64[vals_skipped + t]
                                                                : s->vals.u32[vals_skipped + t];
            }
            __syncthreads();
            if (t < numvals) {
              if (s->chunk.type_kind == TIMESTAMP)
                s->vals.u64[t] = secondary_val;
              else
                s->vals.u32[t] = secondary_val;
            }
          }
        }
        __syncthreads();
        // For strings with direct encoding, we need to convert the lengths into an offset
        if (!is_dictionary(s->chunk.encoding_kind)) {
          if (t < numvals)
            secondary_val = (s->chunk.type_kind == TIMESTAMP) ? s->vals.u64[t] : s->vals.u32[t];
          if (s->chunk.type_kind != TIMESTAMP) {
            lengths_to_positions(s->vals.u32, numvals, t);
            __syncthreads();
          }
        }
        // Adjust the maximum number of values
        if (numvals == 0 && vals_skipped == 0) {
          numvals = s->top.data.max_vals;  // Just so that we don't hang if the stream is corrupted
        }
        __syncthreads();
        if (t == 0) { s->top.data.max_vals = cuda::std::min(s->top.data.max_vals, numvals); }
      }
      __syncthreads();
      // Account for skipped values
      if (num_rowgroups > 0 && !s->is_string) {
        uint32_t run_pos =
          (s->chunk.type_kind == DECIMAL || s->chunk.type_kind == LIST || s->chunk.type_kind == MAP)
            ? s->top.data.index.run_pos[CI_DATA2]
            : s->top.data.index.run_pos[CI_DATA];
        numvals =
          min(numvals + run_pos, (s->chunk.type_kind == BOOLEAN) ? blockDim.x * 2 : blockDim.x);
        num_remaining_rows += run_pos;
      }
      // Decode the primary data stream
      if (s->chunk.type_kind == INT || s->chunk.type_kind == DATE || s->chunk.type_kind == SHORT) {
        // Signed int32 primary data stream
        if (is_rlev1(s->chunk.encoding_kind)) {
          numvals = integer_rlev1(&s->bs, &s->u.rlev1, s->vals.i32, numvals, t);
        } else {
          numvals = integer_rlev2(&s->bs, &s->u.rlev2, s->vals.i32, numvals, t);
        }
        __syncthreads();
      } else if (s->chunk.type_kind == LIST or s->chunk.type_kind == MAP) {
        if (is_rlev1(s->chunk.encoding_kind)) {
          numvals = integer_rlev1<uint64_t>(&s->bs2, &s->u.rlev1, s->vals.u64, numvals, t);
        } else {
          numvals = integer_rlev2<uint64_t>(&s->bs2, &s->u.rlev2, s->vals.u64, numvals, t);
        }
        __syncthreads();
      } else if (s->chunk.type_kind == BYTE) {
        numvals = byte_rle(&s->bs, &s->u.rle8, s->vals.u8, numvals, t);
        __syncthreads();
      } else if (s->chunk.type_kind == BOOLEAN) {
        int n = ((numvals + 7) >> 3);
        if (n > s->top.data.buffered_count) {
          // we can buffer for extraction up to blockDim.x BYTES at a time,
          // since that's the max we can save for the next iteration in secondary_val
          int max_bytes_buffered =
            min(blockDim.x, (num_remaining_rows + 7) / 8);  // guard against stream end
          numvals = byte_rle(&s->bs,
                             &s->u.rle8,
                             &s->vals.u8[s->top.data.buffered_count],
                             max_bytes_buffered - s->top.data.buffered_count,
                             t) +
                    s->top.data.buffered_count;
        } else {
          numvals = s->top.data.buffered_count;
        }
        __syncthreads();
        if (t == 0) {
          s->top.data.buffered_count = 0;
          s->top.data.max_vals       = min(s->top.data.max_vals, blockDim.x);
        }
        __syncthreads();
        // If the condition is false, then it means that s->top.data.max_vals is last set of values.
        // And as numvals is considered to be min(`max_vals+s->top.data.index.run_pos[CI_DATA]`,
        // blockDim.x*2) we have to return numvals >= s->top.data.index.run_pos[CI_DATA].
        auto const is_last_set = (s->top.data.max_vals >= s->top.data.index.run_pos[CI_DATA]);
        auto const max_vals    = (is_last_set ? s->top.data.max_vals + 7 : blockDim.x) / 8;
        n                      = numvals - max_vals;
        if (t < n) {
          secondary_val = s->vals.u8[max_vals + t];
          if (t == 0) { s->top.data.buffered_count = n; }
        }

        numvals = min(numvals * 8, is_last_set ? (s->top.data.max_vals + 7) & (~0x7) : blockDim.x);

      } else if (s->chunk.type_kind == LONG || s->chunk.type_kind == TIMESTAMP ||
                 s->chunk.type_kind == DECIMAL) {
        orc_bytestream_s* bs = (s->chunk.type_kind == DECIMAL) ? &s->bs2 : &s->bs;
        if (is_rlev1(s->chunk.encoding_kind)) {
          numvals = integer_rlev1<int64_t>(bs, &s->u.rlev1, s->vals.i64, numvals, t);
        } else {
          numvals = integer_rlev2<int64_t>(bs,
                                           &s->u.rlev2,
                                           s->vals.i64,
                                           numvals,
                                           t,
                                           false /**has_buffered_values */,
                                           &cache_helper_inst);
        }
        if (s->chunk.type_kind == DECIMAL) {
          // If we're using an index, we may have to drop values from the initial run
          uint32_t skip = 0;
          int val_scale;
          if (num_rowgroups > 0) {
            uint32_t run_pos = s->top.data.index.run_pos[CI_DATA2];
            if (run_pos) {
              skip = min(numvals, run_pos);
              __syncthreads();
              if (t == 0) { s->top.data.index.run_pos[CI_DATA2] = 0; }
              numvals -= skip;
            }
          }
          val_scale = (t < numvals) ? (int)s->vals.i64[skip + t] : 0;
          __syncthreads();
          numvals = decode_decimals(&s->bs,
                                    &s->u.rle8,
                                    s->vals,
                                    val_scale,
                                    numvals,
                                    s->chunk.dtype_id,
                                    s->chunk.decimal_scale,
                                    t);
        }
        __syncthreads();
      } else if (s->chunk.type_kind == FLOAT) {
        numvals = min(numvals, (bytestream_buffer_size - 8u) >> 2);
        if (t < numvals) { s->vals.u32[t] = bytestream_readu32(&s->bs, s->bs.pos + t * 4); }
        __syncthreads();
        if (t == 0) { bytestream_flush_bytes(&s->bs, numvals * 4); }
        __syncthreads();
      } else if (s->chunk.type_kind == DOUBLE) {
        numvals = min(numvals, (bytestream_buffer_size - 8u) >> 3);
        if (t < numvals) { s->vals.u64[t] = bytestream_readu64(&s->bs, s->bs.pos + t * 8); }
        __syncthreads();
        if (t == 0) { bytestream_flush_bytes(&s->bs, numvals * 8); }
        __syncthreads();
      }
      __syncthreads();
      if (numvals == 0 && vals_skipped != 0 && num_rowgroups > 0) {
        // Special case if the secondary streams produced fewer values than the primary stream's RLE
        // run, as a result of initial RLE run offset: keep vals_skipped as non-zero to ensure
        // proper buffered_count/max_vals update below.
      } else {
        vals_skipped = 0;
        if (num_rowgroups > 0) {
          uint32_t run_pos = (s->chunk.type_kind == LIST or s->chunk.type_kind == MAP)
                               ? s->top.data.index.run_pos[CI_DATA2]
                               : s->top.data.index.run_pos[CI_DATA];
          if (run_pos) {
            vals_skipped = min(numvals, run_pos);
            numvals -= vals_skipped;
            __syncthreads();
            if (t == 0) {
              (s->chunk.type_kind == LIST or s->chunk.type_kind == MAP)
                ? s->top.data.index.run_pos[CI_DATA2] = 0
                : s->top.data.index.run_pos[CI_DATA]  = 0;
            }
          }
        }
      }
      if (t == 0 && numvals + vals_skipped > 0) {
        auto const max_vals = s->top.data.max_vals;
        if (max_vals > numvals) {
          if (s->chunk.type_kind == TIMESTAMP) { s->top.data.buffered_count = max_vals - numvals; }
          s->top.data.max_vals = numvals;
        }
      }
      __syncthreads();
      // Use the valid bits to compute non-null row positions until we get a full batch of values to
      // decode
      DecodeRowPositions(s, first_row, t, temp_storage.blk_uint32);
      if (!s->top.data.nrows && !s->u.rowdec.nz_count && !vals_skipped) {
        // This is a bug (could happen with bitstream errors with a bad run that would produce more
        // values than the number of remaining rows)
        return;
      }

      // Store decoded values to output
      if (t < min(min(s->top.data.max_vals, s->u.rowdec.nz_count), s->top.data.nrows) &&
          s->u.rowdec.row[t] != 0 &&
          s->top.data.cur_row + s->u.rowdec.row[t] - 1 < s->top.data.end_row) {
        size_t row = s->top.data.cur_row + s->u.rowdec.row[t] - 1 - first_row;
        if (row < max_num_rows) {
          void* data_out = s->chunk.column_data_base;
          switch (s->chunk.type_kind) {
            case FLOAT:
            case INT: static_cast<uint32_t*>(data_out)[row] = s->vals.u32[t + vals_skipped]; break;
            case DOUBLE:
            case LONG: static_cast<uint64_t*>(data_out)[row] = s->vals.u64[t + vals_skipped]; break;
            case DECIMAL:
              if (s->chunk.dtype_id == type_id::DECIMAL32) {
                static_cast<uint32_t*>(data_out)[row] = s->vals.u32[t + vals_skipped];
              } else if (s->chunk.dtype_id == type_id::DECIMAL64) {
                static_cast<uint64_t*>(data_out)[row] = s->vals.u64[t + vals_skipped];
              } else {
                // decimal128
                static_cast<__uint128_t*>(data_out)[row] = s->vals.u128[t + vals_skipped];
              }
              break;
            case MAP:
            case LIST: {
              // Since the offsets column in cudf is `size_type`,
              // If the limit exceeds then value will be 0, which is Fail.
              cudf_assert(
                (s->vals.u64[t + vals_skipped] <= cuda::std::numeric_limits<size_type>::max()) and
                "Number of elements is more than what size_type can handle");
              list_child_elements                   = s->vals.u64[t + vals_skipped];
              static_cast<uint32_t*>(data_out)[row] = list_child_elements;
            } break;
            case SHORT:
              static_cast<uint16_t*>(data_out)[row] =
                static_cast<uint16_t>(s->vals.u32[t + vals_skipped]);
              break;
            case BYTE: static_cast<uint8_t*>(data_out)[row] = s->vals.u8[t + vals_skipped]; break;
            case BOOLEAN:
              static_cast<uint8_t*>(data_out)[row] =
                (s->vals.u8[(t + vals_skipped) >> 3] >> ((~(t + vals_skipped)) & 7)) & 1;
              break;
            case DATE:
              if (s->chunk.dtype_len == 8) {
                cudf::duration_D days{s->vals.i32[t + vals_skipped]};
                // Convert from days to milliseconds
                static_cast<int64_t*>(data_out)[row] =
                  cuda::std::chrono::duration_cast<cudf::duration_ms>(days).count();
              } else {
                static_cast<uint32_t*>(data_out)[row] = s->vals.u32[t + vals_skipped];
              }
              break;
            case STRING:
            case BINARY:
            case VARCHAR:
            case CHAR: {
              string_index_pair* strdesc = &static_cast<string_index_pair*>(data_out)[row];
              void const* ptr            = nullptr;
              uint32_t count             = 0;
              if (is_dictionary(s->chunk.encoding_kind)) {
                auto const dict_idx = s->vals.u32[t + vals_skipped];
                if (dict_idx < s->chunk.dict_len) {
                  auto const& g_entry = global_dictionary[s->chunk.dictionary_start + dict_idx];

                  ptr   = s->chunk.streams[CI_DICTIONARY] + g_entry.pos;
                  count = g_entry.len;
                }
              } else {
                auto const dict_idx =
                  s->chunk.dictionary_start + s->vals.u32[t + vals_skipped] - secondary_val;

                if (dict_idx + count <= s->chunk.strm_len[CI_DATA]) {
                  ptr   = s->chunk.streams[CI_DATA] + dict_idx;
                  count = secondary_val;
                }
              }
              strdesc->first  = static_cast<char const*>(ptr);
              strdesc->second = count;
              break;
            }
            case TIMESTAMP: {
              auto seconds = s->top.data.tz_epoch + duration_s{s->vals.i64[t + vals_skipped]};

              duration_ns nanos = duration_ns{(static_cast<int64_t>(secondary_val) >> 3) *
                                              kTimestampNanoScale[secondary_val & 7]};

              // ORC stores timestamps as a (seconds, nanos) pair where `nanos` is always
              // non-negative. For a negative timestamp with a fractional part, the Apache writer
              // rounds `seconds` toward zero and puts the leftover positive remainder into `nanos`.
              // To recover the true value we subtract one second whenever `seconds < 0` and `nanos`
              // contributes a non-zero fractional part.
              //
              // The threshold is 1 ms (not 1 ns) to match the Apache writer's nanos encoding:
              // sub-millisecond values round to zero on write, so on read they must not trigger the
              // borrow.
              if (seconds.count() < 0 and nanos.count() >= 1'000'000) { seconds -= duration_s{1}; }

              // Convert to UTC after the adjustment above, because the adjustment must run in the
              // writer's (stored seconds + writer epoch) frame
              seconds += get_ut_offset(tz_table, timestamp_s{seconds});

              static_cast<int64_t*>(data_out)[row] = [&]() {
                using cuda::std::chrono::duration_cast;
                switch (s->chunk.timestamp_type_id) {
                  case type_id::TIMESTAMP_SECONDS:
                    return (seconds + duration_cast<duration_s>(nanos)).count();
                  case type_id::TIMESTAMP_MILLISECONDS:
                    return (seconds + duration_cast<duration_ms>(nanos)).count();
                  case type_id::TIMESTAMP_MICROSECONDS:
                    return (seconds + duration_cast<duration_us>(nanos)).count();
                  case type_id::TIMESTAMP_NANOSECONDS:
                  default:
                    // nanoseconds as output in case of `type_id::EMPTY` and
                    // `type_id::TIMESTAMP_NANOSECONDS`
                    return (seconds + nanos).count();
                }
              }();

              break;
            }
          }
        }
      }
      // Aggregate num of elements for the chunk
      if (s->chunk.type_kind == LIST or s->chunk.type_kind == MAP) {
        list_child_elements = block_reduce(temp_storage.blk_uint64).Sum(list_child_elements);
      }
      __syncthreads();
      // Buffer secondary stream values
      if (s->chunk.type_kind == TIMESTAMP) {
        int buffer_pos = s->top.data.max_vals;
        if (t >= buffer_pos && t < buffer_pos + s->top.data.buffered_count) {
          s->vals.u64[t - buffer_pos] = secondary_val;
        }
      } else if (s->chunk.type_kind == BOOLEAN && t < s->top.data.buffered_count) {
        s->vals.u8[t] = secondary_val;
      }
    }
    __syncthreads();
    if (t == 0) {
      s->top.data.cur_row += s->top.data.nrows;
      if (s->chunk.type_kind == LIST or s->chunk.type_kind == MAP) {
        s->num_child_rows += list_child_elements;
      }
      if (s->is_string && !is_dictionary(s->chunk.encoding_kind) && s->top.data.max_vals > 0) {
        s->chunk.dictionary_start += s->vals.u32[s->top.data.max_vals - 1];
      }
    }
    __syncthreads();
  }
  if (t == 0 and (s->chunk.type_kind == LIST or s->chunk.type_kind == MAP)) {
    if (num_rowgroups > 0) {
      auto const rg_idx                          = blockIdx.x % num_rowgroups;
      auto const col_idx                         = blockIdx.x / num_rowgroups;
      row_groups[rg_idx][col_idx].num_child_rows = s->num_child_rows;
    }
    cuda::atomic_ref<int64_t, cuda::thread_scope_device> ref{chunks[chunk_id].num_child_rows};
    ref.fetch_add(s->num_child_rows, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Launches kernel for decoding NULLs and building string dictionary index tables
 *
 * @param[in] chunks column_desc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] first_row Crop all rows below first_row
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void __host__ decode_nulls_and_string_dictionaries(column_desc* chunks,
                                                   dictionary_entry* global_dictionary,
                                                   size_type num_columns,
                                                   size_type num_stripes,
                                                   int64_t first_row,
                                                   rmm::cuda_stream_view stream)
{
  dim3 dim_grid(num_columns * num_stripes, 2);

  decode_nulls_and_string_dictionaries_kernel<block_size>
    <<<dim_grid, block_size, 0, stream.value()>>>(
      chunks, global_dictionary, num_columns, num_stripes, first_row);
  CUDF_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief Launches kernel for decoding column data
 *
 * @param[in] chunks column_desc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] row_groups Optional row index data [row_group][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] first_row Crop all rows below first_row
 * @param[in] tz_table Timezone translation table
 * @param[in] num_rowgroups Number of row groups in row index data
 * @param[in] rowidx_stride Row index stride
 * @param[in] level nesting level being processed
 * @param[in] error_count Pointer to error count output
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void __host__ decode_column_data(column_desc* chunks,
                                 dictionary_entry* global_dictionary,
                                 device_2dspan<row_group> row_groups,
                                 size_type num_columns,
                                 size_type num_stripes,
                                 int64_t first_row,
                                 table_device_view tz_table,
                                 int64_t num_rowgroups,
                                 size_type rowidx_stride,
                                 size_t level,
                                 size_type* error_count,
                                 rmm::cuda_stream_view stream)
{
  auto const num_blocks = num_columns * (num_rowgroups > 0 ? num_rowgroups : num_stripes);
  decode_column_data_kernel<block_size><<<num_blocks, block_size, 0, stream.value()>>>(
    chunks, global_dictionary, tz_table, row_groups, first_row, rowidx_stride, level, error_count);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::orc::detail
