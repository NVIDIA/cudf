
/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "page_decode.cuh"
#include "timestamp_utils.cuh"

#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>

namespace cudf::io::parquet::detail {

/**
 * @brief Output a string descriptor
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dstv Pointer to row output data (string descriptor or 32-bit hash)
 */
template <typename state_buf>
inline __device__ void gpuOutputString(auto* s, state_buf* sb, int src_pos, void* dstv)
{
  auto [ptr, len] = gpuGetStringData(s, sb, src_pos);
  if (s->setup.col.is_strings_to_cat and s->setup.col.physical_type == Type::BYTE_ARRAY) {
    // Output hash. This hash value is used if the option to convert strings to
    // categoricals is enabled. The seed value is chosen arbitrarily.
    uint32_t constexpr hash_seed = 33;
    cudf::string_view const sv{ptr, static_cast<size_type>(len)};
    *static_cast<uint32_t*>(dstv) =
      cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>{hash_seed}(sv);
  } else {
    // Output string descriptor
    auto* dst   = static_cast<string_index_pair*>(dstv);
    dst->first  = ptr;
    dst->second = len;
  }
}

/**
 * @brief Output a boolean
 *
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename state_buf>
inline __device__ void read_boolean(state_buf* sb, int src_pos, uint8_t* dst)
{
  *dst = sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)];
}

/**
 * @brief Store a 32-bit data element
 *
 * @param[out] dst ptr to output
 * @param[in] src8 raw input bytes
 * @param[in] dict_pos byte position in dictionary
 * @param[in] dict_size size of dictionary
 */
inline __device__ void gpuStoreOutput(uint32_t* dst,
                                      uint8_t const* src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  if (dict_pos < dict_size) {
    *dst = cudf::io::unaligned_load<uint32_t>(src8 + dict_pos);
  } else {
    *dst = 0;
  }
}

/**
 * @brief Store a 64-bit data element
 *
 * @param[out] dst ptr to output
 * @param[in] src8 raw input bytes
 * @param[in] dict_pos byte position in dictionary
 * @param[in] dict_size size of dictionary
 */
inline __device__ void gpuStoreOutput(uint2* dst,
                                      uint8_t const* src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  uint2 v;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    v.x = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 8);
      v.x           = __funnelshift_r(v.x, v.y, ofs);
      v.y           = __funnelshift_r(v.y, next, ofs);
    }
  } else {
    v.x = v.y = 0;
  }
  *dst = v;
}

/**
 * @brief Convert an INT96 Spark timestamp to 64-bit timestamp
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[out] dst Pointer to row output data
 */
template <typename state_buf>
inline __device__ void read_int96_timestamp(auto* s, state_buf* sb, int src_pos, int64_t* dst)
{
  using cuda::std::chrono::duration_cast;

  uint8_t const* src8;
  uint32_t dict_pos, dict_size = s->stream.dict_size, ofs;

  if (s->stream.dict_base) {
    // Dictionary
    dict_pos = (s->stream.dict_bits > 0)
                 ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)]
                 : 0;
    src8     = s->stream.dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    src8     = s->stream.data_start;
  }
  dict_pos *= (uint32_t)s->output_cvt.dtype_len_in;
  ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits

  if (dict_pos + 4 >= dict_size) {
    *dst = 0;
    return;
  }

  uint3 v;
  int64_t nanos, days;
  v.x = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 0);
  v.y = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
  v.z = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 8);
  if (ofs) {
    uint32_t next = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 12);
    v.x           = __funnelshift_r(v.x, v.y, ofs);
    v.y           = __funnelshift_r(v.y, v.z, ofs);
    v.z           = __funnelshift_r(v.z, next, ofs);
  }
  nanos = v.y;
  nanos <<= 32;
  nanos |= v.x;
  // Convert from Julian day at noon to UTC seconds
  days = static_cast<int32_t>(v.z);
  cudf::duration_D d_d{
    days - 2440588};  // TBD: Should be noon instead of midnight, but this matches pyarrow

  *dst = [&]() {
    switch (s->setup.col.ts_clock_rate) {
      case 1:  // seconds
        return duration_cast<duration_s>(d_d).count() +
               duration_cast<duration_s>(duration_ns{nanos}).count();
      case 1'000:  // milliseconds
        return duration_cast<duration_ms>(d_d).count() +
               duration_cast<duration_ms>(duration_ns{nanos}).count();
      case 1'000'000:  // microseconds
        return duration_cast<duration_us>(d_d).count() +
               duration_cast<duration_us>(duration_ns{nanos}).count();
      case 1'000'000'000:  // nanoseconds
      default: return duration_cast<cudf::duration_ns>(d_d).count() + nanos;
    }
  }();
}

/**
 * @brief Output a 64-bit timestamp
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename state_buf>
inline __device__ void read_int64_timestamp(auto* s, state_buf* sb, int src_pos, int64_t* dst)
{
  uint8_t const* src8;
  uint32_t dict_pos, dict_size = s->stream.dict_size, ofs;
  int64_t ts;

  if (s->stream.dict_base) {
    // Dictionary
    dict_pos = (s->stream.dict_bits > 0)
                 ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)]
                 : 0;
    src8     = s->stream.dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    src8     = s->stream.data_start;
  }
  dict_pos *= (uint32_t)s->output_cvt.dtype_len_in;
  ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos + 4 < dict_size) {
    uint2 v;
    int64_t val;
    v.x = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 8);
      v.x           = __funnelshift_r(v.x, v.y, ofs);
      v.y           = __funnelshift_r(v.y, next, ofs);
    }
    val = v.y;
    val <<= 32;
    val |= v.x;
    // Output to desired clock rate
    ts = apply_ts_scale(val, s->output_cvt.ts_scale);
  } else {
    ts = 0;
  }
  *dst = ts;
}

/**
 * @brief Output a byte array as int.
 *
 * @param[in] ptr Pointer to the byte array
 * @param[in] len Byte array length
 * @param[out] dst Pointer to row output data
 */
template <typename T>
__device__ void gpuOutputByteArrayAsInt(char const* ptr, int32_t len, T* dst)
{
  T unscaled = 0;
  for (auto i = 0; i < len; i++) {
    uint8_t v = ptr[i];
    unscaled  = (unscaled << 8) | v;
  }
  // Shift the unscaled value up and back down when it isn't all 8 bytes,
  // which sign extend the value for correctly representing negative numbers.
  unscaled <<= (sizeof(T) - len) * 8;
  unscaled >>= (sizeof(T) - len) * 8;
  *dst = unscaled;
}

/**
 * @brief Output a fixed-length byte array as int.
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename T, typename state_buf>
__device__ void read_fixed_width_byte_array_as_int(auto* s, state_buf* sb, int src_pos, T* dst)
{
  uint32_t const dtype_len_in = s->output_cvt.dtype_len_in;
  uint8_t const* data         = s->stream.dict_base ? s->stream.dict_base : s->stream.data_start;
  uint32_t const pos =
    (s->stream.dict_base
       ? ((s->stream.dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)]
                                    : 0)
       : src_pos) *
    dtype_len_in;
  uint32_t const dict_size = s->stream.dict_size;

  T unscaled = 0;
  for (unsigned int i = 0; i < dtype_len_in; i++) {
    uint32_t v = (pos + i < dict_size) ? data[pos + i] : 0;
    unscaled   = (unscaled << 8) | v;
  }
  // Shift the unscaled value up and back down when it isn't all 8 bytes,
  // which sign extend the value for correctly representing negative numbers.
  if (dtype_len_in < sizeof(T)) {
    unscaled <<= (sizeof(T) - dtype_len_in) * 8;
    unscaled >>= (sizeof(T) - dtype_len_in) * 8;
  }
  *dst = unscaled;
}

/**
 * @brief Output a small fixed-length value
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename T, typename state_buf>
inline __device__ void read_fixed_width_value_fast(auto* s, state_buf* sb, int src_pos, T* dst)
{
  uint8_t const* dict;
  uint32_t dict_pos, dict_size = s->stream.dict_size;

  if (s->stream.dict_base) {
    // Dictionary
    dict_pos = (s->stream.dict_bits > 0)
                 ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)]
                 : 0;
    dict     = s->stream.dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    dict     = s->stream.data_start;
  }
  dict_pos *= (uint32_t)s->output_cvt.dtype_len_in;
  gpuStoreOutput(dst, dict, dict_pos, dict_size);
}

/**
 * @brief Output a N-byte value
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst8 Pointer to row output data
 * @param[in] len Length of element
 */
template <typename state_buf>
inline __device__ void read_nbyte_fixed_width_value(
  auto* s, state_buf* sb, int src_pos, uint8_t* dst8, int len)
{
  uint8_t const* dict;
  uint32_t dict_pos, dict_size = s->stream.dict_size;

  if (s->stream.dict_base) {
    // Dictionary
    dict_pos = (s->stream.dict_bits > 0)
                 ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)]
                 : 0;
    dict     = s->stream.dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    dict     = s->stream.data_start;
  }
  dict_pos *= (uint32_t)s->output_cvt.dtype_len_in;
  if (len & 3) {
    // Generic slow path
    for (unsigned int i = 0; i < len; i++) {
      dst8[i] = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
    }
  } else {
    // Copy 4 bytes at a time
    uint8_t const* src8 = dict;
    unsigned int ofs    = 3 & reinterpret_cast<size_t>(src8);
    src8 -= ofs;  // align to 32-bit boundary
    ofs <<= 3;    // bytes -> bits
    for (unsigned int i = 0; i < len; i += 4) {
      uint32_t bytebuf;
      if (dict_pos < dict_size) {
        bytebuf = *reinterpret_cast<uint32_t const*>(src8 + dict_pos);
        if (ofs) {
          uint32_t bytebufnext = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
          bytebuf              = __funnelshift_r(bytebuf, bytebufnext, ofs);
        }
      } else {
        bytebuf = 0;
      }
      dict_pos += 4;
      *reinterpret_cast<uint32_t*>(dst8 + i) = bytebuf;
    }
  }
}

/**
 * Output a BYTE_STREAM_SPLIT value of type `T`.
 *
 * Data is encoded as N == sizeof(T) streams of length M, forming an NxM sized matrix.
 * Rows are streams, columns are individual values.
 *
 * @param dst pointer to output data
 * @param src pointer to first byte of input data in stream 0
 * @param stride number of bytes per input stream (M)
 */
template <typename T>
__device__ inline void gpuOutputByteStreamSplit(uint8_t* dst, uint8_t const* src, size_type stride)
{
  for (int i = 0; i < sizeof(T); i++) {
    dst[i] = src[i * stride];
  }
}

/**
 * Output a 64-bit BYTE_STREAM_SPLIT encoded timestamp.
 *
 * Data is encoded as N streams of length M, forming an NxM sized matrix. Rows are streams,
 * columns are individual values.
 *
 * @param dst pointer to output data
 * @param src pointer to first byte of input data in stream 0
 * @param stride number of bytes per input stream (M)
 * @param ts_scale timestamp scale
 */
inline __device__ void gpuOutputSplitInt64Timestamp(int64_t* dst,
                                                    uint8_t const* src,
                                                    size_type stride,
                                                    int32_t ts_scale)
{
  gpuOutputByteStreamSplit<int64_t>(reinterpret_cast<uint8_t*>(dst), src, stride);
  *dst = apply_ts_scale(*dst, ts_scale);
}

/**
 * Output a BYTE_STREAM_SPLIT encoded decimal as an integer type.
 *
 * Data is encoded as N streams of length M, forming an NxM sized matrix. Rows are streams,
 * columns are individual values.
 *
 * @param dst pointer to output data
 * @param src pointer to first byte of input data in stream 0
 * @param stride number of bytes per input stream (M)
 * @param dtype_len_in length of the `FIXED_LEN_BYTE_ARRAY` used to represent the decimal
 */
template <typename T>
__device__ void gpuOutputSplitFixedLenByteArrayAsInt(T* dst,
                                                     uint8_t const* src,
                                                     size_type stride,
                                                     uint32_t dtype_len_in)
{
  T unscaled = 0;
  // fixed_len_byte_array decimals are big endian
  for (unsigned int i = 0; i < dtype_len_in; i++) {
    unscaled = (unscaled << 8) | src[i * stride];
  }
  // Shift the unscaled value up and back down when it isn't all 8 bytes,
  // which sign extend the value for correctly representing negative numbers.
  if (dtype_len_in < sizeof(T)) {
    unscaled <<= (sizeof(T) - dtype_len_in) * 8;
    unscaled >>= (sizeof(T) - dtype_len_in) * 8;
  }
  *dst = unscaled;
}

}  // namespace cudf::io::parquet::detail
