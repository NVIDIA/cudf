/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace cudf::io::parquet::detail {

namespace delta {

inline __device__ void put_uleb128(uint8_t*& p, uleb128_t v)
{
  while (v > 0x7f) {
    *(p++) = v | 0x80;
    v >>= 7;
  }
  *(p++) = v;
}

inline __device__ void put_zz128(uint8_t*& p, zigzag128_t v)
{
  zigzag128_t s = (v < 0);
  put_uleb128(p, (v ^ -s) * 2 + s);
}

// A block size of 128, with 4 mini-blocks of 32 values each fits nicely without consuming
// too much shared memory.
// The parquet spec requires block_size to be a multiple of 128, and values_per_mini_block
// to be a multiple of 32.
// TODO: if these are ever made configurable, be sure to fix the page size calculation in
// delta_data_len() (page_enc.cu).
constexpr int block_size            = 128;
constexpr int num_mini_blocks       = 4;
constexpr int values_per_mini_block = block_size / num_mini_blocks;
constexpr int buffer_size           = 2 * block_size;

// An extra sanity checks to enforce compliance with the parquet specification.
static_assert(block_size % 128 == 0);
static_assert(values_per_mini_block % 32 == 0);

__device__ constexpr int rolling_idx(int index) { return rolling_index<buffer_size>(index); }

// Version of bit packer that can handle up to 64 bits values.
// T is the type to use for processing. if nbits <= 32 use uint32_t, otherwise unsigned long long
// (not uint64_t because of atomicOr's typing). allowing this to be selectable since there's a
// measurable impact to using the wider types.
template <typename scratch_type>
inline __device__ void bitpack_mini_block(
  uint8_t* dst, uleb128_t val, uint32_t count, uint8_t nbits, void* temp_space)
{
  using wide_type = cuda::std::
    conditional_t<cuda::std::is_same_v<scratch_type, unsigned long long>, __uint128_t, uint64_t>;
  using cudf::detail::warp_size;
  scratch_type constexpr mask = sizeof(scratch_type) * 8 - 1;
  auto constexpr div          = sizeof(scratch_type) * 8;

  auto const lane_id = threadIdx.x % warp_size;
  auto const warp_id = threadIdx.x / warp_size;

  auto const scratch = reinterpret_cast<scratch_type*>(temp_space) + warp_id * warp_size;

  // zero out scratch
  scratch[lane_id] = 0;
  __syncwarp();

  // TODO: see if there is any savings using special packing for easy bitwidths (1,2,4,8,16...)
  // like what's done for the RLE encoder.
  if (nbits == div) {
    if (lane_id < count) {
      for (int i = 0; i < sizeof(scratch_type); i++) {
        dst[lane_id * sizeof(scratch_type) + i] = val & 0xff;
        val >>= 8;
      }
    }
    return;
  }

  if (lane_id <= count) {
    // Shift symbol left by up to mask bits.
    wide_type v2 = val;
    v2 <<= (lane_id * nbits) & mask;

    // Copy N bit word into two N/2 bit words while following C++ strict aliasing rules.
    scratch_type v1[2];
    memcpy(&v1, &v2, sizeof(wide_type));

    // Atomically write result to scratch.
    if (v1[0]) { atomicOr(scratch + ((lane_id * nbits) / div), v1[0]); }
    if (v1[1]) { atomicOr(scratch + ((lane_id * nbits) / div) + 1, v1[1]); }
  }
  __syncwarp();

  // Copy scratch data to final destination.
  auto const available_bytes = util::div_rounding_up_safe(count * nbits, 8U);
  auto const scratch_bytes   = reinterpret_cast<uint8_t const*>(scratch);

  for (uint32_t i = lane_id; i < available_bytes; i += warp_size) {
    dst[i] = scratch_bytes[i];
  }
  __syncwarp();
}

}  // namespace delta

// Object used to turn a stream of integers into a DELTA_BINARY_PACKED stream. This takes as input
// 128 values with validity at a time, saving them until there are enough values for a block
// to be written.
// T is the input data type (either int32_t or int64_t).
template <typename T>
class delta_binary_packer {
 public:
  using U            = std::make_unsigned_t<T>;
  using block_reduce = cub::BlockReduce<T, delta::block_size>;
  using warp_reduce  = cub::WarpReduce<U>;
  using index_scan   = cub::BlockScan<size_type, delta::block_size>;

 private:
  uint8_t* _dst;                             // sink to dump encoded values to
  T* _buffer;                                // buffer to store values to be encoded
  size_type _current_idx;                    // index of first value in buffer
  uint32_t _num_values;                      // total number of values to encode
  size_type _values_in_buffer;               // current number of values stored in _buffer
  uint8_t _mb_bits[delta::num_mini_blocks];  // bitwidth for each mini-block

  // pointers to shared scratch memory for the warp and block scans/reduces
  index_scan::TempStorage* _scan_tmp;
  typename warp_reduce::TempStorage* _warp_tmp;
  typename block_reduce::TempStorage* _block_tmp;

  void* _bitpack_tmp;  // pointer to shared scratch memory used in bitpacking

  // Write the delta binary header. Only call from thread 0.
  inline __device__ void write_header()
  {
    delta::put_uleb128(_dst, delta::block_size);
    delta::put_uleb128(_dst, delta::num_mini_blocks);
    delta::put_uleb128(_dst, _num_values);
    delta::put_zz128(_dst, _buffer[0]);
  }

  // Write the block header. Only call from thread 0.
  inline __device__ void write_block_header(zigzag128_t block_min)
  {
    delta::put_zz128(_dst, block_min);
    memcpy(_dst, _mb_bits, 4);
    _dst += 4;
  }

  // Signed subtraction with defined wrapping behavior.
  inline __device__ T subtract(T a, T b)
  {
    return static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
  }

 public:
  inline __device__ auto num_values() const { return _num_values; }

  // Initialize the object. Only call from thread 0.
  inline __device__ void init(uint8_t* dest, uint32_t num_values, T* buffer, void* temp_storage)
  {
    _dst              = dest;
    _num_values       = num_values;
    _buffer           = buffer;
    _scan_tmp         = reinterpret_cast<index_scan::TempStorage*>(temp_storage);
    _warp_tmp         = reinterpret_cast<typename warp_reduce::TempStorage*>(temp_storage);
    _block_tmp        = reinterpret_cast<typename block_reduce::TempStorage*>(temp_storage);
    _bitpack_tmp      = _buffer + delta::buffer_size;
    _current_idx      = 0;
    _values_in_buffer = 0;
    _buffer[0]        = 0;
  }

  // Each thread calls this to add its current value.
  inline __device__ void add_value(T value, bool is_valid)
  {
    // Figure out the correct position for the given value.
    size_type const valid = is_valid;
    size_type pos;
    size_type num_valid;
    index_scan(*_scan_tmp).ExclusiveSum(valid, pos, num_valid);

    if (is_valid) { _buffer[delta::rolling_idx(pos + _current_idx + _values_in_buffer)] = value; }
    __syncthreads();

    if (num_valid > 0 && threadIdx.x == 0) {
      _values_in_buffer += num_valid;
      // if first pass write header
      if (_current_idx == 0) {
        write_header();
        _current_idx = 1;
        _values_in_buffer -= 1;
      }
    }
    __syncthreads();

    if (_values_in_buffer >= delta::block_size) { flush(); }
  }

  // Called by each thread to flush data to the sink.
  inline __device__ uint8_t* flush()
  {
    using cudf::detail::warp_size;

    __shared__ T block_min;

    int const t       = threadIdx.x;
    int const warp_id = t / warp_size;
    int const lane_id = t % warp_size;

    // if no values have been written, still need to write the header
    if (t == 0 and _current_idx == 0) { write_header(); }

    __syncthreads();

    // if there are no values to write, just return
    if (_values_in_buffer <= 0) { return _dst; }

    // Calculate delta for this thread.
    size_type const idx = _current_idx + t;
    T const delta       = idx < _num_values ? subtract(_buffer[delta::rolling_idx(idx)],
                                                 _buffer[delta::rolling_idx(idx - 1)])
                                            : cuda::std::numeric_limits<T>::max();

    // Find min delta for the block.
    auto const min_delta = block_reduce(*_block_tmp).Reduce(delta, cuda::minimum{});

    if (t == 0) { block_min = min_delta; }
    __syncthreads();

    // Compute frame of reference for the block.
    U const norm_delta = idx < _num_values ? subtract(delta, block_min) : 0;

    // Get max normalized delta for each warp, and use that to determine how many bits to use
    // for the bitpacking of this warp.
    U const warp_max = warp_reduce(_warp_tmp[warp_id]).Reduce(norm_delta, cuda::maximum{});
    __syncwarp();

    if (lane_id == 0) { _mb_bits[warp_id] = sizeof(long long) * 8 - __clzll(warp_max); }
    __syncthreads();

    // write block header
    if (t == 0) { write_block_header(block_min); }
    __syncthreads();

    // Now each warp encodes its data...can calculate starting offset with _mb_bits.
    // NOTE: using a switch here rather than a loop because the compiler produces code that
    // uses fewer registers.
    int cumulative_bits = 0;
    switch (warp_id) {
      case 3: cumulative_bits += _mb_bits[2]; [[fallthrough]];
      case 2: cumulative_bits += _mb_bits[1]; [[fallthrough]];
      case 1: cumulative_bits += _mb_bits[0];
    }
    uint8_t* const mb_ptr = _dst + cumulative_bits * delta::values_per_mini_block / 8;

    // encoding happens here
    auto const warp_idx = _current_idx + warp_id * delta::values_per_mini_block;
    if (warp_idx < _num_values) {
      auto const num_enc = min(size_type{delta::values_per_mini_block},
                               static_cast<size_type>(_num_values) - warp_idx);
      if (_mb_bits[warp_id] > 32) {
        delta::bitpack_mini_block<unsigned long long>(
          mb_ptr, norm_delta, num_enc, _mb_bits[warp_id], _bitpack_tmp);
      } else {
        delta::bitpack_mini_block<uint32_t>(
          mb_ptr, norm_delta, num_enc, _mb_bits[warp_id], _bitpack_tmp);
      }
    }
    __syncthreads();

    // Last warp updates global delta ptr.
    if (warp_id == delta::num_mini_blocks - 1 && lane_id == 0) {
      _dst              = mb_ptr + _mb_bits[warp_id] * delta::values_per_mini_block / 8;
      _current_idx      = min(warp_idx + size_type{delta::values_per_mini_block},
                         static_cast<size_type>(_num_values));
      _values_in_buffer = max(_values_in_buffer - delta::block_size, size_type{0});
    }
    __syncthreads();

    return _dst;
  }
};

}  // namespace cudf::io::parquet::detail
