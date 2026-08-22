/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/reverse.h>

namespace cudf::hashing::detail {

template <typename Key>
  requires(not cudf::is_nested<Key>())
struct Spark_MurmurHash3_x86_32 {
  using result_type = int32_t;

  CUDF_HOST_DEVICE constexpr Spark_MurmurHash3_x86_32() = delete;
  CUDF_HOST_DEVICE constexpr Spark_MurmurHash3_x86_32(result_type seed)
    : m_seed(static_cast<uint32_t>(seed))
  {
  }

  [[nodiscard]] __device__ inline uint32_t fmix32(uint32_t h) const
  {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  [[nodiscard]] __device__ inline uint32_t getblock32(cuda::std::byte const* data,
                                                      cudf::size_type offset) const
  {
    // Read a 4-byte value from the data pointer as individual bytes for safe
    // unaligned access (very likely for string types).
    auto block = reinterpret_cast<uint8_t const*>(data + offset);
    return block[0] | (block[1] << 8) | (block[2] << 16) | (block[3] << 24);
  }

  [[nodiscard]] result_type __device__ inline operator()(Key const& key) const
  {
    return compute(key);
  }

  template <typename T>
  result_type __device__ inline compute(T const& key) const
  {
    return compute_bytes(reinterpret_cast<cuda::std::byte const*>(&key), sizeof(T));
  }

  uint32_t __device__ inline compute_remaining_bytes(cuda::std::byte const* data,
                                                     cudf::size_type len,
                                                     cudf::size_type tail_offset,
                                                     uint32_t h) const
  {
    // Process remaining bytes that do not fill a four-byte chunk using Spark's approach
    // (does not conform to normal MurmurHash3).
    for (auto i = tail_offset; i < len; i++) {
      // We require a two-step cast to get the k1 value from the byte. First,
      // we must cast to a signed int8_t. Then, the sign bit is preserved when
      // casting to uint32_t under 2's complement. Java preserves the sign when
      // casting byte-to-int, but C++ does not.
      uint32_t k1 = static_cast<uint32_t>(cuda::std::to_integer<int8_t>(data[i]));
      k1 *= c1;
      k1 = rotate_bits_left(k1, rot_c1);
      k1 *= c2;
      h ^= k1;
      h = rotate_bits_left(h, rot_c2);
      h = h * 5 + c3;
    }
    return h;
  }

  result_type __device__ compute_bytes(cuda::std::byte const* data, cudf::size_type const len) const
  {
    constexpr cudf::size_type BLOCK_SIZE = 4;
    cudf::size_type const nblocks        = len / BLOCK_SIZE;
    cudf::size_type const tail_offset    = nblocks * BLOCK_SIZE;
    uint32_t h                           = m_seed;

    // Process all four-byte chunks.
    for (cudf::size_type i = 0; i < nblocks; i++) {
      uint32_t k1 = getblock32(data, i * BLOCK_SIZE);
      k1 *= c1;
      k1 = rotate_bits_left(k1, rot_c1);
      k1 *= c2;
      h ^= k1;
      h = rotate_bits_left(h, rot_c2);
      h = h * 5 + c3;
    }

    h = compute_remaining_bytes(data, len, tail_offset, h);

    // Finalize hash.
    h ^= static_cast<uint32_t>(len);
    h = fmix32(h);
    return static_cast<result_type>(h);
  }

 private:
  uint32_t m_seed;
  static constexpr uint32_t c1     = 0xcc9e2d51;
  static constexpr uint32_t c2     = 0x1b873593;
  static constexpr uint32_t c3     = 0xe6546b64;
  static constexpr uint32_t rot_c1 = 15;
  static constexpr uint32_t rot_c2 = 13;
};

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<bool>::operator()(bool const& key) const
  -> result_type
{
  return compute<uint32_t>(key);
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<int8_t>::operator()(int8_t const& key) const
  -> result_type
{
  return compute<uint32_t>(key);
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<uint8_t>::operator()(uint8_t const& key) const
  -> result_type
{
  return compute<uint32_t>(key);
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<int16_t>::operator()(int16_t const& key) const
  -> result_type
{
  return compute<uint32_t>(key);
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<uint16_t>::operator()(uint16_t const& key) const
  -> result_type
{
  return compute<uint32_t>(key);
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<float>::operator()(float const& key) const
  -> result_type
{
  return compute<float>(normalize_nans_and_zeros(key));
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<double>::operator()(double const& key) const
  -> result_type
{
  return compute<double>(normalize_nans_and_zeros(key));
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<cudf::string_view>::operator()(
  cudf::string_view const& key) const -> result_type
{
  auto const data = reinterpret_cast<cuda::std::byte const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const -> result_type
{
  return compute<uint64_t>(key.value());
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const -> result_type
{
  return compute<uint64_t>(key.value());
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const -> result_type
{
  // Generates the Spark MurmurHash3 hash value, mimicking the conversion:
  // java.math.BigDecimal.valueOf(unscaled_value, _scale).unscaledValue().toByteArray()
  // https://github.com/apache/spark/blob/ce5ddad990373636e94071e7cef2f31021add07b/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/hash.scala#L391-L396
  __int128_t const val               = key.value();
  constexpr cudf::size_type key_size = sizeof(__int128_t);
  cuda::std::byte const* data        = reinterpret_cast<cuda::std::byte const*>(&val);

  // Small negative values start with 0xff..., small positive values start with 0x00...
  bool const is_negative           = val < 0;
  cuda::std::byte const zero_value = is_negative ? cuda::std::byte{0xff} : cuda::std::byte{0x00};

  // If the value can be represented with a shorter than 16-byte integer, the
  // leading bytes of the little-endian value are truncated and are not hashed.
  auto const reverse_begin = cuda::std::reverse_iterator(data + key_size);
  auto const reverse_end   = cuda::std::reverse_iterator(data);
  auto const first_nonzero_byte =
    thrust::find_if_not(thrust::seq,
                        reverse_begin,
                        reverse_end,
                        [zero_value](cuda::std::byte const& v) { return v == zero_value; })
      .base();
  // Max handles special case of 0 and -1 which would shorten to 0 length otherwise
  cudf::size_type length =
    cuda::std::max(1, static_cast<cudf::size_type>(cuda::std::distance(data, first_nonzero_byte)));

  // Preserve the 2's complement sign bit by adding a byte back on if necessary.
  // e.g. 0x0000ff would shorten to 0x00ff. The 0x00 byte is retained to
  // preserve the sign bit, rather than leaving an "f" at the front which would
  // change the sign bit. However, 0x00007f would shorten to 0x7f. No extra byte
  // is needed because the leftmost bit matches the sign bit. Similarly for
  // negative values: 0xffff00 --> 0xff00 and 0xffff80 --> 0x80.
  if ((length < key_size) && (is_negative ^ bool(data[length - 1] & cuda::std::byte{0x80}))) {
    ++length;
  }

  // Convert to big endian by reversing the range of nonzero bytes. Only those bytes are hashed.
  __int128_t big_endian_value = 0;
  auto big_endian_data        = reinterpret_cast<cuda::std::byte*>(&big_endian_value);
  thrust::reverse_copy(thrust::seq, data, data + length, big_endian_data);
  return compute_bytes(big_endian_data, length);
}

}  // namespace cudf::hashing::detail
