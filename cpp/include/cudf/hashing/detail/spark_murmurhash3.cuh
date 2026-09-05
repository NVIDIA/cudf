/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/reverse.h>

namespace cudf::hashing::detail {

template <typename Key>
  requires(not cudf::is_nested<Key>())
struct Spark_MurmurHash3_x86_32 {
  // Unsigned internally, like every other cudf hasher, so the seed and the running hash share one
  // type.  `spark_murmurhash3_x86_32` converts back to `int32_t` for its output column, matching
  // Spark's signed `Int` result.
  using result_type = uint32_t;

  CUDF_HOST_DEVICE constexpr Spark_MurmurHash3_x86_32() = delete;
  /// The seed is mixed as an unsigned value, matching `MurmurHash3_x86_32` and the Spark JNI
  /// hasher.  The result stays signed because Spark's hash returns a signed `Int`.
  CUDF_HOST_DEVICE constexpr Spark_MurmurHash3_x86_32(uint32_t seed) : m_seed(seed) {}

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
    // unaligned access (very likely for string types). The bytes are combined in
    // little-endian order as MurmurHash3 defines, unlike `cudf::io::unaligned_load`,
    // which reproduces native byte order and lives in a header this one cannot include.
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
    if constexpr (sizeof(T) % 4 == 0) {
      // A whole number of blocks with no tail.  Hashing the words directly lets the compiler use
      // wide aligned loads instead of reassembling each block byte by byte.  The word order is the
      // device's own, which is little-endian, so this matches `getblock32`.
      auto const words = cuda::std::bit_cast<cuda::std::array<uint32_t, sizeof(T) / 4>>(key);
      uint32_t h       = m_seed;
      for (auto const word : words) {
        h = mix_block(word, h);
      }
      h ^= static_cast<uint32_t>(sizeof(T));
      return static_cast<result_type>(fmix32(h));
    } else {
      return compute_bytes(reinterpret_cast<cuda::std::byte const*>(&key), sizeof(T));
    }
  }

  /*
   * Mix one four-byte block into the running hash.  Spark applies this to every trailing byte as
   * well, which is where it departs from MurmurHash3.
   */
  [[nodiscard]] __device__ inline uint32_t mix_block(uint32_t k1, uint32_t h) const
  {
    k1 *= c1;
    k1 = rotate_bits_left(k1, rot_c1);
    k1 *= c2;
    h ^= k1;
    h = rotate_bits_left(h, rot_c2);
    return h * 5 + c3;
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
      h = mix_block(static_cast<uint32_t>(cuda::std::to_integer<int8_t>(data[i])), h);
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
      h = mix_block(getblock32(data, i * BLOCK_SIZE), h);
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
  // BOOL8 is "0 == false, else true", so canonicalize before hashing: a stored byte of 2 must
  // hash as 1, not as 2.
  return compute<uint32_t>(key ? 1u : 0u);
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
  // Spark hashes the raw bit pattern as a 32-bit integer.
  return compute<uint32_t>(cuda::std::bit_cast<uint32_t>(normalize_nans_and_zeros(key)));
}

template <>
__device__ inline auto Spark_MurmurHash3_x86_32<double>::operator()(double const& key) const
  -> result_type
{
  // Spark hashes the raw bit pattern as a 64-bit integer.
  return compute<uint64_t>(cuda::std::bit_cast<uint64_t>(normalize_nans_and_zeros(key)));
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

  // Number of bytes in the minimal two's complement representation, matching
  // `BigInteger.toByteArray().length`, which is `bitLength() / 8 + 1`.  Negative values are
  // complemented first so that leading sign bits count as leading zeros.  Both 0 and -1 have a
  // bit length of 0 and so keep a single byte.
  auto const magnitude = static_cast<__uint128_t>(val < 0 ? ~val : val);
  auto const mag_hi    = static_cast<cuda::std::uint64_t>(magnitude >> 64);
  auto const mag_lo    = static_cast<cuda::std::uint64_t>(magnitude);
  auto const bit_length =
    mag_hi != 0 ? 128 - cuda::std::countl_zero(mag_hi) : 64 - cuda::std::countl_zero(mag_lo);
  auto const length = static_cast<cudf::size_type>(bit_length / 8) + 1;

  // Spark hashes the big-endian representation, so reverse the bytes and shift the significant
  // ones down.  Doing this in registers avoids staging a byte buffer in local memory.
  auto const swap32 = [](cuda::std::uint32_t v) { return __byte_perm(v, 0, 0x0123); };
  auto const swap64 = [swap32](cuda::std::uint64_t v) {
    return (static_cast<cuda::std::uint64_t>(swap32(static_cast<cuda::std::uint32_t>(v))) << 32) |
           swap32(static_cast<cuda::std::uint32_t>(v >> 32));
  };
  auto const value = static_cast<__uint128_t>(val);
  auto const swapped =
    (static_cast<__uint128_t>(swap64(static_cast<cuda::std::uint64_t>(value))) << 64) |
    swap64(static_cast<cuda::std::uint64_t>(value >> 64));
  auto const big_endian = swapped >> (8 * (key_size - length));

  // Hash the low `length` bytes of `big_endian`, matching what `compute_bytes` would do over the
  // equivalent byte buffer.
  auto const nblocks = length / 4;
  uint32_t h         = m_seed;
  for (cudf::size_type i = 0; i < nblocks; i++) {
    h = mix_block(static_cast<uint32_t>(big_endian >> (32 * i)), h);
  }
  for (cudf::size_type i = nblocks * 4; i < length; i++) {
    h = mix_block(static_cast<uint32_t>(static_cast<cuda::std::int8_t>(big_endian >> (8 * i))), h);
  }
  h ^= static_cast<uint32_t>(length);
  return static_cast<result_type>(fmix32(h));
}

}  // namespace cudf::hashing::detail
