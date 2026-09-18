/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include <vector_types.h>

#include <cstddef>
#include <cstdint>

namespace cudf::hashing::detail {

template <typename Key>
  requires(not cudf::is_nested<Key>())
struct Spark_MurmurHash3_x86_32 {
 private:
  static constexpr std::size_t BLOCK_SIZE = 4;
  static constexpr uint32_t c1            = 0xcc9e2d51;
  static constexpr uint32_t c2            = 0x1b873593;
  static constexpr uint32_t c3            = 0xe6546b64;
  static constexpr uint32_t rot_c1        = 15;
  static constexpr uint32_t rot_c2        = 13;

 public:
  // Unsigned internally, like every other cudf hasher, so the seed and the running hash share one
  // type. `spark_murmurhash3_x86_32` converts back to `int32_t` for its output column, matching
  // Spark's signed `Int` result.
  using result_type = uint32_t;

  CUDF_HOST_DEVICE constexpr Spark_MurmurHash3_x86_32() = delete;
  /// The seed is mixed as an unsigned value, matching `MurmurHash3_x86_32` and the Spark JNI
  /// hasher.
  CUDF_HOST_DEVICE constexpr Spark_MurmurHash3_x86_32(uint32_t seed) : m_seed(seed) {}

  [[nodiscard]] __device__ static inline uint32_t fmix32(uint32_t h)
  {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  /*
   * Mix one four-byte block into the running hash. Spark applies this to every trailing byte as
   * well, which is where it departs from MurmurHash3.
   */
  [[nodiscard]] __device__ static inline uint32_t mix_block(uint32_t k1, uint32_t h)
  {
    k1 *= c1;
    k1 = rotate_bits_left(k1, rot_c1);
    k1 *= c2;
    h ^= k1;
    h = rotate_bits_left(h, rot_c2);
    h = h * 5 + c3;
    return h;
  }

  [[nodiscard]] __device__ static inline uint32_t getblock32(cuda::std::byte const* data,
                                                             std::size_t offset)
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

  /**
   * @brief Hash `length` bytes using four-byte block and signed tail-byte generators
   *
   * Four-byte blocks are mixed whole. Leftover bytes are sign-extended and mixed individually,
   * matching Spark's tail-byte handling.
   */
  template <typename BlockFn, typename ByteFn>
    requires(cuda::std::is_same_v<int8_t, cuda::std::invoke_result_t<ByteFn, std::size_t>>)
  [[nodiscard]] __device__ inline uint32_t hash_blocks(std::size_t length,
                                                       BlockFn extract_block,
                                                       ByteFn extract_byte) const
  {
    auto const nblocks = length / BLOCK_SIZE;
    uint32_t h         = m_seed;

    // Process all four-byte chunks.
    for (std::size_t i = 0; i < nblocks; ++i) {
      h = mix_block(extract_block(i), h);
    }

    // Process remaining bytes that do not fill a four-byte chunk using Spark's approach
    // (does not conform to normal MurmurHash3).
    for (std::size_t i = nblocks * BLOCK_SIZE; i < length; ++i) {
      h = mix_block(static_cast<uint32_t>(extract_byte(i)), h);
    }

    // Finalize hash.
    h ^= static_cast<uint32_t>(length);
    h = fmix32(h);
    return h;
  }

  template <typename T>
    requires(sizeof(T) % BLOCK_SIZE == 0)
  uint32_t __device__ inline compute(T const& key) const
  {
    // A whole number of four-byte blocks with no tail. Mix the words directly in the
    // device's little-endian order, matching `getblock32`.
    auto const words = cuda::std::bit_cast<cuda::std::array<uint32_t, sizeof(T) / BLOCK_SIZE>>(key);
    return hash_blocks(
      sizeof(T), [=](std::size_t i) { return words[i]; }, [](std::size_t) { return int8_t{}; });
  }

  uint32_t __device__ compute_bytes(cuda::std::byte const* data, std::size_t const length) const
  {
    return hash_blocks(
      length,
      [=](std::size_t i) { return getblock32(data, i * BLOCK_SIZE); },
      // Preserve the sign for later widening, like Java does.
      [=](std::size_t i) { return cuda::std::to_integer<int8_t>(data[i]); });
  }

  /**
   * @brief Hash the low `length` bytes of an integral value in little-endian order
   *
   * Four-byte blocks are mixed whole. Leftover bytes are sign-extended and mixed individually,
   * matching Spark's tail-byte handling without staging an intermediate byte buffer.
   */
  template <typename T>
    requires(cuda::std::is_integral_v<T>)
  [[nodiscard]] uint32_t __device__ inline hash_low_bytes(T value, std::size_t length) const
  {
    return hash_blocks(
      length,
      [=](std::size_t i) { return static_cast<uint32_t>(value >> (32 * i)); },
      // Preserve the sign for later widening, like Java does.
      [=](std::size_t i) { return static_cast<int8_t>(value >> (8 * i)); });
  }

 private:
  uint32_t m_seed;
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
  __int128_t const val           = key.value();
  constexpr std::size_t key_size = sizeof(__int128_t);

  // Number of bytes in the minimal two's complement representation, matching
  // `BigInteger.toByteArray().length`, which is `bitLength() / 8 + 1`. Negative values are
  // complemented first so that leading sign bits count as leading zeros. Both 0 and -1 have a
  // bit length of 0 and so keep a single byte.
  auto const magnitude  = static_cast<__uint128_t>(val < 0 ? ~val : val);
  auto const bit_length = 128 - cuda::std::countl_zero(magnitude);
  auto const length     = static_cast<std::size_t>(bit_length / 8) + 1;

  // Spark hashes the big-endian representation, so reverse the bytes and shift the significant
  // ones down. Doing this in registers avoids staging a byte buffer in local memory.
  auto const swap128 = [](__int128_t v) {
    auto const words    = cuda::std::bit_cast<uint4>(v);
    auto const permuted = uint4{.x = __byte_perm(words.w, 0, 0x0123),
                                .y = __byte_perm(words.z, 0, 0x0123),
                                .z = __byte_perm(words.y, 0, 0x0123),
                                .w = __byte_perm(words.x, 0, 0x0123)};
    return cuda::std::bit_cast<__uint128_t>(permuted);
  };
  auto const big_endian = swap128(val) >> (8 * (key_size - length));

  // Hash the low `length` bytes of `big_endian`, matching what `compute_bytes` would do over the
  // equivalent byte buffer.
  return hash_low_bytes(big_endian, length);
}

}  // namespace cudf::hashing::detail
