/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cudf::detail {

/**
 * @brief Loads an aligned word, zero filling bytes outside the string.
 *
 * The aligned address may precede the string by up to three bytes. Only form and dereference
 * a pointer for bytes inside the string; neither allocation padding nor adjacent rows are required.
 */
__device__ inline uint32_t string_load_aligned_word(string_view str, std::uintptr_t address)
{
  auto const begin = reinterpret_cast<std::uintptr_t>(str.data());
  auto const end   = begin + str.size_bytes();
  if (address >= begin && address + sizeof(uint32_t) <= end) {
    return *reinterpret_cast<uint32_t const*>(address);
  }
  uint32_t word = 0;
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    auto const current = address + byte;
    if (current >= begin && current < end) {
      word |= static_cast<uint32_t>(*reinterpret_cast<uint8_t const*>(current)) << (byte * 8);
    }
  }
  return word;
}

/**
 * @brief Reconstructs four/eight little-endian bytes from aligned 32-bit loads and funnel shifts.
 *
 * Like strings gather's load_uint4, this uses overlapping aligned words. The first/last word
 * uses bounded byte loads when necessary, so arbitrary slices, short strings and tails are safe.
 */
template <int Bytes>
__device__ auto string_load_funnel(string_view str, size_type offset)
  -> std::conditional_t<Bytes == 8, uint64_t, uint32_t>
{
  static_assert(Bytes == 4 || Bytes == 8);
  if (offset >= str.size_bytes()) return 0;
  auto const address = reinterpret_cast<std::uintptr_t>(str.data() + offset);
  auto const shift   = static_cast<unsigned int>(address & 3) * 8;
  auto const aligned = address & ~std::uintptr_t{3};
  auto const first   = string_load_aligned_word(str, aligned);
  if constexpr (Bytes == 4) {
    auto const second = shift ? string_load_aligned_word(str, aligned + 4) : 0;
    return __funnelshift_r(first, second, shift);
  } else {
    auto const second = string_load_aligned_word(str, aligned + 4);
    auto const third  = shift ? string_load_aligned_word(str, aligned + 8) : 0;
    auto const low    = __funnelshift_r(first, second, shift);
    auto const high   = __funnelshift_r(second, third, shift);
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
  }
}

/**
 * @brief Reads complete suffix words after a known-equal cached prefix.
 *
 * The input points inside the original string, after at least four cached bytes. Aligning its
 * first load down therefore stays inside that string. The caller must consume only complete
 * words; only the final overlapping aligned word can need byte loads at the original string end.
 */
template <int Bytes, int CachedBytes>
struct string_funnel_suffix_reader {
  static_assert((Bytes == 4 || Bytes == 8) && CachedBytes >= 4);
  using word_type = std::conditional_t<Bytes == 8, uint64_t, uint32_t>;
  std::uintptr_t address;
  std::uintptr_t end;
  unsigned int shift;
  uint32_t carry{};

  __device__ explicit string_funnel_suffix_reader(string_view suffix)
    : address(reinterpret_cast<std::uintptr_t>(suffix.data()) & ~std::uintptr_t{3}),
      end(reinterpret_cast<std::uintptr_t>(suffix.data()) + suffix.size_bytes()),
      shift(static_cast<unsigned int>(reinterpret_cast<std::uintptr_t>(suffix.data()) & 3) * 8)
  {
    if (shift) carry = *reinterpret_cast<uint32_t const*>(address);
  }

  __device__ uint32_t tail_word(std::uintptr_t next) const
  {
    if (next + 4 <= end) return *reinterpret_cast<uint32_t const*>(next);
    // A complete unaligned suffix word guarantees at least one byte in the overlapping word.
    auto const* p = reinterpret_cast<uint8_t const*>(next);
    uint32_t word = p[0];
    if (next + 1 < end) word |= static_cast<uint32_t>(p[1]) << 8;
    if (next + 2 < end) word |= static_cast<uint32_t>(p[2]) << 16;
    return word;
  }

  __device__ word_type next_full()
  {
    if (shift == 0) {
      auto const first = *reinterpret_cast<uint32_t const*>(address);
      if constexpr (Bytes == 4) {
        address += 4;
        return first;
      } else {
        auto const second = *reinterpret_cast<uint32_t const*>(address + 4);
        address += 8;
        return static_cast<uint64_t>(first) | (static_cast<uint64_t>(second) << 32);
      }
    }
    auto const first = carry;
    if constexpr (Bytes == 4) {
      auto const second = tail_word(address + 4);
      carry             = second;
      address += 4;
      return __funnelshift_r(first, second, shift);
    } else {
      auto const second = *reinterpret_cast<uint32_t const*>(address + 4);
      auto const third  = tail_word(address + 8);
      carry             = third;
      address += 8;
      auto const low  = __funnelshift_r(first, second, shift);
      auto const high = __funnelshift_r(second, third, shift);
      return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
    }
  }
};

/**
 * @brief Loads a zero-padded big-endian prefix, with an evaluation-only byte-loop fallback.
 */
template <typename Integer>
__device__ Integer load_big_endian(string_view const& str, size_type offset, bool funnel = true)
{
  static_assert(std::is_unsigned_v<Integer> && (sizeof(Integer) == 4 || sizeof(Integer) == 8));
  if (funnel) {
    auto const little = string_load_funnel<sizeof(Integer)>(str, offset);
    auto const first  = __byte_perm(static_cast<uint32_t>(little), 0, 0x0123);
    if constexpr (sizeof(Integer) == 4) {
      return first;
    } else {
      auto const second = __byte_perm(static_cast<uint32_t>(little >> 32), 0, 0x0123);
      return (static_cast<uint64_t>(first) << 32) | second;
    }
  }
  constexpr auto width = static_cast<size_type>(sizeof(Integer));
  Integer value        = 0;
  for (size_type byte = 0; byte < width; ++byte) {
    value <<= std::numeric_limits<uint8_t>::digits;
    if (offset + byte < str.size_bytes()) {
      value |= static_cast<Integer>(static_cast<uint8_t>(str.data()[offset + byte]));
    }
  }
  return value;
}

/**
 * @brief Column-bounded prefix loads may read adjacent rows; mask bytes beyond this string.
 *
 * Bounds are device addresses of the allocated character region, not alignment padding.
 */
__device__ inline uint32_t string_load_pool_word(string_view str,
                                                 std::uintptr_t address,
                                                 std::uintptr_t begin,
                                                 std::uintptr_t end)
{
  return address >= begin && address + 4 <= end ? *reinterpret_cast<uint32_t const*>(address)
                                                : string_load_aligned_word(str, address);
}
template <typename Integer>
__device__ Integer
load_big_endian_pool(string_view str, int offset, std::uintptr_t begin, std::uintptr_t end)
{
  static_assert(std::is_unsigned_v<Integer> && (sizeof(Integer) == 4 || sizeof(Integer) == 8));
  constexpr int Bytes = sizeof(Integer);
  if (offset >= str.size_bytes()) return 0;
  auto const address = reinterpret_cast<std::uintptr_t>(str.data() + offset);
  auto const shift   = static_cast<unsigned int>(address & 3) * 8;
  auto const aligned = address & ~std::uintptr_t{3};
  auto const a       = string_load_pool_word(str, aligned, begin, end);
  auto const b    = (Bytes == 8 || shift) ? string_load_pool_word(str, aligned + 4, begin, end) : 0;
  auto const c    = (Bytes == 8 && shift) ? string_load_pool_word(str, aligned + 8, begin, end) : 0;
  uint64_t little = __funnelshift_r(a, b, shift);
  if constexpr (Bytes == 8) little |= uint64_t(__funnelshift_r(b, c, shift)) << 32;
  int valid = min(Bytes, str.size_bytes() - offset);
  if (valid < Bytes) little &= (uint64_t{1} << (valid * 8)) - 1;
  uint32_t first = __byte_perm(uint32_t(little), 0, 0x0123);
  if constexpr (Bytes == 4)
    return first;
  else
    return (uint64_t(first) << 32) | __byte_perm(uint32_t(little >> 32), 0, 0x0123);
}

}  // namespace cudf::detail
