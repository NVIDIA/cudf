/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cuda/std/climits>

#include <cassert>

/**
 * @file bit.hpp
 * @brief Utilities for bit and bitmask operations.
 */

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Returns the number of bits the given type can hold.
 *
 * @tparam T The type to query
 * @return `sizeof(T)` in bits
 */
template <typename T>
constexpr CUDF_HOST_DEVICE inline std::size_t size_in_bits()
{
  static_assert(CHAR_BIT == 8, "Size of a byte must be 8 bits.");
  return sizeof(T) * CHAR_BIT;
}
}  // namespace detail

/**
 * @addtogroup utility_bitmask
 * @{
 */

/**
 * @brief Returns the index of the word containing the specified bit.
 *
 * @param bit_index The index of the bit to query
 * @return The index of the word containing the specified bit
 */
constexpr CUDF_HOST_DEVICE inline size_type word_index(size_type bit_index)
{
  return bit_index / detail::size_in_bits<bitmask_type>();
}

/**
 * @brief Returns the position within a word of the specified bit.
 *
 * @param bit_index The index of the bit to query
 * @return The position within a word of the specified bit
 */
constexpr CUDF_HOST_DEVICE inline size_type intra_word_index(size_type bit_index)
{
  return bit_index % detail::size_in_bits<bitmask_type>();
}

/**
 * @brief Sets the specified bit to `1`
 *
 * This function is not thread-safe, i.e., attempting to update bits within the
 * same word concurrently from multiple threads results in undefined behavior.
 *
 * @param bitmask The bitmask containing the bit to set
 * @param bit_index Index of the bit to set
 */
CUDF_HOST_DEVICE inline void set_bit_unsafe(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  bitmask[word_index(bit_index)] |= (bitmask_type{1} << intra_word_index(bit_index));
}

/**
 * @brief Sets the specified bit to `0`
 *
 * This function is not thread-safe, i.e., attempting to update bits within the
 * same word concurrently from multiple threads results in undefined behavior.
 *
 * @param bitmask The bitmask containing the bit to clear
 * @param bit_index The index of the bit to clear
 */
CUDF_HOST_DEVICE inline void clear_bit_unsafe(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  bitmask[word_index(bit_index)] &= ~(bitmask_type{1} << intra_word_index(bit_index));
}

/**
 * @brief Indicates whether the specified bit is set to `1`
 *
 * @param bitmask The bitmask containing the bit to test
 * @param bit_index Index of the bit to test
 * @return true The specified bit is `1`
 * @return false  The specified bit is `0`
 */
CUDF_HOST_DEVICE inline bool bit_is_set(bitmask_type const* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  return bitmask[word_index(bit_index)] & (bitmask_type{1} << intra_word_index(bit_index));
}

/**
 * @brief optional-like interface to check if a specified bit of a bitmask is set.
 *
 * @param bitmask The bitmask containing the bit to clear
 * @param bit_index Index of the bit to test
 * @param default_value Value to return if `bitmask` is nullptr
 * @return true The specified bit is `1`
 * @return false  The specified bit is `0`
 * @return `default_value` if `bitmask` is nullptr
 */
CUDF_HOST_DEVICE inline bool bit_value_or(bitmask_type const* bitmask,
                                          size_type bit_index,
                                          bool default_value)
{
  return bitmask != nullptr ? bit_is_set(bitmask, bit_index) : default_value;
}

/**
 * @brief Returns a bitmask word with the `n` least significant bits set.
 *
 * Behavior is undefined if `n < 0` or if `n >= size_in_bits<bitmask_type>()`
 *
 * @param n The number of least significant bits to set
 * @return A bitmask word with `n` least significant bits set
 */
constexpr CUDF_HOST_DEVICE inline bitmask_type set_least_significant_bits(size_type n)
{
  assert(0 <= n && n < static_cast<size_type>(detail::size_in_bits<bitmask_type>()));
  return ((bitmask_type{1} << n) - 1);
}

/**
 * @brief Returns a bitmask word with the `n` most significant bits set.
 *
 * Behavior is undefined if `n < 0` or if `n >= size_in_bits<bitmask_type>()`
 *
 * @param n The number of most significant bits to set
 * @return A bitmask word with `n` most significant bits set
 */
constexpr CUDF_HOST_DEVICE inline bitmask_type set_most_significant_bits(size_type n)
{
  constexpr size_type word_size{detail::size_in_bits<bitmask_type>()};
  assert(0 <= n && n < word_size);
  return ~((bitmask_type{1} << (word_size - n)) - 1);
}

#ifdef __CUDACC__

/**
 * @brief Sets the specified bit to `1`
 *
 * @note This operation requires a global atomic operation. Therefore, it is
 * not recommended to use this function in performance critical regions. When
 * possible, it is more efficient to compute and update an entire word at
 * once using `set_word`.
 *
 * This function is thread-safe.
 *
 * @param bitmask The bitmask containing the bit to set
 * @param bit_index  Index of the bit to set
 */
__device__ inline void set_bit(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  atomicOr(&bitmask[word_index(bit_index)], (bitmask_type{1} << intra_word_index(bit_index)));
}

/**
 * @brief Sets the specified bit to `0`
 *
 * @note This operation requires a global atomic operation. Therefore, it is
 * not recommended to use this function in performance critical regions. When
 * possible, it is more efficient to compute and update an entire element at
 * once using `set_element`.

 * This function is thread-safe.
 *
 * @param bitmask The bitmask to modify
 * @param bit_index  Index of the bit to clear
 */
__device__ inline void clear_bit(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  atomicAnd(&bitmask[word_index(bit_index)], ~(bitmask_type{1} << intra_word_index(bit_index)));
}
#endif
/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
