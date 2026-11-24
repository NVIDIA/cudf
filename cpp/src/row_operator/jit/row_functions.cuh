/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "jit/abi.cuh"

namespace cudf_abi {

/** @brief A scalar value of the specified ABI element type. */
template <typename Element>
struct plain_layout {};

/** @brief A single-level list of scalar values. */
template <typename Element>
struct list_layout {};

/** @brief Dictionary indirection to scalar keys. */
template <typename Element>
struct dictionary_layout {};

/** @brief Dictionary indirection to single-level lists of scalar values. */
template <typename Element>
struct dictionary_list_layout {};

#define CUDF_FOR_EACH_VALUE(Layout, DO_IT) \
  DO_IT(Layout<int8_t>);                   \
  DO_IT(Layout<int16_t>);                  \
  DO_IT(Layout<int32_t>);                  \
  DO_IT(Layout<int64_t>);                  \
  DO_IT(Layout<__int128>);                 \
  DO_IT(Layout<uint8_t>);                  \
  DO_IT(Layout<uint16_t>);                 \
  DO_IT(Layout<uint32_t>);                 \
  DO_IT(Layout<uint64_t>);                 \
  DO_IT(Layout<float32_t>);                \
  DO_IT(Layout<float64_t>);                \
  DO_IT(Layout<bool8_t>);                  \
  DO_IT(Layout<string_view>);              \
  DO_IT(Layout<decimal32>);                \
  DO_IT(Layout<decimal64>);                \
  DO_IT(Layout<decimal128>);               \
  DO_IT(Layout<timestamp_D>);              \
  DO_IT(Layout<timestamp_s>);              \
  DO_IT(Layout<timestamp_ms>);             \
  DO_IT(Layout<timestamp_us>);             \
  DO_IT(Layout<timestamp_ns>);             \
  DO_IT(Layout<duration_D>);               \
  DO_IT(Layout<duration_s>);               \
  DO_IT(Layout<duration_ms>);              \
  DO_IT(Layout<duration_us>);              \
  DO_IT(Layout<duration_ns>);

#define CUDF_FOR_EACH_COLUMN_LAYOUT(DO_IT)      \
  CUDF_FOR_EACH_VALUE(plain_layout, DO_IT)      \
  CUDF_FOR_EACH_VALUE(list_layout, DO_IT)       \
  CUDF_FOR_EACH_VALUE(dictionary_layout, DO_IT) \
  CUDF_FOR_EACH_VALUE(dictionary_list_layout, DO_IT)

__device__ uint32_t hash_combine(uint32_t a, uint32_t b);

__device__ uint64_t hash_combine(uint64_t a, uint64_t b);

__device__ uint128_t hash_combine(uint128_t a, uint128_t b);

#define CUDF_DECLARE_ROW_MURMURHASH3_X86_32(Layout) \
  __device__ uint32_t murmurhash3_x86_32(           \
    void const* column, size_type row, uint32_t seed, bool has_nulls, Layout)

#define CUDF_DECLARE_ROW_MURMURHASH3_X64_128(Layout) \
  __device__ uint128_t murmurhash3_x64_128(          \
    void const* column, size_type row, uint64_t seed, bool has_nulls, Layout)

#define CUDF_DECLARE_ROW_XXHASH_32(Layout) \
  __device__ uint32_t xxhash_32(           \
    void const* column, size_type row, uint32_t seed, bool has_nulls, Layout)

#define CUDF_DECLARE_ROW_XXHASH_64(Layout) \
  __device__ uint64_t xxhash_64(           \
    void const* column, size_type row, uint64_t seed, bool has_nulls, Layout)

#define CUDF_DECLARE_ROW_SPARK_MURMURHASH3_X86_32(Layout) \
  __device__ uint32_t spark_murmurhash3_x86_32(           \
    void const* column, size_type row, uint32_t seed, bool has_nulls, Layout)

CUDF_FOR_EACH_COLUMN_LAYOUT(CUDF_DECLARE_ROW_MURMURHASH3_X86_32)
CUDF_FOR_EACH_COLUMN_LAYOUT(CUDF_DECLARE_ROW_MURMURHASH3_X64_128)
CUDF_FOR_EACH_COLUMN_LAYOUT(CUDF_DECLARE_ROW_SPARK_MURMURHASH3_X86_32)
CUDF_FOR_EACH_COLUMN_LAYOUT(CUDF_DECLARE_ROW_XXHASH_32)
CUDF_FOR_EACH_COLUMN_LAYOUT(CUDF_DECLARE_ROW_XXHASH_64)

}  // namespace cudf_abi
