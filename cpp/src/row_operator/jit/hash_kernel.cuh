/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "row_operator/jit/row_functions.cuh"

namespace cudf::row_operators {

using cudf_abi::hash_combine;
using cudf_abi::int64_t;
using cudf_abi::size_type;
using cudf_abi::uint128_t;
using cudf_abi::uint32_t;
using cudf_abi::uint64_t;

struct Hash32 {
  using type = uint32_t;

  static __device__ void assign(void* __restrict__ low,
                                void* __restrict__,
                                size_type row,
                                uint32_t hash)
  {
    static_cast<uint32_t*>(low)[row] = hash;
  }
};

struct Hash64 {
  using type = uint64_t;

  static __device__ void assign(void* __restrict__ low,
                                void* __restrict__,
                                size_type row,
                                uint64_t hash)
  {
    static_cast<uint64_t*>(low)[row] = hash;
  }
};

struct Hash128 {
  using type = uint128_t;

  static __device__ void assign(void* __restrict__ low,
                                void* __restrict__ high,
                                size_type row,
                                uint128_t hash)
  {
    static_cast<uint64_t*>(low)[row]  = static_cast<uint64_t>(hash);
    static_cast<uint64_t*>(high)[row] = static_cast<uint64_t>(hash >> 64);
  }
};

#define CUDF_DEFINE_ROW_HASHER(Name, Seed)                                               \
  template <int Index, typename Layout>                                                  \
  struct Name {                                                                          \
    static constexpr int index = Index;                                                  \
                                                                                         \
    static __device__ auto hash(cudf_abi::column_device_view const* __restrict__ column, \
                                size_type row,                                           \
                                uint64_t seed,                                           \
                                bool has_nulls)                                          \
    {                                                                                    \
      return cudf_abi::Name(column, row, static_cast<Seed>(seed), has_nulls, Layout{});  \
    }                                                                                    \
  }

CUDF_DEFINE_ROW_HASHER(murmurhash3_x86_32, uint32_t);
CUDF_DEFINE_ROW_HASHER(spark_murmurhash3_x86_32, uint32_t);
CUDF_DEFINE_ROW_HASHER(murmurhash3_x64_128, uint64_t);
CUDF_DEFINE_ROW_HASHER(xxhash_32, uint32_t);
CUDF_DEFINE_ROW_HASHER(xxhash_64, uint64_t);

#undef CUDF_DEFINE_ROW_HASHER

/**
 * @brief Computes one hash per row using typed column hashers.
 *
 * Each typed hasher corresponds to one input column. Seed-chained algorithms pass each column's
 * result to the next column as its seed. Other algorithms hash columns with the original seed
 * and combine their results in column order.
 *
 * @tparam Output Policy supplying the hash `type` and `assign()` operation
 * @tparam SeedChaining Whether each column's hash becomes the next column's seed
 * @tparam HasNulls Whether hashing must check input validity
 * @tparam SingleRowPerThread Whether each thread processes at most one row; otherwise, threads
 * use a grid-stride loop
 * @tparam Hasher0 First input column's typed hasher
 * @tparam Hashers Remaining input columns' typed hashers, in column order
 * @param columns Input device column views in column order
 * @param num_rows Number of rows to hash
 * @param seed Initial hash seed
 * @param low Output array for 32-bit or 64-bit hashes, or the low 64 bits of 128-bit hashes
 * @param high Output array for the high 64 bits of 128-bit hashes; ignored and may be null for
 * 32-bit and 64-bit hashes
 */
template <typename Output,
          bool SeedChaining,
          bool HasNulls,
          bool SingleRowPerThread,
          typename Hasher0,
          typename... Hashers>
__device__ void hash_kernel(cudf_abi::column_device_view const* __restrict__ columns,
                            size_type num_rows,
                            cudf_abi::uint64_t seed,
                            void* __restrict__ low,
                            void* __restrict__ high)
{
  auto const apply = [&](auto row) {
    typename Output::type hash = Hasher0::hash(&columns[Hasher0::index], row, seed, HasNulls);
    if constexpr (SeedChaining) {
      ((hash = Hashers::hash(&columns[Hashers::index], row, hash, HasNulls)), ...);
    } else {
      ((hash = hash_combine(hash, Hashers::hash(&columns[Hashers::index], row, seed, HasNulls))),
       ...);
    }
    Output::assign(low, high, row, hash);
  };

  // A full grid gives each row its own thread. With fewer blocks, each thread hashes multiple
  // rows, advancing by the total number of threads after each row.
  if constexpr (SingleRowPerThread) {
    auto const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < static_cast<uint32_t>(num_rows)) { apply(static_cast<size_type>(index)); }
  } else {
    auto const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    auto const start  = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int64_t row = start; row < num_rows; row += stride) {
      apply(row);
    }
  }
}

}  // namespace cudf::row_operators
