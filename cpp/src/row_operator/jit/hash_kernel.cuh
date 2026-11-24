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
  static __device__ void assign(void* low, void*, size_type row, uint32_t hash)
  {
    static_cast<uint32_t*>(low)[row] = hash;
  }
};

struct Hash64 {
  static __device__ void assign(void* low, void*, size_type row, uint64_t hash)
  {
    static_cast<uint64_t*>(low)[row] = hash;
  }
};

struct Hash128 {
  static __device__ void assign(void* low, void* high, size_type row, uint128_t hash)
  {
    static_cast<uint64_t*>(low)[row]  = static_cast<uint64_t>(hash);
    static_cast<uint64_t*>(high)[row] = static_cast<uint64_t>(hash >> 64);
  }
};

#define CUDF_DEFINE_ROW_HASHER(Name, Seed)                                              \
  template <int Index, typename Layout>                                                 \
  struct Name {                                                                         \
    static constexpr int index = Index;                                                 \
    static __device__ auto hash(cudf_abi::column_device_view const* column,             \
                                size_type row,                                          \
                                uint64_t seed,                                          \
                                bool has_nulls)                                         \
    {                                                                                   \
      return cudf_abi::Name(column, row, static_cast<Seed>(seed), has_nulls, Layout{}); \
    }                                                                                   \
  }

CUDF_DEFINE_ROW_HASHER(murmurhash3_x86_32, uint32_t);
CUDF_DEFINE_ROW_HASHER(spark_murmurhash3_x86_32, uint32_t);
CUDF_DEFINE_ROW_HASHER(murmurhash3_x64_128, uint64_t);
CUDF_DEFINE_ROW_HASHER(xxhash_32, uint32_t);
CUDF_DEFINE_ROW_HASHER(xxhash_64, uint64_t);

#undef CUDF_DEFINE_ROW_HASHER

template <typename Output, bool SeedChaining, typename Hasher0, typename... Hashers>
__device__ void hash_kernel(cudf_abi::column_device_view const* columns,
                            size_type num_rows,
                            cudf_abi::uint64_t seed,
                            bool has_nulls,
                            void* low,
                            void* high)
{
  auto const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto const start  = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t row = start; row < num_rows; row += stride) {
    auto hash = Hasher0::hash(&columns[Hasher0::index], row, seed, has_nulls);
    if constexpr (SeedChaining) {
      ((hash = Hashers::hash(
          &columns[Hashers::index], row, static_cast<cudf_abi::uint64_t>(hash), has_nulls)),
       ...);
    } else {
      ((hash = hash_combine(hash, Hashers::hash(&columns[Hashers::index], row, seed, has_nulls))),
       ...);
    }
    Output::assign(low, high, row, hash);
  }
}

}  // namespace cudf::row_operators
