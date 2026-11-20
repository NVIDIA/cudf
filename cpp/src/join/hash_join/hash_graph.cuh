/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

#include <cuda/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/utility>

namespace cudf::detail {

/// The hash and index of a build row. A probe compares the stored hash before it reads the row.
using hash_graph_record = cuda::std::pair<hash_value_type, size_type>;

/// Multiply-high maps a 32-bit hash into [0, num_buckets) without division or modulo.
__device__ inline cuda::std::uint32_t hash_graph_bucket(hash_value_type hash,
                                                        cuda::std::uint32_t num_buckets)
{
  return cuda::mul_hi(hash, num_buckets);
}

/// Device-side view of the HashGraph. Build rows are grouped by hash bucket, and bucket `b`
/// holds `records[offsets[b], offsets[b + 1])`. Rows with different keys share a bucket, so
/// every probe compares against all records of its bucket.
struct hash_graph_ref {
  size_type const* offsets;
  hash_graph_record const* records;
  cuda::std::uint32_t num_buckets;

  /// Calls `on_match` with the build row of every record equal to `key`
  template <typename Equal, typename OnMatch>
  __device__ void for_each_match(hash_graph_record key, Equal equal, OnMatch on_match) const
  {
    auto const bucket = hash_graph_bucket(key.first, num_buckets);
    auto const end    = offsets[bucket + 1];
    for (auto position = offsets[bucket]; position < end; ++position) {
      auto const record = records[position];
      if (equal(key, record)) { on_match(record.second); }
    }
  }
};

}  // namespace cudf::detail
