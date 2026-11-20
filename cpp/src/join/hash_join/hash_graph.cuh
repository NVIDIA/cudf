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

/// What the count pass records about a probe row for retrieval: the position of its first
/// matching record, or `hash_graph_no_match`. `hash_graph_compare_rows` is set when a record of
/// the bucket shares the row's hash without being a match, so that retrieval has to compare rows
/// again instead of trusting the stored hashes.
using hash_graph_probe_state                             = cuda::std::uint32_t;
constexpr hash_graph_probe_state hash_graph_no_match     = 0xffff'ffff;
constexpr hash_graph_probe_state hash_graph_compare_rows = 0x8000'0000;

/// Device-side view of the HashGraph. Build rows are grouped by hash bucket, and bucket `b`
/// holds `records[offsets[b], offsets[b + 1])`. Rows with different keys share a bucket, so
/// counting the matches of a probe row compares it against all records of its bucket.
struct hash_graph_ref {
  size_type const* offsets;
  hash_graph_record const* records;
  cuda::std::uint32_t num_buckets;
};

}  // namespace cudf::detail
