/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels_common.cuh"

#include <cudf/types.hpp>

#include <cstdint>
#include <limits>

namespace cudf::detail {

constexpr size_type hash_csr_empty_slot               = size_type{-1};
constexpr std::uint64_t hash_csr_empty_entry          = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t hash_csr_empty_build_position = std::numeric_limits<std::uint64_t>::max();

__device__ inline std::uint64_t pack_hash_csr_entry(hash_value_type hash, size_type row)
{
  return (static_cast<std::uint64_t>(hash) << 32) | static_cast<std::uint32_t>(row);
}

__device__ inline hash_value_type unpack_hash_csr_hash(std::uint64_t value)
{
  return static_cast<hash_value_type>(value >> 32);
}

__device__ inline size_type unpack_hash_csr_row(std::uint64_t value)
{
  return static_cast<size_type>(static_cast<std::uint32_t>(value));
}

__device__ inline std::uint64_t pack_hash_csr_build_position(std::uint32_t slot, size_type rank)
{
  return (static_cast<std::uint64_t>(slot) << 32) | static_cast<std::uint32_t>(rank);
}

__device__ inline std::uint32_t unpack_hash_csr_build_slot(std::uint64_t value)
{
  return static_cast<std::uint32_t>(value >> 32);
}

__device__ inline size_type unpack_hash_csr_build_rank(std::uint64_t value)
{
  return static_cast<size_type>(static_cast<std::uint32_t>(value));
}

struct hash_csr_map_view {
  std::uint64_t* entries;
  std::uint32_t capacity;
  std::uint32_t mask;

  template <typename Equal>
  __device__ std::uint32_t find_or_insert(probe_key_type key, Equal equal) const
  {
    auto const desired = pack_hash_csr_entry(key.first, key.second);
    for (std::uint32_t step = 0; step < capacity; ++step) {
      auto const slot = (static_cast<std::uint32_t>(key.first) + step) & mask;
      auto const old  = atomicCAS(reinterpret_cast<unsigned long long*>(entries + slot),
                                 static_cast<unsigned long long>(hash_csr_empty_entry),
                                 static_cast<unsigned long long>(desired));
      if (old == hash_csr_empty_entry) { return slot; }
      if (unpack_hash_csr_hash(old) == key.first &&
          equal(key, probe_key_type{unpack_hash_csr_hash(old), unpack_hash_csr_row(old)})) {
        return slot;
      }
    }
    return capacity;
  }

  template <typename Equal>
  __device__ std::uint32_t find(probe_key_type key, Equal equal) const
  {
    for (std::uint32_t step = 0; step < capacity; ++step) {
      auto const slot    = (static_cast<std::uint32_t>(key.first) + step) & mask;
      auto const current = entries[slot];
      if (current == hash_csr_empty_entry) { return capacity; }
      if (unpack_hash_csr_hash(current) == key.first &&
          equal(key, probe_key_type{unpack_hash_csr_hash(current), unpack_hash_csr_row(current)})) {
        return slot;
      }
    }
    return capacity;
  }
};

struct hash_csr_view {
  size_type const* cumulative_ends;
  size_type const* values;

  __device__ size_type begin(size_type slot) const
  {
    return slot == 0 ? size_type{0} : cumulative_ends[slot - 1];
  }

  __device__ size_type end(size_type slot) const { return cumulative_ends[slot]; }

  __device__ size_type size(size_type slot) const { return end(slot) - begin(slot); }
};

}  // namespace cudf::detail
