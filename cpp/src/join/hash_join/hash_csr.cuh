/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

#include <cuco/pair.cuh>
#include <cuda/atomic>

#include <cstdint>

namespace cudf::detail {

using hash_csr_key_type            = cuco::pair<hash_value_type, size_type>;
using hash_csr_build_position_type = cuco::pair<std::uint32_t, size_type>;

struct hash_csr_map_view {
  hash_csr_key_type* entries;
  std::uint32_t capacity;
  std::uint32_t mask;

  template <typename Equal>
  __device__ std::uint32_t insert(hash_csr_key_type key, Equal equal) const
  {
    for (std::uint32_t step = 0; step < capacity; ++step) {
      auto const slot = (static_cast<std::uint32_t>(key.first) + step) & mask;
      auto entry_ref =
        cuda::atomic_ref<hash_csr_key_type, cuda::thread_scope_device>{entries[slot]};
      auto old = hash_csr_key_type{hash_value_type{-1}, CUDF_SIZE_TYPE_SENTINEL};
      if (entry_ref.compare_exchange_strong(old, key, cuda::memory_order_relaxed)) { return slot; }
      if (equal(key, old)) { return slot; }
    }
    return capacity;
  }

  template <typename Equal>
  __device__ std::uint32_t find(hash_csr_key_type key, Equal equal) const
  {
    for (std::uint32_t step = 0; step < capacity; ++step) {
      auto const slot    = (static_cast<std::uint32_t>(key.first) + step) & mask;
      auto const current = entries[slot];
      if (current.second == CUDF_SIZE_TYPE_SENTINEL) { return capacity; }
      if (equal(key, current)) { return slot; }
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

  __device__ size_type size(size_type slot) const { return cumulative_ends[slot] - begin(slot); }
};

}  // namespace cudf::detail
