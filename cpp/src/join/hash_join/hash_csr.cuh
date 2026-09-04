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
#include <cuda/std/cstdint>

namespace cudf::detail {

/// One open-addressed slot: the row hash and the index of the build row that claimed it.
using hash_table_entry_type = cuco::pair<hash_value_type, size_type>;

/// Where a build row landed: the slot it claimed and its rank among the rows sharing that slot.
/// Computing the rank during the build lets retrieval index straight into the CSR without a
/// second pass.
using build_position_type = cuco::pair<cuda::std::uint32_t, size_type>;

/// Device-side view of the open-addressed table, linearly probed with a power-of-two capacity.
struct hash_table_ref {
  hash_table_entry_type* entries;
  cuda::std::uint32_t capacity;  ///< Power of two, so the probe index is a mask instead of a modulo

  __device__ cuda::std::uint32_t mask() const { return capacity - 1; }

  template <typename Equal>
  __device__ cuda::std::uint32_t insert(hash_table_entry_type key, Equal equal) const
  {
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto const slot = (static_cast<cuda::std::uint32_t>(key.first) + step) & mask();
      auto entry_ref =
        cuda::atomic_ref<hash_table_entry_type, cuda::thread_scope_device>{entries[slot]};
      auto old = hash_table_entry_type{hash_value_type{-1}, size_type{CUDF_SIZE_TYPE_SENTINEL}};
      if (entry_ref.compare_exchange_strong(old, key, cuda::memory_order_relaxed)) { return slot; }
      if (equal(key, old)) { return slot; }
    }
    return capacity;
  }

  template <typename Equal>
  __device__ cuda::std::uint32_t find(hash_table_entry_type key, Equal equal) const
  {
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto const slot    = (static_cast<cuda::std::uint32_t>(key.first) + step) & mask();
      auto const current = entries[slot];
      if (current.second == CUDF_SIZE_TYPE_SENTINEL) { return capacity; }
      if (equal(key, current)) { return slot; }
    }
    return capacity;
  }
};

struct csr_ref {
  size_type const* cumulative_ends;
  size_type const* values;

  __device__ size_type begin(size_type slot) const
  {
    return slot == 0 ? size_type{0} : cumulative_ends[slot - 1];
  }

  __device__ size_type size(size_type slot) const { return cumulative_ends[slot] - begin(slot); }
};

}  // namespace cudf::detail
