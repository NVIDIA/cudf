/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>

namespace cudf::detail {

/// A row index and the high bits of its hash share one 32-bit hash table slot.
using hash_table_slot_type = cuda::std::uint32_t;

/// Device-side view of the open-addressed table. The low bits hold the representative build
/// row, and the remaining bits hold a hash fingerprint. Fingerprint matches always undergo
/// a full row comparison; reducing the fingerprint width cannot introduce false matches.
/// For N build rows, bit_width(N) bits encode the row index. The all-ones row index is
/// outside [0, N), leaving the all-ones slot available as the empty sentinel.
struct hash_table_ref {
  hash_table_slot_type* slots;
  cuda::std::uint32_t capacity;
  cuda::std::uint32_t row_mask;
  cuda::fast_mod_div<cuda::std::uint32_t> modulo;

  template <typename Equal>
  __device__ bool equal(cuda::std::pair<hash_value_type, size_type> key,
                        hash_table_slot_type slot,
                        Equal check_row_equality) const
  {
    return ((key.first ^ slot) & ~row_mask) == 0 &&
           check_row_equality(key, {key.first, static_cast<size_type>(slot & row_mask)});
  }

  template <typename Equal>
  __device__ size_type insert(cuda::std::pair<hash_value_type, size_type> key,
                              Equal equal_rows) const
  {
    auto const desired = (key.first & ~row_mask) | static_cast<cuda::std::uint32_t>(key.second);
    auto slot          = key.first % modulo;
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto slot_ref =
        cuda::atomic_ref<hash_table_slot_type, cuda::thread_scope_device>{slots[slot]};
      auto old = cuda::std::numeric_limits<hash_table_slot_type>::max();
      if (slot_ref.compare_exchange_strong(old, desired, cuda::memory_order_relaxed)) {
        return key.second;
      }
      if (equal(key, old, equal_rows)) { return static_cast<size_type>(old & row_mask); }
      ++slot;
      if (slot == capacity) { slot = 0; }
    }
    return CUDF_SIZE_TYPE_SENTINEL;
  }

  template <bool IsBuild = false, typename Equal>
  __device__ size_type find(cuda::std::pair<hash_value_type, size_type> key, Equal equal_rows) const
  {
    auto slot = key.first % modulo;
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto const current = slots[slot];
      if (current == cuda::std::numeric_limits<hash_table_slot_type>::max()) {
        return CUDF_SIZE_TYPE_SENTINEL;
      }
      // Under null_equality::UNEQUAL a nested row containing nulls need not equal itself.
      // The fill pass must still find the row that claimed this slot during construction.
      if constexpr (IsBuild) {
        if (static_cast<size_type>(current & row_mask) == key.second) { return key.second; }
      }
      if (equal(key, current, equal_rows)) { return static_cast<size_type>(current & row_mask); }
      ++slot;
      if (slot == capacity) { slot = 0; }
    }
    return CUDF_SIZE_TYPE_SENTINEL;
  }
};

/// CSR segments are indexed by representative build row, including zero-length segments for
/// rows that did not claim a hash table slot. This avoids an offset for every empty hash slot.
struct csr_ref {
  size_type const* offsets;
  size_type const* values;

  __device__ size_type begin(size_type row) const { return offsets[row]; }

  __device__ size_type size(size_type row) const { return offsets[row + 1] - offsets[row]; }
};

}  // namespace cudf::detail
