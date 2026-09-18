/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/stream>

namespace cudf::groupby::detail::hash {

/// One open-addressed slot: the first input row that claimed the slot.
using slot_type = size_type;

/// Each input row records its group count index and its rank among the rows of that group.
using build_position_type = cuda::std::pair<cuda::std::uint32_t, size_type>;

/// Slot recorded for rows that are excluded from the groupby because their keys contain nulls.
constexpr cuda::std::uint32_t hash_csr_no_slot =
  cuda::std::numeric_limits<cuda::std::uint32_t>::max();

constexpr thread_index_type hash_csr_block_size = 256;

/// Inputs with at least this many rows size the set from an estimate of the number of distinct
/// keys instead of the row count; smaller inputs get a set for every row right away.
constexpr size_type hash_csr_min_rows_to_estimate = 1 << 21;
/// Slots of the set the distinct keys of a sample of the rows are counted in.
constexpr cuda::std::uint32_t hash_csr_sample_capacity = 1u << 20;
/// Fewest slots of a set sized from an estimate, so that a handful of hot slots still spread
/// over many cache lines.
constexpr cuda::std::uint32_t hash_csr_min_estimated_capacity = 1u << 18;
/// A probe this long is vanishingly unlikely at a load factor of one half, so it means the set
/// is (nearly) full and the build has to restart with a larger one.
constexpr cuda::std::uint32_t hash_csr_max_probes = 256;

/// Device view of the linearly probed open-addressed set that maps each distinct key to a slot.
struct hash_set_ref {
  slot_type* slots;
  cuda::std::uint32_t capacity;
  cuda::std::uint32_t max_probes;  ///< Probes after which the set is declared full

  /**
   * @brief Returns the slot owned by the key of `row`, claiming an empty slot when the key is new.
   *
   * Most groupby rows repeat a key that is already in the set, so each slot is read before any
   * attempt to claim it and the compare-and-swap only runs on empty slots.
   *
   * @param representative Receives the row stored in the matching slot, or the empty sentinel
   * @return The slot and whether `row` claimed it, or `capacity` when `max_probes` slots were
   * probed without success
   */
  template <typename Equal>
  __device__ cuda::std::pair<cuda::std::uint32_t, bool> insert(size_type row,
                                                               hash_value_type hash_value,
                                                               Equal const& equal,
                                                               size_type& representative) const
  {
    representative = cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    auto slot      = static_cast<cuda::std::uint32_t>(hash_value % capacity);
    for (cuda::std::uint32_t step = 0; step < max_probes; ++step) {
      auto slot_ref = cuda::atomic_ref<slot_type, cuda::thread_scope_device>{slots[slot]};
      auto current  = slot_ref.load(cuda::memory_order_relaxed);
      if (current == cudf::detail::CUDF_SIZE_TYPE_SENTINEL &&
          slot_ref.compare_exchange_strong(current, row, cuda::memory_order_relaxed)) {
        representative = row;
        return {slot, true};
      }
      if (equal(row, current)) {
        representative = current;
        return {slot, false};
      }
      slot = slot + 1 == capacity ? 0 : slot + 1;
    }
    return {capacity, false};
  }
};

/**
 * @brief Inserts every valid row into the set and, when `positions` is given, records the slot
 * of each row and its rank within that slot.
 *
 * Ranks are handed out by one atomic per distinct slot per warp: lanes that landed in the same
 * slot combine their increments, which keeps low-cardinality inputs from serializing on a few
 * counters.
 *
 * When `overflow` is given, a row whose probe runs out of `set.max_probes` slots sets it and
 * the rows still to come are skipped: the caller then rebuilds with a larger set.
 */
template <typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_build_kernel(size_type num_rows,
                                       bitmask_type const* valid_rows,
                                       build_position_type* positions,
                                       size_type* slot_counts,
                                       bool count_by_representative,
                                       hash_set_ref set,
                                       Equal equal,
                                       Hasher hasher,
                                       cuda::std::int32_t* overflow)
{
  auto const block  = cooperative_groups::this_thread_block();
  auto const warp   = cooperative_groups::tiled_partition<cudf::detail::warp_size>(block);
  auto const lane   = warp.thread_rank();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  // One thread checks the cross-block abort flag. A block that starts before an overflow
  // finishes its bounded probes; the host still discards the partial build and retries.
  if (overflow != nullptr) {
    auto const stop =
      block.thread_rank() == 0 &&
      cuda::atomic_ref<cuda::std::int32_t, cuda::thread_scope_device>{*overflow}.load(
        cuda::memory_order_relaxed) != 0;
    if (__syncthreads_or(stop)) { return; }
  }
  // Every lane of a warp runs the same number of iterations and participates in the partition.
  // Lanes past the last row use the sentinel slot.
  for (auto first_row = cudf::detail::grid_1d::global_thread_id() - lane; first_row < num_rows;
       first_row += stride) {
    auto const row = first_row + lane;
    auto const is_active =
      row < num_rows &&
      (valid_rows == nullptr || cudf::bit_is_set(valid_rows, static_cast<size_type>(row)));
    auto slot = hash_csr_no_slot;
    if (is_active) {
      auto const index = static_cast<size_type>(row);
      size_type representative{};
      slot = set.insert(index, hasher(index), equal, representative).first;
      if (slot == set.capacity) {
        cuda::atomic_ref<cuda::std::int32_t, cuda::thread_scope_device>{*overflow}.store(
          1, cuda::memory_order_relaxed);
        slot = hash_csr_no_slot;
      } else if (count_by_representative) {
        slot = static_cast<cuda::std::uint32_t>(representative);
      }
    }
    // Without aggregations only the distinct keys matter, and the set alone provides them.
    if (positions == nullptr) { continue; }

    auto const peers = cooperative_groups::labeled_partition(warp, slot);
    if (slot != hash_csr_no_slot) {
      size_type first_rank{};
      if (peers.thread_rank() == 0) {
        first_rank =
          cuda::atomic_ref<size_type, cuda::thread_scope_device>{slot_counts[slot]}.fetch_add(
            static_cast<size_type>(peers.size()), cuda::memory_order_relaxed);
      }
      first_rank     = peers.shfl(first_rank, 0);
      positions[row] = {slot, first_rank + static_cast<size_type>(peers.thread_rank())};
    } else if (row < num_rows) {
      // Materialize values to avoid binding references to host constants.
      positions[row] = {cuda::std::uint32_t{hash_csr_no_slot},
                        size_type{cudf::detail::CUDF_SIZE_TYPE_SENTINEL}};
    }
  }
}

/**
 * @brief Inserts every `stride`-th valid row into the set and counts the sampled rows and the
 * slots they claim.
 *
 * Sampling estimates distinct-key cardinality so the full HashCSR set can use fewer slots on
 * inputs with repeated keys.
 *
 * Rows are sampled one at a time rather than in runs, since neighboring rows often share a key.
 * `counts[0]` receives the number of valid sampled rows and `counts[1]` the number of distinct
 * keys among them.
 */
template <typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_sample_kernel(size_type num_rows,
                                        size_type stride,
                                        bitmask_type const* valid_rows,
                                        hash_set_ref set,
                                        Equal equal,
                                        Hasher hasher,
                                        size_type* counts)
{
  auto const row = cudf::detail::grid_1d::global_thread_id() * stride;
  auto const is_valid =
    row < num_rows &&
    (valid_rows == nullptr || cudf::bit_is_set(valid_rows, static_cast<size_type>(row)));
  auto is_new = false;
  if (is_valid) {
    auto const index = static_cast<size_type>(row);
    size_type representative{};
    is_new = set.insert(index, hasher(index), equal, representative).second;
  }
  // Every thread of the block reaches both counts, which is what they require.
  auto const num_valid = __syncthreads_count(is_valid);
  auto const num_new   = __syncthreads_count(is_new);
  if (threadIdx.x == 0) {
    cuda::atomic_ref<size_type, cuda::thread_scope_device>{counts[0]}.fetch_add(
      num_valid, cuda::memory_order_relaxed);
    cuda::atomic_ref<size_type, cuda::thread_scope_device>{counts[1]}.fetch_add(
      num_new, cuda::memory_order_relaxed);
  }
}

/// Scatters each valid row to its group: the start offset of its slot plus its rank in the slot.
CUDF_KERNEL void hash_csr_fill_kernel(size_type num_rows,
                                      build_position_type const* positions,
                                      size_type const* slot_offsets,
                                      size_type* grouped_rows)
{
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (auto row = cudf::detail::grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const position = positions[row];
    if (position.first == hash_csr_no_slot) { continue; }
    grouped_rows[slot_offsets[position.first] + position.second] = static_cast<size_type>(row);
  }
}

template <typename Equal, typename Hasher>
void launch_hash_csr_build_kernel(size_type num_rows,
                                  bitmask_type const* valid_rows,
                                  build_position_type* positions,
                                  size_type* slot_counts,
                                  bool count_by_representative,
                                  hash_set_ref set,
                                  Equal equal,
                                  Hasher hasher,
                                  cuda::std::int32_t* overflow,
                                  cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = cudf::detail::grid_1d{num_rows, hash_csr_block_size};
  hash_csr_build_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows,
    valid_rows,
    positions,
    slot_counts,
    count_by_representative,
    set,
    equal,
    hasher,
    overflow);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename Equal, typename Hasher>
void launch_hash_csr_sample_kernel(size_type num_rows,
                                   size_type stride,
                                   bitmask_type const* valid_rows,
                                   hash_set_ref set,
                                   Equal equal,
                                   Hasher hasher,
                                   size_type* counts,
                                   cuda::stream_ref stream)
{
  auto const num_samples = cudf::util::div_rounding_up_safe(num_rows, stride);
  if (num_samples == 0) { return; }
  auto const config = cudf::detail::grid_1d{num_samples, hash_csr_block_size};
  hash_csr_sample_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, stride, valid_rows, set, equal, hasher, counts);
  CUDF_CUDA_TRY(cudaGetLastError());
}

inline void launch_hash_csr_fill_kernel(size_type num_rows,
                                        build_position_type const* positions,
                                        size_type const* slot_offsets,
                                        size_type* grouped_rows,
                                        cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = cudf::detail::grid_1d{num_rows, hash_csr_block_size};
  hash_csr_fill_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, positions, slot_offsets, grouped_rows);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::groupby::detail::hash
