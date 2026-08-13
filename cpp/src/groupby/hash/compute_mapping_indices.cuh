/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compute_mapping_indices.hpp"
#include "helpers.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cooperative_groups.h>
#include <cuco/static_set_ref.cuh>
#include <cuda/std/atomic>

#include <algorithm>

namespace cudf::groupby::detail::hash {
template <typename SetType>
__device__ void find_local_mapping(cooperative_groups::thread_block const& block,
                                   size_type idx,
                                   size_type num_input_rows,
                                   SetType shared_set,
                                   bitmask_type const* row_bitmask,
                                   size_type* cardinality,
                                   size_type* local_mapping_indices,
                                   size_type* shared_set_indices)
{
  auto const is_valid_input =
    idx < num_input_rows and (not row_bitmask or cudf::bit_is_set(row_bitmask, idx));
  auto const [result_idx, inserted] = [&]() {
    if (is_valid_input) {
      auto const result      = shared_set.insert_and_find(idx);
      auto const matched_idx = *result.first;
      auto const inserted    = result.second;
      if (inserted) {  // inserted a new element
        auto const ref_cardinality =
          cuda::atomic_ref<size_type, cuda::thread_scope_block>{*cardinality};
        auto const shared_set_index = ref_cardinality.fetch_add(1, cuda::std::memory_order_relaxed);

        // The value of `shared_set_index` is before increment, thus if we have
        // `shared_set_index == GROUPBY_CARDINALITY_THRESHOLD` the value of cardinality
        // will be at least `GROUPBY_CARDINALITY_THRESHOLD + 1`.
        // This will trigger fallback to global memory.
        if (shared_set_index >= GROUPBY_CARDINALITY_THRESHOLD) { return cuda::std::pair{0, true}; }

        shared_set_indices[shared_set_index] = idx;
        local_mapping_indices[idx]           = shared_set_index;
      }
      return cuda::std::pair{matched_idx, inserted};
    }
    return cuda::std::pair{0, false};  // dummy values
  }();
  // Syncing the thread block is needed so that updates in `local_mapping_indices` are visible to
  // all threads in the thread block.
  block.sync();
  if (is_valid_input) {
    // element was already in set
    if (!inserted) { local_mapping_indices[idx] = local_mapping_indices[result_idx]; }
  }
}

template <typename SetRef>
__device__ void find_global_mapping(cooperative_groups::thread_block const& block,
                                    size_type cardinality,
                                    SetRef global_set,
                                    size_type* shared_set_indices,
                                    size_type* global_mapping_indices)
{
  // for all unique keys in shared memory hash set, stores their matches in
  // global hash set to `global_mapping_indices`
  for (auto idx = block.thread_rank(); idx < cardinality; idx += block.num_threads()) {
    auto const input_idx = shared_set_indices[idx];
    auto const key_idx   = *global_set.insert_and_find(input_idx).first;

    global_mapping_indices[block.group_index().x * GROUPBY_CARDINALITY_THRESHOLD + idx] = key_idx;
  }
}

/*
 * @brief Inserts keys into the shared memory hash set, and stores the block-wise rank for a given
 * row index in `local_mapping_indices`. If the number of unique keys found in a threadblock exceeds
 * `GROUPBY_CARDINALITY_THRESHOLD`, the threads in that block will exit without updating
 * `global_set` or setting `global_mapping_indices`. Else, we insert the unique keys found to the
 * global hash set, and save the row index of the global sparse table in `global_mapping_indices`.
 */
template <class SetRef>
CUDF_KERNEL void mapping_indices_kernel(size_type num_input_rows,
                                        SetRef global_set,
                                        bitmask_type const* row_bitmask,
                                        size_type* local_mapping_indices,
                                        size_type* global_mapping_indices,
                                        size_type* block_cardinality,
                                        cuda::std::atomic_flag* needs_global_memory_fallback)
{
  __shared__ size_type shared_set_indices[GROUPBY_CARDINALITY_THRESHOLD];

  // Shared set initialization
  __shared__ size_type slots[valid_extent.value()];

  auto raw_set = cuco::static_set_ref{
    cuco::empty_key<size_type>{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    global_set.key_eq(),
    probing_scheme_t{global_set.hash_function()},
    cuco::thread_scope_block,
    cuco::bucket_storage_ref<size_type, GROUPBY_BUCKET_SIZE, decltype(valid_extent)>{valid_extent,
                                                                                     slots}};
  auto shared_set = raw_set.rebind_operators(cuco::insert_and_find);

  auto const block = cooperative_groups::this_thread_block();
  shared_set.initialize(block);

  __shared__ size_type cardinality;
  if (block.thread_rank() == 0) { cardinality = 0; }
  block.sync();

  // All threads in the block will participate in the loop, and sync.
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (auto idx = cudf::detail::grid_1d::global_thread_id();
       idx - block.thread_rank() < num_input_rows;
       idx += stride) {
    find_local_mapping(block,
                       idx,
                       num_input_rows,
                       shared_set,
                       row_bitmask,
                       &cardinality,
                       local_mapping_indices,
                       shared_set_indices);

    block.sync();
    if (cardinality > GROUPBY_CARDINALITY_THRESHOLD) {
      if (block.thread_rank() == 0) {
        needs_global_memory_fallback->test_and_set(cuda::std::memory_order_relaxed);
      }
      break;
    }
  }

  // Insert unique keys from shared to global hash set if block-cardinality
  // doesn't exceed the threshold upper-limit
  if (cardinality <= GROUPBY_CARDINALITY_THRESHOLD) {
    find_global_mapping(block, cardinality, global_set, shared_set_indices, global_mapping_indices);
  }

  if (block.thread_rank() == 0) { block_cardinality[block.group_index().x] = cardinality; }
}

template <class SetRef>
int32_t max_active_blocks_mapping_kernel()
{
  int32_t max_active_blocks{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, mapping_indices_kernel<SetRef>, GROUPBY_BLOCK_SIZE, 0));
  return max_active_blocks;
}

template <class SetRef>
void compute_mapping_indices(size_type grid_size,
                             size_type num_rows,
                             SetRef global_set,
                             bitmask_type const* row_bitmask,
                             size_type* local_mapping_indices,
                             size_type* global_mapping_indices,
                             size_type* block_cardinality,
                             cuda::std::atomic_flag* needs_global_memory_fallback,
                             rmm::cuda_stream_view stream)
{
  mapping_indices_kernel<<<grid_size, GROUPBY_BLOCK_SIZE, 0, stream>>>(
    num_rows,
    global_set,
    row_bitmask,
    local_mapping_indices,
    global_mapping_indices,
    block_cardinality,
    needs_global_memory_fallback);
  CUDF_CUDA_TRY(cudaGetLastError());
}
}  // namespace cudf::groupby::detail::hash
