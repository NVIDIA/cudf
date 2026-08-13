/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_global_memory_aggs.hpp"
#include "compute_mapping_indices.hpp"
#include "compute_shared_memory_aggs.hpp"
#include "compute_single_pass_aggs.hpp"
#include "extract_single_pass_aggs.hpp"
#include "helpers.cuh"
#include "output_utils.hpp"

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_set.cuh>
#include <cuda/iterator>
#include <thrust/for_each.h>

namespace cudf::groupby::detail::hash {

template <typename SetType>
std::pair<rmm::device_uvector<size_type>, bool> compute_single_pass_aggs(
  SetType& global_set,
  bitmask_type const* row_bitmask,
  std::span<aggregation_request const> requests,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Collect the single-pass aggregations that can be processed in this function.
  // The compound aggregations that require multiple passes will be handled separately later on.
  auto const [values, agg_kinds, aggs, is_agg_intermediate, has_compound_aggs] =
    extract_single_pass_aggs(requests, stream);
  auto const d_agg_kinds = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, cudf::get_current_device_resource_ref());
  auto const num_rows = values.num_rows();

  // Performs naive global memory aggregations when the workload is not compatible with shared
  // memory, such as when aggregating dictionary columns, when there is insufficient dynamic
  // shared memory for shared memory aggregations, or when SUM_OVERFLOW aggregations are
  // present.
  auto const run_aggs_by_global_mem_kernel = [&] {
    auto [agg_results, unique_key_indices] = compute_global_memory_aggs(
      row_bitmask, values, global_set, agg_kinds, d_agg_kinds, is_agg_intermediate, stream, mr);
    finalize_output(values, aggs, agg_results, cache, stream);
    return std::pair{std::move(unique_key_indices), has_compound_aggs};
  };

  // Grid size used for both index mapping and shared memory aggregation kernels.
  auto const grid_size = [&] {
    auto const max_blocks_mapping =
      max_active_blocks_mapping_kernel<typename SetType::ref_type<cuco::insert_and_find_tag>>();
    auto const max_blocks_aggs = max_active_blocks_shmem_aggs_kernel();
    // We launch the same grid size for both kernels, thus we need to take the minimum of the two.
    auto const max_blocks    = std::min(max_blocks_mapping, max_blocks_aggs);
    auto const max_grid_size = max_blocks * cudf::detail::num_multiprocessors();
    auto const num_blocks    = cudf::util::div_rounding_up_safe(num_rows, GROUPBY_BLOCK_SIZE);
    return std::min(max_grid_size, num_blocks);
  }();

  // grid_size is zero means the shared memory kernel cannot be launched, since input cannot be
  // empty: empty input should already been handled before reaching here.
  if (grid_size <= 0) { return run_aggs_by_global_mem_kernel(); }

  auto const [can_use_shared_mem_kernel, available_shmem_size] =
    is_shared_memory_compatible(agg_kinds, values, grid_size);

  if (!can_use_shared_mem_kernel) { return run_aggs_by_global_mem_kernel(); }

  // Maps from the global row index of the input table to its block-wise rank.
  rmm::device_uvector<size_type> local_mapping_indices(num_rows, stream);
  // Maps from the block-wise rank to the row index of result table.
  rmm::device_uvector<size_type> global_mapping_indices(grid_size * GROUPBY_CARDINALITY_THRESHOLD,
                                                        stream);
  // Initialize it with a sentinel value, so later we can identify which ones are unused and which
  // ones need to be updated.
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    global_mapping_indices.begin(),
    global_mapping_indices.end(),
    cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
  // Compute the cardinality (the number of unique keys) encounter by each thread block.
  rmm::device_uvector<size_type> block_cardinality(grid_size, stream);

  // Flag indicating whether a global memory aggregation fallback is required or not.
  rmm::device_scalar<cuda::std::atomic_flag> needs_global_memory_fallback(stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    needs_global_memory_fallback.data(), 0, sizeof(cuda::std::atomic_flag), stream.value()));

  auto set_ref_insert = global_set.ref(cuco::op::insert_and_find);
  compute_mapping_indices(grid_size,
                          num_rows,
                          set_ref_insert,
                          row_bitmask,
                          local_mapping_indices.data(),
                          global_mapping_indices.data(),
                          block_cardinality.data(),
                          needs_global_memory_fallback.data(),
                          stream);

  auto const needs_fallback = [&] {
    cuda::std::atomic_flag h_needs_fallback;
    // Cannot use `device_scalar::value` as it requires a copy constructor, which
    // `atomic_flag` doesn't have.
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(&h_needs_fallback,
                                             needs_global_memory_fallback.data(),
                                             sizeof(cuda::std::atomic_flag),
                                             stream));
    stream.synchronize();
    return h_needs_fallback.test(cuda::std::memory_order_relaxed);
  }();
  if (needs_fallback) { return run_aggs_by_global_mem_kernel(); }

  auto unique_keys = extract_populated_keys(global_set, num_rows, stream, mr);

  // Now, update the target indices for computing aggregations using the shared memory kernel.
  {
    auto key_transform_map = compute_key_transform_map(
      num_rows, unique_keys, stream, cudf::get_current_device_resource_ref());
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<cudf::size_type>{0},
      grid_size * GROUPBY_BLOCK_SIZE,
      [key_transform_map      = key_transform_map.begin(),
       global_mapping_indices = global_mapping_indices.begin()] __device__(auto const idx) {
        auto const block_id    = idx / GROUPBY_BLOCK_SIZE;
        auto const thread_rank = idx % GROUPBY_BLOCK_SIZE;
        auto const mapping_idx = block_id * GROUPBY_CARDINALITY_THRESHOLD + thread_rank;
        auto const old_idx     = global_mapping_indices[mapping_idx];
        if (old_idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
          global_mapping_indices[mapping_idx] = key_transform_map[old_idx];
        }
      });
  }

  auto const d_spass_values = table_device_view::create(values, stream);
  auto agg_results          = create_results_table(
    static_cast<size_type>(unique_keys.size()), values, agg_kinds, is_agg_intermediate, stream, mr);
  auto d_results_ptr = mutable_table_device_view::create(*agg_results, stream);
  compute_shared_memory_aggs(grid_size,
                             available_shmem_size,
                             num_rows,
                             row_bitmask,
                             local_mapping_indices.data(),
                             global_mapping_indices.data(),
                             block_cardinality.data(),
                             *d_spass_values,
                             *d_results_ptr,
                             d_agg_kinds.data(),
                             stream);

  finalize_output(values, aggs, agg_results, cache, stream);
  return {std::move(unique_keys), has_compound_aggs};
}
}  // namespace cudf::groupby::detail::hash
