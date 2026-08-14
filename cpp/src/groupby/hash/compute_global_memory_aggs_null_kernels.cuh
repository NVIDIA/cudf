/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_global_memory_aggs_null_kernels.hpp"
#include "single_pass_functors.cuh"

#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

namespace cudf::groupby::detail::hash {

template <typename Index, typename Function>
CUDF_KERNEL void filtered_single_pass_aggs_kernel(Index num_items, Function fn)
{
  auto const idx = static_cast<Index>(cudf::detail::grid_1d::global_thread_id());
  if (idx < num_items) { fn(idx); }
}

template <typename Index, typename Function>
void launch_filtered_single_pass_aggs(Index num_items, Function fn, rmm::cuda_stream_view stream)
{
  if (num_items == 0) { return; }

  // Match the launch geometry used by the original Thrust kernel. A smaller block size causes a
  // significant runtime regression in the hash table operations.
  constexpr auto block_size = 256;
  cudf::detail::grid_1d config{num_items, block_size};
  filtered_single_pass_aggs_kernel<<<config.num_blocks,
                                     config.num_threads_per_block,
                                     0,
                                     stream.value()>>>(num_items, fn);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <bool IsDictionary>
void launch_null_sparse_filtered(nullable_insert_and_find_ref set_ref,
                                 bitmask_type const* row_bitmask,
                                 aggregation::Kind const* aggs,
                                 table_device_view const& input_values,
                                 mutable_table_device_view const& output_values,
                                 size_type num_rows,
                                 rmm::cuda_stream_view stream)
{
  launch_filtered_single_pass_aggs(
    num_rows,
    compute_filtered_single_pass_aggs_sparse_output_fn<IsDictionary, nullable_insert_and_find_ref>{
      set_ref, row_bitmask, aggs, input_values, output_values},
    stream);
}

template <bool IsDictionary>
void launch_null_dense_filtered(size_type const* target_indices,
                                aggregation::Kind const* aggs,
                                table_device_view const& input_values,
                                mutable_table_device_view const& output_values,
                                int64_t num_items,
                                rmm::cuda_stream_view stream)
{
  launch_filtered_single_pass_aggs(num_items,
                                   compute_filtered_single_pass_aggs_dense_output_fn<IsDictionary>{
                                     target_indices, aggs, input_values, output_values},
                                   stream);
}

}  // namespace cudf::groupby::detail::hash
