/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_shared_memory_aggs.hpp"
#include "global_memory_aggregator.cuh"
#include "helpers.cuh"
#include "shared_memory_aggregator.cuh"
#include "single_pass_functors.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cooperative_groups.h>
#include <cuda/std/cstddef>

namespace cudf::groupby::detail::hash {
namespace {
/// Shared memory data alignment
CUDF_HOST_DEVICE cudf::size_type constexpr ALIGNMENT = 16;

// Allocates shared memory required for output columns. Exits if there is insufficient memory to
// perform shared memory aggregation for the current output column.
__device__ void calculate_columns_to_aggregate(cudf::size_type& col_start,
                                               cudf::size_type& col_end,
                                               cudf::mutable_table_device_view output_values,
                                               cudf::size_type output_size,
                                               cudf::size_type* shmem_agg_res_offsets,
                                               cudf::size_type* shmem_agg_mask_offsets,
                                               cudf::size_type cardinality,
                                               cudf::size_type total_agg_size)
{
  col_start                       = col_end;
  cudf::size_type bytes_allocated = 0;

  auto const valid_col_size =
    cudf::util::round_up_safe(static_cast<cudf::size_type>(sizeof(bool) * cardinality), ALIGNMENT);

  while (bytes_allocated < total_agg_size && col_end < output_size) {
    auto const col_idx = col_end;
    auto const next_col_size =
      cudf::util::round_up_safe(cudf::type_dispatcher<cudf::dispatch_storage_type>(
                                  output_values.column(col_idx).type(), size_of_functor{}) *
                                  cardinality,
                                ALIGNMENT);
    auto const next_col_total_size = next_col_size + valid_col_size;

    if (bytes_allocated + next_col_total_size > total_agg_size) { break; }

    shmem_agg_res_offsets[col_end]  = bytes_allocated;
    shmem_agg_mask_offsets[col_end] = bytes_allocated + next_col_size;

    bytes_allocated += next_col_total_size;
    ++col_end;
  }
}

// Each block initialize its own shared memory aggregation results
__device__ void initialize_shmem_aggregations(cooperative_groups::thread_block const& block,
                                              cudf::size_type col_start,
                                              cudf::size_type col_end,
                                              cudf::mutable_table_device_view output_values,
                                              cuda::std::byte* shmem_agg_storage,
                                              cudf::size_type* shmem_agg_res_offsets,
                                              cudf::size_type* shmem_agg_mask_offsets,
                                              cudf::size_type cardinality,
                                              cudf::aggregation::Kind const* d_agg_kinds)
{
  for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
    for (auto idx = block.thread_rank(); idx < cardinality; idx += block.num_threads()) {
      auto target =
        reinterpret_cast<cuda::std::byte*>(shmem_agg_storage + shmem_agg_res_offsets[col_idx]);
      auto target_mask =
        reinterpret_cast<bool*>(shmem_agg_storage + shmem_agg_mask_offsets[col_idx]);
      cudf::detail::dispatch_type_and_aggregation(output_values.column(col_idx).type(),
                                                  d_agg_kinds[col_idx],
                                                  initialize_shmem{},
                                                  target,
                                                  target_mask,
                                                  idx);
    }
  }
}

__device__ void compute_pre_aggregations(cudf::size_type col_start,
                                         cudf::size_type col_end,
                                         bitmask_type const* row_bitmask,
                                         cudf::table_device_view source,
                                         cudf::size_type num_input_rows,
                                         cudf::size_type* local_mapping_index,
                                         cuda::std::byte* shmem_agg_storage,
                                         cudf::size_type* shmem_agg_res_offsets,
                                         cudf::size_type* shmem_agg_mask_offsets,
                                         cudf::aggregation::Kind const* d_agg_kinds,
                                         cudf::size_type agg_location_offset)
{
  // Aggregates global memory sources to shared memory targets
  for (auto source_idx = cudf::detail::grid_1d::global_thread_id(); source_idx < num_input_rows;
       source_idx += cudf::detail::grid_1d::grid_stride()) {
    if (not row_bitmask or cudf::bit_is_set(row_bitmask, source_idx)) {
      auto const target_idx = local_mapping_index[source_idx] + agg_location_offset;
      for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
        auto const source_col = source.column(col_idx);

        cuda::std::byte* target =
          reinterpret_cast<cuda::std::byte*>(shmem_agg_storage + shmem_agg_res_offsets[col_idx]);
        bool* target_mask =
          reinterpret_cast<bool*>(shmem_agg_storage + shmem_agg_mask_offsets[col_idx]);

        cudf::detail::dispatch_type_and_aggregation(source_col.type(),
                                                    d_agg_kinds[col_idx],
                                                    shmem_element_aggregator{},
                                                    target,
                                                    target_mask,
                                                    target_idx,
                                                    source_col,
                                                    source_idx);
      }
    }
  }
}

__device__ void compute_final_aggregations(cooperative_groups::thread_block const& block,
                                           cudf::size_type col_start,
                                           cudf::size_type col_end,
                                           cudf::table_device_view input_values,
                                           cudf::mutable_table_device_view target,
                                           cudf::size_type cardinality,
                                           cudf::size_type num_agg_locations,
                                           cudf::size_type* global_mapping_index,
                                           cuda::std::byte* shmem_agg_storage,
                                           cudf::size_type* agg_res_offsets,
                                           cudf::size_type* agg_mask_offsets,
                                           cudf::aggregation::Kind const* d_agg_kinds)
{
  // Aggregates shared memory sources to global memory targets
  for (auto idx = block.thread_rank(); idx < num_agg_locations; idx += block.num_threads()) {
    auto const target_idx =
      global_mapping_index[(block.group_index().x * GROUPBY_CARDINALITY_THRESHOLD) +
                           (idx % cardinality)];
    for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
      auto target_col = target.column(col_idx);

      cuda::std::byte* source =
        reinterpret_cast<cuda::std::byte*>(shmem_agg_storage + agg_res_offsets[col_idx]);
      bool* source_mask = reinterpret_cast<bool*>(shmem_agg_storage + agg_mask_offsets[col_idx]);

      cudf::detail::dispatch_type_and_aggregation(input_values.column(col_idx).type(),
                                                  d_agg_kinds[col_idx],
                                                  gmem_element_aggregator{},
                                                  target_col,
                                                  target_idx,
                                                  input_values.column(col_idx),
                                                  source,
                                                  source_mask,
                                                  idx);
    }
  }
}

/* Takes the local_mapping_index and global_mapping_index to compute
 * pre (shared) and final (global) aggregates*/
CUDF_KERNEL void single_pass_shmem_aggs_kernel(cudf::size_type num_rows,
                                               bitmask_type const* row_bitmask,
                                               cudf::size_type* local_mapping_index,
                                               cudf::size_type* global_mapping_index,
                                               cudf::size_type* block_cardinality,
                                               cudf::table_device_view input_values,
                                               cudf::mutable_table_device_view output_values,
                                               cudf::aggregation::Kind const* d_agg_kinds,
                                               cudf::size_type total_agg_size,
                                               cudf::size_type offsets_size)
{
  auto const block       = cooperative_groups::this_thread_block();
  auto const cardinality = block_cardinality[block.group_index().x];
  if (cardinality > GROUPBY_CARDINALITY_THRESHOLD or cardinality == 0) { return; }

  auto constexpr min_shmem_agg_locations = 32;
  auto const multiplication_factor       = min_shmem_agg_locations / cardinality;
  auto const num_agg_locations =
    cuda::std::max(multiplication_factor, decltype(multiplication_factor){1}) * cardinality;
  auto const agg_location_offset =
    multiplication_factor > 1 ? (block.thread_rank() % multiplication_factor) * cardinality : 0;

  auto const num_cols = output_values.num_columns();

  __shared__ cudf::size_type col_start;
  __shared__ cudf::size_type col_end;
  extern __shared__ cuda::std::byte shmem_agg_storage[];

  cudf::size_type* shmem_agg_res_offsets =
    reinterpret_cast<cudf::size_type*>(shmem_agg_storage + total_agg_size);
  cudf::size_type* shmem_agg_mask_offsets =
    reinterpret_cast<cudf::size_type*>(shmem_agg_storage + total_agg_size + offsets_size);

  if (block.thread_rank() == 0) {
    col_start = 0;
    col_end   = 0;
  }
  // Workaround: use __syncthreads() instead of block.sync() throughout this
  // kernel. cooperative_groups::thread_block::sync() does not properly fence
  // shared memory on sm_120 with CUDA 13.2, causing init stores to be
  // invisible to subsequent phases.
  __syncthreads();

  while (col_end < num_cols) {
    __syncthreads();
    if (block.thread_rank() == 0) {
      calculate_columns_to_aggregate(col_start,
                                     col_end,
                                     output_values,
                                     num_cols,
                                     shmem_agg_res_offsets,
                                     shmem_agg_mask_offsets,
                                     num_agg_locations,
                                     total_agg_size);
    }
    __syncthreads();

    initialize_shmem_aggregations(block,
                                  col_start,
                                  col_end,
                                  output_values,
                                  shmem_agg_storage,
                                  shmem_agg_res_offsets,
                                  shmem_agg_mask_offsets,
                                  num_agg_locations,
                                  d_agg_kinds);
    __syncthreads();

    compute_pre_aggregations(col_start,
                             col_end,
                             row_bitmask,
                             input_values,
                             num_rows,
                             local_mapping_index,
                             shmem_agg_storage,
                             shmem_agg_res_offsets,
                             shmem_agg_mask_offsets,
                             d_agg_kinds,
                             agg_location_offset);
    __syncthreads();

    compute_final_aggregations(block,
                               col_start,
                               col_end,
                               input_values,
                               output_values,
                               cardinality,
                               num_agg_locations,
                               global_mapping_index,
                               shmem_agg_storage,
                               shmem_agg_res_offsets,
                               shmem_agg_mask_offsets,
                               d_agg_kinds);
  }
}
}  // namespace

size_type get_available_shared_memory_size(cudf::size_type grid_size)
{
  auto const active_blocks_per_sm =
    cudf::util::div_rounding_up_safe(grid_size, cudf::detail::num_multiprocessors());

  size_t dynamic_shmem_size = 0;
  CUDF_CUDA_TRY(cudaOccupancyAvailableDynamicSMemPerBlock(
    &dynamic_shmem_size, single_pass_shmem_aggs_kernel, active_blocks_per_sm, GROUPBY_BLOCK_SIZE));
  return cudf::util::round_down_safe(static_cast<cudf::size_type>(0.5 * dynamic_shmem_size),
                                     ALIGNMENT);
}

int32_t max_active_blocks_shmem_aggs_kernel()
{
  int32_t max_active_blocks{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, single_pass_shmem_aggs_kernel, GROUPBY_BLOCK_SIZE, 0));
  return max_active_blocks;
}

void compute_shared_memory_aggs(cudf::size_type grid_size,
                                size_type available_shmem_size,
                                cudf::size_type num_input_rows,
                                bitmask_type const* row_bitmask,
                                cudf::size_type* local_mapping_index,
                                cudf::size_type* global_mapping_index,
                                cudf::size_type* block_cardinality,
                                cudf::table_device_view input_values,
                                cudf::mutable_table_device_view output_values,
                                cudf::aggregation::Kind const* d_agg_kinds,
                                rmm::cuda_stream_view stream)
{
  // For each aggregation, need one offset determining where the aggregation is
  // performed, another indicating the validity of the aggregation
  auto const offsets_size = compute_shmem_offsets_size(output_values.num_columns());
  // The rest of shmem is utilized for the actual arrays in shmem
  CUDF_EXPECTS(available_shmem_size > offsets_size * 2,
               "No enough space for shared memory aggregations");
  auto const shmem_agg_size = available_shmem_size - offsets_size * 2;
  single_pass_shmem_aggs_kernel<<<grid_size, GROUPBY_BLOCK_SIZE, available_shmem_size, stream>>>(
    num_input_rows,
    row_bitmask,
    local_mapping_index,
    global_mapping_index,
    block_cardinality,
    input_values,
    output_values,
    d_agg_kinds,
    shmem_agg_size,
    offsets_size);
  CUDF_CUDA_TRY(cudaGetLastError());
}
}  // namespace cudf::groupby::detail::hash
