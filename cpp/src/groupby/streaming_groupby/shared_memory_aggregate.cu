/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shared_memory_aggregate.hpp"
#include "shared_memory_aggregator.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/cstddef>

#include <algorithm>
#include <cstddef>

namespace cudf::groupby::detail::streaming {
namespace {

/// Functor returning the size of a type
struct size_of_fn {
  template <typename T>
  CUDF_HOST_DEVICE constexpr size_type operator()() const noexcept
  {
    return sizeof(T);
  }
};

/// Thread block size of the shared memory aggregation kernel
auto constexpr SHMEM_AGGS_BLOCK_SIZE = 256;

/// Minimum number of input rows per (block, group) pair for which privatizing the whole
/// result table in shared memory is expected to pay off the per-block flush to global memory
auto constexpr MIN_ROWS_PER_BLOCK_GROUP = 8;

/// Maximum number of groups: beyond that, global memory atomics rarely collide and privatizing
/// the result table does not pay off its initialization and flush
auto constexpr MAX_NUM_GROUPS = 4096;

/// Minimum number of resident blocks per SM: with a single block per SM, too few warps are in
/// flight to hide the global memory latency of reading the input rows
auto constexpr MIN_ACTIVE_BLOCKS_PER_SM = 2;

std::size_t compute_shmem_size(table_view const& output_values, size_type num_groups)
{
  auto const mask_size =
    util::round_up_safe(sizeof(bool) * num_groups, std::size_t{SHMEM_ALIGNMENT});
  auto const offsets_size = 2 * sizeof(size_type) * output_values.num_columns();
  std::size_t total       = offsets_size;
  for (auto const& col : output_values) {
    auto const elem_size =
      static_cast<std::size_t>(type_dispatcher<dispatch_storage_type>(col.type(), size_of_fn{}));
    total += util::round_up_safe(elem_size * num_groups, std::size_t{SHMEM_ALIGNMENT}) + mask_size;
  }
  return total;
}

/*
 * Each thread block keeps a private copy of all `num_groups` aggregation results in shared
 * memory, indexed directly by the dense target index of each input row.  After aggregating its
 * share of the input rows, the block atomically merges its partial results into the global
 * result table, which may concurrently be updated by other blocks and other `aggregate()` calls.
 */
CUDF_KERNEL void __launch_bounds__(SHMEM_AGGS_BLOCK_SIZE)
  streaming_shmem_aggs_kernel(size_type num_rows,
                              size_type const* target_indices,
                              size_type num_groups,
                              table_device_view input_values,
                              mutable_table_device_view output_values,
                              aggregation::Kind const* d_agg_kinds,
                              size_type agg_storage_size)
{
  extern __shared__ __align__(SHMEM_ALIGNMENT) cuda::std::byte shmem_agg_storage[];

  auto const num_cols = output_values.num_columns();
  auto* const shmem_agg_res_offsets =
    reinterpret_cast<size_type*>(shmem_agg_storage + agg_storage_size);
  auto* const shmem_agg_mask_offsets = shmem_agg_res_offsets + num_cols;

  if (threadIdx.x == 0) {
    auto const mask_size =
      util::round_up_safe(static_cast<size_type>(sizeof(bool) * num_groups), SHMEM_ALIGNMENT);
    size_type bytes_allocated = 0;
    for (size_type col_idx = 0; col_idx < num_cols; ++col_idx) {
      auto const col_size = util::round_up_safe(
        type_dispatcher<dispatch_storage_type>(output_values.column(col_idx).type(), size_of_fn{}) *
          num_groups,
        SHMEM_ALIGNMENT);
      shmem_agg_res_offsets[col_idx]  = bytes_allocated;
      shmem_agg_mask_offsets[col_idx] = bytes_allocated + col_size;
      bytes_allocated += col_size + mask_size;
    }
  }
  __syncthreads();

  for (size_type col_idx = 0; col_idx < num_cols; ++col_idx) {
    auto* const target = shmem_agg_storage + shmem_agg_res_offsets[col_idx];
    auto* const target_mask =
      reinterpret_cast<bool*>(shmem_agg_storage + shmem_agg_mask_offsets[col_idx]);
    for (size_type idx = threadIdx.x; idx < num_groups; idx += blockDim.x) {
      dispatch_shmem_aggregation(input_values.column(col_idx).type(),
                                 d_agg_kinds[col_idx],
                                 initialize_shmem_fn{},
                                 target,
                                 target_mask,
                                 idx);
    }
  }
  __syncthreads();

  for (auto row_idx = cudf::detail::grid_1d::global_thread_id(); row_idx < num_rows;
       row_idx += cudf::detail::grid_1d::grid_stride()) {
    auto const source_idx = static_cast<size_type>(row_idx);
    auto const target_idx = target_indices[source_idx];
    if (target_idx == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) { continue; }
    for (size_type col_idx = 0; col_idx < num_cols; ++col_idx) {
      dispatch_shmem_aggregation(
        input_values.column(col_idx).type(),
        d_agg_kinds[col_idx],
        shmem_element_aggregator{},
        shmem_agg_storage + shmem_agg_res_offsets[col_idx],
        reinterpret_cast<bool*>(shmem_agg_storage + shmem_agg_mask_offsets[col_idx]),
        target_idx,
        input_values.column(col_idx),
        source_idx);
    }
  }
  __syncthreads();

  for (size_type idx = threadIdx.x; idx < num_groups; idx += blockDim.x) {
    for (size_type col_idx = 0; col_idx < num_cols; ++col_idx) {
      dispatch_shmem_aggregation(
        input_values.column(col_idx).type(),
        d_agg_kinds[col_idx],
        gmem_element_aggregator{},
        output_values.column(col_idx),
        idx,
        shmem_agg_storage + shmem_agg_res_offsets[col_idx],
        reinterpret_cast<bool*>(shmem_agg_storage + shmem_agg_mask_offsets[col_idx]),
        idx);
    }
  }
}

}  // namespace

bool shared_memory_aggregate(size_type num_rows,
                             size_type const* target_indices,
                             size_type num_groups,
                             table_view const& input_values,
                             table_view const& output_values,
                             table_device_view const& d_input_values,
                             mutable_table_device_view const& d_output_values,
                             host_span<aggregation::Kind const> agg_kinds,
                             aggregation::Kind const* d_agg_kinds,
                             cuda::stream_ref stream)
{
  if (num_rows == 0 || num_groups == 0 || num_groups > MAX_NUM_GROUPS) { return false; }

  auto const num_sms = cudf::detail::num_multiprocessors();
  // Every block flushes all `num_groups` partial results, so cap the grid such that each
  // (block, group) pair still covers enough input rows to amortize the flush.
  auto const max_flush_blocks =
    static_cast<int64_t>(num_rows) / (static_cast<int64_t>(num_groups) * MIN_ROWS_PER_BLOCK_GROUP);
  // Too few blocks to saturate the device: the global memory path is expected to be faster.
  if (max_flush_blocks < num_sms) { return false; }

  // Dictionary values and SUM_OVERFLOW are not supported by the shared memory aggregators.
  for (std::size_t i = 0; i < agg_kinds.size(); ++i) {
    if (is_dictionary(input_values.column(i).type()) || agg_kinds[i] == aggregation::SUM_OVERFLOW) {
      return false;
    }
  }

  auto const max_shmem_per_block = [] {
    int device_id{};
    CUDF_CUDA_TRY(cudaGetDevice(&device_id));
    int value{};
    CUDF_CUDA_TRY(
      cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
    return static_cast<std::size_t>(value);
  }();
  auto const shmem_size = compute_shmem_size(output_values, num_groups);
  if (shmem_size > max_shmem_per_block) { return false; }
  auto const agg_storage_size =
    static_cast<size_type>(shmem_size - 2 * sizeof(size_type) * output_values.num_columns());

  // The attribute is shared by all launches of the kernel on the device, including those from
  // concurrent `aggregate()` calls, so always raise it to the device maximum rather than to this
  // launch's size: a smaller value set by another caller could otherwise fail this launch.
  if (shmem_size > 48 * 1024) {
    CUDF_CUDA_TRY(cudaFuncSetAttribute(streaming_shmem_aggs_kernel,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(max_shmem_per_block)));
  }

  int max_active_blocks{};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, streaming_shmem_aggs_kernel, SHMEM_AGGS_BLOCK_SIZE, shmem_size));
  if (max_active_blocks < MIN_ACTIVE_BLOCKS_PER_SM) { return false; }

  auto const grid_size =
    std::min({static_cast<int64_t>(max_active_blocks) * num_sms,
              util::div_rounding_up_safe<int64_t>(num_rows, SHMEM_AGGS_BLOCK_SIZE),
              max_flush_blocks});
  if (grid_size < num_sms) { return false; }

  streaming_shmem_aggs_kernel<<<static_cast<int>(grid_size),
                                SHMEM_AGGS_BLOCK_SIZE,
                                shmem_size,
                                stream.get()>>>(num_rows,
                                                target_indices,
                                                num_groups,
                                                d_input_values,
                                                d_output_values,
                                                d_agg_kinds,
                                                agg_storage_size);
  CUDF_CUDA_TRY(cudaGetLastError());
  return true;
}

}  // namespace cudf::groupby::detail::streaming
