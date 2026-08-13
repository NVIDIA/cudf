/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "join/filter_join_indices/filter_join_indices_kernel.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::detail {

/**
 * @brief Kernel to evaluate predicate on join index pairs and store results
 *
 * This kernel evaluates a predicate on matched pairs from the join indices.
 * Non-match indices from outer joins are always included without predicate evaluation.
 *
 * @tparam max_block_size The size of the thread block, used to set launch bounds
 * @tparam has_nulls Indicates whether the expression may evaluate to null
 * @tparam has_complex_type Indicates whether the expression may contain complex types
 */
template <cudf::size_type max_block_size, bool has_nulls, bool has_complex_type>
CUDF_KERNEL __launch_bounds__(max_block_size) void filter_join_indices_kernel(
  cudf::table_device_view left_table,
  cudf::table_device_view right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  bool* predicate_results)
{
  // Shared memory for intermediate storage
  extern __shared__ char raw_intermediate_storage[];
  auto* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);

  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  auto const tid    = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls, has_complex_type>{
    left_table, right_table, device_expression_data};

  for (cudf::size_type i = tid; i < left_indices.size(); i += stride) {
    auto const left_row_index  = left_indices[i];
    auto const right_row_index = right_indices[i];

    auto result = cudf::ast::detail::value_expression_result<bool, has_nulls>{};

    // Check for non-match sentinels (used by outer joins for unmatched rows)
    bool const has_non_match =
      (left_row_index == cudf::JoinNoMatch || right_row_index == cudf::JoinNoMatch);

    // Mixed join behavior: preserve unmatched rows, filter only matched rows
    if (has_non_match) {
      // Always include unmatched rows (non-match indices) without predicate evaluation
      predicate_results[i] = true;
    } else if (left_row_index >= 0 && left_row_index < left_table.num_rows() &&
               right_row_index >= 0 && right_row_index < right_table.num_rows()) {
      // Valid matched pair - evaluate predicate
      evaluator.evaluate(result, left_row_index, right_row_index, 0, thread_intermediate_storage);
      predicate_results[i] = result.is_valid() && result.value();
    } else {
      // Invalid indices (out of bounds - shouldn't happen in normal mixed join workflow)
      predicate_results[i] = false;
    }
  }
}

/**
 * @brief Template dispatch function to launch the appropriate kernel
 */
template <bool has_nulls, bool has_complex_type>
void launch_filter_gather_map_kernel(
  cudf::table_device_view const& left_table,
  cudf::table_device_view const& right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::detail::grid_1d const& config,
  std::size_t shmem_per_block,
  bool* predicate_results,
  rmm::cuda_stream_view stream)
{
  filter_join_indices_kernel<MAX_BLOCK_SIZE, has_nulls, has_complex_type>
    <<<config.num_blocks, config.num_threads_per_block, shmem_per_block, stream.value()>>>(
      left_table,
      right_table,
      left_indices,
      right_indices,
      device_expression_data,
      predicate_results);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::detail
