/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "join/filter_join_indices/filter_join_indices_output_size_kernel.hpp"
#include "join/join_common_utils.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/atomic>

namespace cudf::detail {

/**
 * @brief Fills the per-output contribution counts of `filter_join_indices` without materializing
 *        the filtered index vectors.
 *
 * The total output size is the sum of `output_counts`, which is laid out per join kind so that
 * each entry records how many output rows the corresponding input contributes:
 * - INNER_JOIN: `output_counts` is indexed per input pair; entry `i` is `1` if the predicate
 *   passes and `0` otherwise.
 * - FULL_JOIN: `output_counts` is indexed per input pair; entry `i` is `1` for a preserved pair
 *   (predicate passes or the pair already contains a `JoinNoMatch`) and `2` for a failed valid
 *   pair (which splits into `(left, JoinNoMatch)` and `(JoinNoMatch, right)`).
 * - LEFT_JOIN: `output_counts` is indexed per left row; the kernel atomically accumulates the
 *   number of passing pairs for each left row. Left rows with no passing pair are floored to `1`
 *   by the host afterwards to account for the synthetic `(left, JoinNoMatch)` entry. The buffer
 *   must be zero-initialized before the launch.
 */
template <bool has_nulls, bool has_complex_type>
CUDF_KERNEL __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE) void filter_join_indices_output_size_kernel(
  cudf::table_device_view left_table,
  cudf::table_device_view right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::join_kind join_kind,
  cudf::size_type* output_counts)
{
  extern __shared__ char raw_intermediate_storage[];
  auto* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  auto const tid    = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls, has_complex_type>{
    left_table, right_table, device_expression_data};

  for (auto i = tid; i < static_cast<cudf::thread_index_type>(left_indices.size()); i += stride) {
    auto const left_row_index  = left_indices[i];
    auto const right_row_index = right_indices[i];

    bool const has_non_match =
      (left_row_index == cudf::JoinNoMatch || right_row_index == cudf::JoinNoMatch);

    bool predicate_pass = false;
    bool both_valid     = false;
    if (has_non_match) {
      // Outer-join unmatched pair: treat as passing so it is preserved in the output count.
      predicate_pass = true;
    } else if (left_row_index >= 0 && left_row_index < left_table.num_rows() &&
               right_row_index >= 0 && right_row_index < right_table.num_rows()) {
      auto result = cudf::ast::detail::value_expression_result<bool, has_nulls>{};
      evaluator.evaluate(result, left_row_index, right_row_index, 0, thread_intermediate_storage);
      predicate_pass = result.is_valid() && result.value();
      both_valid     = true;
    }

    switch (join_kind) {
      case cudf::join_kind::INNER_JOIN: output_counts[i] = predicate_pass ? 1 : 0; break;
      case cudf::join_kind::FULL_JOIN:
        output_counts[i] = (both_valid && !predicate_pass) ? 2 : 1;
        break;
      case cudf::join_kind::LEFT_JOIN:
        if (predicate_pass && left_row_index >= 0 && left_row_index < left_table.num_rows()) {
          cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> count_ref{
            output_counts[left_row_index]};
          count_ref.fetch_add(1, cuda::memory_order_relaxed);
        }
        break;
      default: break;
    }
  }
}

template <bool has_nulls, bool has_complex_type>
void launch_filter_output_size_kernel(
  cudf::table_device_view const& left_table,
  cudf::table_device_view const& right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::detail::grid_1d const& config,
  std::size_t shmem_per_block,
  cudf::join_kind join_kind,
  cudf::size_type* output_counts,
  rmm::cuda_stream_view stream)
{
  filter_join_indices_output_size_kernel<has_nulls, has_complex_type>
    <<<config.num_blocks, config.num_threads_per_block, shmem_per_block, stream.value()>>>(
      left_table,
      right_table,
      left_indices,
      right_indices,
      device_expression_data,
      join_kind,
      output_counts);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::detail
