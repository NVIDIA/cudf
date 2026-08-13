/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compute_column_kernel.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::detail {
/**
 * @brief Kernel for evaluating an expression on a table to produce a new column.
 *
 * This evaluates an expression over a table to produce a new column. Also called an n-ary
 * transform.
 *
 * @tparam max_block_size The size of the thread block, used to set launch
 * bounds and minimize register usage.
 * @tparam has_null Indicates whether the output column may contain nulls.
 * @tparam has_complex_type Indicates whether the output column may contain complex types.
 *
 * @param table The table device view used for evaluation.
 * @param device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param output_column The destination for the results of evaluating the expression.
 */
template <cudf::size_type max_block_size, bool has_null, bool has_complex_type>
__launch_bounds__(max_block_size) CUDF_KERNEL
  void compute_column_kernel(table_device_view const table,
                             ast::detail::expression_device_view device_expression_data,
                             mutable_column_device_view output_column)
{
  // The (required) extern storage of the shared memory array leads to
  // conflicting declarations between different templates. The easiest
  // workaround is to declare an arbitrary (here char) array type then cast it
  // after the fact to the appropriate type.
  extern __shared__ char raw_intermediate_storage[];
  ast::detail::IntermediateDataType<has_null>* intermediate_storage =
    reinterpret_cast<ast::detail::IntermediateDataType<has_null>*>(raw_intermediate_storage);

  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];
  auto start_idx    = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  auto evaluator    = cudf::ast::detail::expression_evaluator<has_null, has_complex_type>(
    table, device_expression_data);

  for (thread_index_type row_index = start_idx; row_index < table.num_rows(); row_index += stride) {
    auto output_dest = ast::detail::mutable_column_expression_result<has_null>(output_column);
    evaluator.evaluate(output_dest, row_index, thread_intermediate_storage);
  }
}

// Template function to launch the appropriate kernel based on has_nulls and has_complex_type
template <bool HasNull, bool HasComplexType>
void launch_compute_column_kernel(table_device_view const& table_device,
                                  ast::detail::expression_device_view device_expression_data,
                                  mutable_column_device_view& mutable_output_device,
                                  cudf::detail::grid_1d const& config,
                                  size_t shmem_per_block,
                                  rmm::cuda_stream_view stream)
{
  compute_column_kernel<MAX_BLOCK_SIZE, HasNull, HasComplexType>
    <<<config.num_blocks, config.num_threads_per_block, shmem_per_block, stream.value()>>>(
      table_device, device_expression_data, mutable_output_device);
  CUDF_CUDA_TRY(cudaGetLastError());
}
}  // namespace cudf::detail
