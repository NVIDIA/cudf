/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/join_common_utils.hpp"
#include "join/mixed_join_kernels_semi.cuh"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

namespace cudf {
namespace detail {

template <cudf::size_type block_size, bool has_nulls>
CUDF_KERNEL void __launch_bounds__(block_size)
  mixed_join_semi(table_device_view left_table,
                  table_device_view right_table,
                  table_device_view probe,
                  table_device_view build,
                  row_equality const equality_probe,
                  hash_set_ref_type set_ref,
                  cudf::device_span<bool> left_table_keep_mask,
                  cudf::ast::detail::expression_device_view device_expression_data)
{
  // Normally the casting of a shared memory array is used to create multiple
  // arrays of different types from the shared memory buffer, but here it is
  // used to circumvent conflicts between arrays of different types between
  // different template instantiations due to the extern specifier.
  extern __shared__ char raw_intermediate_storage[];
  auto intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    intermediate_storage + (threadIdx.x * device_expression_data.num_intermediates);

  // Equality evaluator to use
  auto const evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);

  // The cuco API passes parameters in the same (left, right) order we use here,
  // so no swapping needed
  auto constexpr swap_tables = false;
  auto const equality        = single_expression_equality<has_nulls>{
    evaluator, thread_intermediate_storage, swap_tables, equality_probe};

  // Create set ref with the new equality comparator
  auto const set_ref_equality = set_ref.rebind_key_eq(equality);

  // Total number of rows to query the set
  auto const outer_num_rows = left_table.num_rows();
  auto const grid_stride    = cudf::detail::grid_1d::grid_stride<block_size>();

  // Find all the rows in the left table that are in the hash table
  for (auto outer_row_index = cudf::detail::grid_1d::global_thread_id<block_size>();
       outer_row_index < outer_num_rows;
       outer_row_index += grid_stride) {
    left_table_keep_mask[outer_row_index] = set_ref_equality.contains(outer_row_index);
  }
}

void launch_mixed_join_semi(bool has_nulls,
                            table_device_view left_table,
                            table_device_view right_table,
                            table_device_view probe,
                            table_device_view build,
                            row_equality const equality_probe,
                            hash_set_ref_type set_ref,
                            cudf::device_span<bool> left_table_keep_mask,
                            cudf::ast::detail::expression_device_view device_expression_data,
                            detail::grid_1d const config,
                            int64_t shmem_size_per_block,
                            rmm::cuda_stream_view stream)
{
  if (has_nulls) {
    mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        left_table,
        right_table,
        probe,
        build,
        equality_probe,
        set_ref,
        left_table_keep_mask,
        device_expression_data);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
        left_table,
        right_table,
        probe,
        build,
        equality_probe,
        set_ref,
        left_table_keep_mask,
        device_expression_data);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

}  // namespace detail
}  // namespace cudf
