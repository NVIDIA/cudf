/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_column_kernel.hpp"
#include "runtime/context.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>

namespace cudf {
namespace detail {
std::unique_ptr<column> compute_column(table_view const& table,
                                       ast::expression const& expr,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  if (get_context().use_jit()) { return compute_column_jit(table, expr, stream, mr); }

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls = expr.may_evaluate_null(table, stream);

  auto const parser = ast::detail::expression_parser{expr, table, has_nulls, stream, mr};

  auto const has_complex_type = parser.has_complex_type();

  auto const output_column_mask_state =
    has_nulls ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED;

  auto output_column = cudf::make_fixed_width_column(
    parser.output_type(), table.num_rows(), output_column_mask_state, stream, mr);
  if (table.num_rows() == 0) { return output_column; }
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);

  // Configure kernel parameters
  auto const& device_expression_data = parser.device_expression_data;
  int device_id;
  CUDF_CUDA_TRY(cudaGetDevice(&device_id));
  int shmem_limit_per_block;
  CUDF_CUDA_TRY(
    cudaDeviceGetAttribute(&shmem_limit_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
  auto const block_size =
    parser.shmem_per_thread != 0
      ? std::min(MAX_BLOCK_SIZE,
                 static_cast<size_type>(shmem_limit_per_block / parser.shmem_per_thread))
      : MAX_BLOCK_SIZE;
  auto const config          = cudf::detail::grid_1d{table.num_rows(), block_size};
  auto const shmem_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  auto table_device = table_device_view::create(table, stream);

  // Execute the kernel with the appropriate template parameters
  if (has_nulls) {
    if (has_complex_type) {
      launch_compute_column_kernel<true, true>(*table_device,
                                               device_expression_data,
                                               *mutable_output_device,
                                               config,
                                               shmem_per_block,
                                               stream);
    } else {
      launch_compute_column_kernel<true, false>(*table_device,
                                                device_expression_data,
                                                *mutable_output_device,
                                                config,
                                                shmem_per_block,
                                                stream);
    }
  } else {
    if (has_complex_type) {
      launch_compute_column_kernel<false, true>(*table_device,
                                                device_expression_data,
                                                *mutable_output_device,
                                                config,
                                                shmem_per_block,
                                                stream);
    } else {
      launch_compute_column_kernel<false, false>(*table_device,
                                                 device_expression_data,
                                                 *mutable_output_device,
                                                 config,
                                                 shmem_per_block,
                                                 stream);
    }
  }

  CUDF_CHECK_CUDA(stream.value());
  output_column->set_null_count(
    cudf::detail::null_count(mutable_output_device->null_mask(), 0, output_column->size(), stream));
  return output_column;
}

}  // namespace detail

std::unique_ptr<column> compute_column(table_view const& table,
                                       ast::expression const& expr,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_column(table, expr, stream, mr);
}

}  // namespace cudf
