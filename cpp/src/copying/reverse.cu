/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/scan.h>

namespace cudf {
namespace detail {
std::unique_ptr<table> reverse(table_view const& source_table,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  size_type num_rows = source_table.num_rows();
  auto elements      = make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_type>([num_rows] __device__(auto i) {
      return num_rows - i - 1;
    }));
  auto elements_end = elements + source_table.num_rows();

  return gather(source_table, elements, elements_end, out_of_bounds_policy::DONT_CHECK, stream, mr);
}

std::unique_ptr<column> reverse(column_view const& source_column,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  return std::move(
    cudf::detail::reverse(table_view({source_column}), stream, mr)->release().front());
}
}  // namespace detail

std::unique_ptr<table> reverse(table_view const& source_table,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::reverse(source_table, stream, mr);
}

std::unique_ptr<column> reverse(column_view const& source_column,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::reverse(source_column, stream, mr);
}
}  // namespace cudf
