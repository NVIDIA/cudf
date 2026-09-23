/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/reshape.hpp>
#include <cudf/detail/transpose.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/transpose.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace detail {
std::pair<std::unique_ptr<column>, table_view> transpose(table_view const& input,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  // If there are no rows in the input, return successfully
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    return {std::make_unique<column>(), table_view{}};
  }

  // Check datatype homogeneity
  auto const dtype = input.column(0).type();
  CUDF_EXPECTS(
    std::all_of(
      input.begin(), input.end(), [dtype](auto const& col) { return dtype == col.type(); }),
    "Column type mismatch");

  auto output_column = cudf::detail::interleave_columns(input, stream, mr);
  auto one_iter      = cuda::counting_iterator<size_type>{1};
  auto splits_iter   = cuda::transform_iterator(
    one_iter, [width = input.num_columns()](size_type idx) { return idx * width; });
  auto splits = std::vector<size_type>(splits_iter, splits_iter + input.num_rows() - 1);
  auto output_column_views = detail::split(output_column->view(), splits, stream);

  return {std::move(output_column), table_view(output_column_views)};
}
}  // namespace detail

std::pair<std::unique_ptr<column>, table_view> transpose(table_view const& input,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::transpose(input, stream, mr);
}

}  // namespace cudf
