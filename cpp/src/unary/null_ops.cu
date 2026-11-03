/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "true_if.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>

namespace cudf {
namespace detail {
std::unique_ptr<column> is_null(cudf::column_view const& input,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  auto input_device_view = column_device_view::create(input, stream);
  auto device_view       = *input_device_view;
  auto predicate = [device_view] __device__(auto index) { return (device_view.is_null(index)); };
  return detail::true_if(cuda::counting_iterator<cudf::size_type>{0},
                         cuda::counting_iterator{input.size()},
                         input.size(),
                         predicate,
                         stream,
                         mr);
}

std::unique_ptr<column> is_valid(cudf::column_view const& input,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  auto input_device_view = column_device_view::create(input, stream);
  auto device_view       = *input_device_view;
  auto predicate = [device_view] __device__(auto index) { return device_view.is_valid(index); };
  return detail::true_if(cuda::counting_iterator<cudf::size_type>{0},
                         cuda::counting_iterator{input.size()},
                         input.size(),
                         predicate,
                         stream,
                         mr);
}

}  // namespace detail

std::unique_ptr<column> is_null(cudf::column_view const& input,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_null(input, stream, mr);
}

std::unique_ptr<column> is_valid(cudf::column_view const& input,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_valid(input, stream, mr);
}

}  // namespace cudf
