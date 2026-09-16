/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {
std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> bools_to_mask(
  column_view const& input, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == type_id::BOOL8, "Input is not of type bool");

  if (input.is_empty()) { return std::pair(std::make_unique<rmm::device_buffer>(), 0); }

  auto input_device_view_ptr = column_device_view::create(input, stream);
  auto input_device_view     = *input_device_view_ptr;
  auto pred                  = [] __device__(bool element) { return element; };
  if (input.nullable()) {
    // Nulls are considered false
    auto input_begin = make_null_replacement_iterator<bool>(input_device_view, false);

    auto mask = detail::valid_if(input_begin, input_begin + input.size(), pred, stream, mr);

    return std::pair(std::make_unique<rmm::device_buffer>(std::move(mask.first)), mask.second);
  } else {
    auto mask = detail::valid_if(
      input_device_view.begin<bool>(), input_device_view.end<bool>(), pred, stream, mr);

    return std::pair(std::make_unique<rmm::device_buffer>(std::move(mask.first)), mask.second);
  }
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> bools_to_mask(
  column_view const& input, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::bools_to_mask(input, stream, mr);
}

}  // namespace cudf
