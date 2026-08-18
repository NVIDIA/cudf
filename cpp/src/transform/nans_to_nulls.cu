/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/stream>

namespace cudf {
namespace detail {
struct dispatch_nan_to_null {
  template <typename T>
  std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> operator()(
    column_view const& input, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    requires(std::is_floating_point_v<T>)
  {
    auto input_device_view_ptr = column_device_view::create(input, stream);
    auto input_device_view     = *input_device_view_ptr;

    auto pred = [input_device_view] __device__(cudf::size_type idx) {
      return not(cuda::std::isnan(input_device_view.element<T>(idx)) ||
                 input_device_view.is_null(idx));
    };

    auto mask = detail::valid_if(cuda::counting_iterator<cudf::size_type>{0},
                                 cuda::counting_iterator<cudf::size_type>{input.size()},
                                 pred,
                                 stream,
                                 mr);

    return std::pair(std::make_unique<rmm::device_buffer>(std::move(mask.first)), mask.second);
  }

  template <typename T>
  std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> operator()(
    column_view const& input, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    requires(!std::is_floating_point_v<T>)
  {
    CUDF_FAIL("Input column can't be a non-floating type");
  }
};

std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> nans_to_nulls(
  column_view const& input, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::is_floating_point(input.type()),
               "Input must be a floating point type",
               std::invalid_argument);
  if (input.is_empty()) { return std::pair(std::make_unique<rmm::device_buffer>(), 0); }

  return cudf::type_dispatcher(input.type(), dispatch_nan_to_null{}, input, stream, mr);
}

struct copy_float_data_fn {
  column_view const& input;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;

  template <typename T>
    requires(std::is_floating_point_v<T>)
  std::unique_ptr<rmm::device_buffer> operator()() const
  {
    return std::make_unique<rmm::device_buffer>(
      input.data<T>(), input.size() * sizeof(T), stream, mr);
  }

  template <typename T>
    requires(not std::is_floating_point_v<T>)
  std::unique_ptr<rmm::device_buffer> operator()() const
  {
    CUDF_FAIL("Input must be a floating point type");
  }
};

std::unique_ptr<column> column_nans_to_nulls(column_view const& input,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::is_floating_point(input.type()),
               "Input must be a floating point type",
               std::invalid_argument);
  if (input.is_empty()) { return make_empty_column(input.type()); }

  auto [null_mask, null_count] =
    cudf::type_dispatcher(input.type(), dispatch_nan_to_null{}, input, stream, mr);

  auto data = cudf::type_dispatcher(input.type(), copy_float_data_fn{input, stream, mr});
  return std::make_unique<column>(input.type(),
                                  input.size(),
                                  std::move(*data.release()),
                                  std::move(*null_mask.release()),
                                  null_count);
}
}  // namespace detail

std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> nans_to_nulls(
  column_view const& input, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::nans_to_nulls(input, stream, mr);
}

std::unique_ptr<column> column_nans_to_nulls(column_view const& input,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::column_nans_to_nulls(input, stream, mr);
}

}  // namespace cudf
