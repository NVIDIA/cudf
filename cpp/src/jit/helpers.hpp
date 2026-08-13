/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <jit/cache.hpp>
#include <jit/span.cuh>

#include <algorithm>
#include <span>
#include <variant>
#include <vector>

namespace cudf {
namespace jit {

[[deprecated("Use scalar_column_view instead")]] bool is_scalar(cudf::size_type base_column_size,
                                                                cudf::size_type column_size);

[[deprecated("Use get_projection_size instead")]] typename std::vector<column_view>::const_iterator
get_transform_base_column(std::vector<column_view> const& inputs);

size_type get_projection_size(
  std::span<std::variant<column_view, scalar_column_view> const> inputs);

std::map<uint32_t, std::string> build_ptx_params(std::span<std::string const> output_typenames,
                                                 std::span<std::string const> input_typenames,
                                                 bool has_user_data);

template <typename T>
rmm::device_uvector<T> to_device_vector(std::vector<T> const& host,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<T> device{host.size(), stream, mr};
  cudf::detail::cuda_memcpy_async<T>(device, host, stream);
  return device;
}

template <typename DeviceView, typename ColumnView>
std::tuple<std::vector<std::unique_ptr<DeviceView, std::function<void(DeviceView*)>>>,
           rmm::device_uvector<DeviceView>>
column_views_to_device(std::span<ColumnView const> views,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<DeviceView, std::function<void(DeviceView*)>>> handles;

  std::transform(views.begin(), views.end(), std::back_inserter(handles), [&](auto const& view) {
    return DeviceView::create(view, stream);
  });

  // Use pinned host memory so cuda_memcpy_async takes the device-accessible
  // path instead of cudaMemcpyBatchAsync with deferred source reads.
  auto host_array = detail::make_empty_pinned_vector<DeviceView>(handles.size(), stream);
  for (auto const& h : handles) {
    host_array.push_back(*h);
  }

  rmm::device_uvector<DeviceView> device_array{handles.size(), stream, mr};
  cudf::detail::cuda_memcpy<DeviceView>(device_array, host_array, stream);

  return std::make_tuple(std::move(handles), std::move(device_array));
}

kernel get_udf_kernel(std::string const& source_file,
                      std::string const& kernel_name,
                      std::string const& cuda_source);

rtcx::blob get_udf_kernel_fragment(std::string const& source_file,
                                   std::string const& kernel_name,
                                   std::string const& udf_type);

}  // namespace jit
}  // namespace cudf
