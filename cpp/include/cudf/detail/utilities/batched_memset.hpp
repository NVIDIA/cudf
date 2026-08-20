/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cub/device/device_copy.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/transform.h>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Helper to batched memset a host span of device spans to the provided value
 *
 * @param host_buffers Host span of device spans of data
 * @param value Value to memset all device spans to
 * @param stream Stream used for device memory operations and kernel launches
 *
 * @return The data in device spans all set to value
 */
template <typename T>
void batched_memset(cudf::host_span<cudf::device_span<T> const> host_buffers,
                    T const value,
                    rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // Copy buffer spans into device memory and then get sizes
  auto buffers = cudf::detail::make_device_uvector_async(
    host_buffers, stream, cudf::get_current_device_resource_ref());

  // Vector of sizes of all buffer spans
  auto sizes = cuda::transform_iterator(
    buffers.begin(), cuda::proclaim_return_type<std::size_t>([] __device__(auto const& buffer) {
      return buffer.size();
    }));

  // Constant iterator to the value to memset
  auto iter_in = cuda::make_constant_iterator(cuda::make_constant_iterator(value));

  // Iterator to each device span pointer
  auto iter_out = cuda::transform_iterator(
    buffers.begin(),
    cuda::proclaim_return_type<T*>([] __device__(auto const& buffer) { return buffer.data(); }));

  std::size_t temp_storage_bytes = 0;
  auto const num_buffers         = host_buffers.size();

  cub::DeviceCopy::Batched(
    nullptr, temp_storage_bytes, iter_in, iter_out, sizes, num_buffers, stream);

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  cub::DeviceCopy::Batched(
    d_temp_storage.data(), temp_storage_bytes, iter_in, iter_out, sizes, num_buffers, stream);
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
