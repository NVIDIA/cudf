/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <cuda/std/execution>
#include <cuda/stream>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Helper to batched memcpy specified numbers of bytes from source device iterators to
 * destination device iterators
 *
 * @tparam SrcIterator **[inferred]** The type of device-accessible source addresses iterator
 * @tparam DstIterator **[inferred]** The type of device-accessible destination address iterator
 * @tparam SizeIterator **[inferred]** The type of device-accessible buffer size iterator
 *
 * @param src_iter Device-accessible iterator to source addresses
 * @param dst_iter Device-accessible iterator to destination addresses
 * @param size_iter Device-accessible iterator to the buffer sizes (in bytes)
 * @param num_buffs Number of buffers to be copied
 * @param stream CUDA stream to use
 */
template <typename SrcIterator, typename DstIterator, typename SizeIterator>
void batched_memcpy_async(SrcIterator src_iter,
                          DstIterator dst_iter,
                          SizeIterator size_iter,
                          size_t num_buffs,
                          cuda::stream_ref stream)
{
  auto env =
    cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream},
                              cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                         cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(src_iter, dst_iter, size_iter, num_buffs, env));
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
