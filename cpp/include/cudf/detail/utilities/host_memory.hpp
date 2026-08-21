/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <cstddef>

namespace cudf::detail {

/**
 * @brief Property tag for the CUDA native host memory pool.
 *
 * Advertised by `cuda_host_pinned_pool_memory_resource` (the pool backed by
 * `cudaMallocFromPoolAsync` / `cudaFreeAsync`).  The typed resource ref below carries it through
 * type erasure so callers can select the right pool without going through the public
 * `get_pinned_memory_resource()` API.
 */
struct stream_ordered_host_accessible_t {};

/**
 * @brief Typed resource ref for the stream-ordered pinned host pool.
 *
 * Carrying `stream_ordered_host_accessible_t` in the template parameters means callers that
 * construct `rmm_host_allocator` from this ref get a concrete type that still carries the
 * property (before it is erased into `rmm::host_async_resource_ref`).  This is the return type
 * of `get_stream_ordered_pinned_memory_resource()`.
 */
using stream_ordered_host_device_async_resource_ref =
  cuda::mr::resource_ref<cuda::mr::host_accessible,
                         cuda::mr::device_accessible,
                         stream_ordered_host_accessible_t>;

/**
 * @brief Get the memory resource to be used for pageable memory allocations.
 *
 * @return Reference to the pageable memory resource
 */
CUDF_EXPORT rmm::host_async_resource_ref get_pageable_memory_resource();

/**
 * @brief Get the stream-ordered pinned memory resource.
 *
 * The underlying resource uses `cudaMallocFromPoolAsync` / `cudaFreeAsync` so both allocation and
 * deallocation are truly stream-ordered.
 *
 * @return A `stream_ordered_host_device_async_resource_ref` backed by the default pinned pool
 */
CUDF_EXPORT stream_ordered_host_device_async_resource_ref
get_stream_ordered_pinned_memory_resource();

/**
 * @brief Get the allocator to be used for the host memory allocation.
 *
 * @param size The number of elements of type T to allocate
 * @param stream The stream to use for the allocation
 * @return The allocator to be used for the host memory allocation
 */
template <typename T>
rmm_host_allocator<T> get_host_allocator(std::size_t size, rmm::cuda_stream_view stream)
{
  if (size * sizeof(T) <= get_allocate_host_as_pinned_threshold()) {
    return {get_stream_ordered_pinned_memory_resource(), stream};
  }
  return {get_pageable_memory_resource(), stream};
}

}  // namespace cudf::detail
