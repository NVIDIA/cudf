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
 * @brief Get the memory resource to be used for pageable memory allocations.
 *
 * @return Reference to the pageable memory resource
 */
CUDF_EXPORT rmm::host_async_resource_ref get_pageable_memory_resource();

/**
 * @brief Get the allocator to be used for the host memory allocation.
 *
 * @param size The number of elements of type T to allocate
 * @param stream The stream to use for the allocation
 * @return The allocator to be used for the host memory allocation
 */
template <typename T>
rmm_host_allocator<T> get_host_allocator(std::size_t size, cuda::stream_ref stream)
{
  if (size * sizeof(T) <= get_allocate_host_as_pinned_threshold()) {
    return {get_pinned_memory_resource(), stream};
  }
  return {get_pageable_memory_resource(), stream};
}

}  // namespace cudf::detail
