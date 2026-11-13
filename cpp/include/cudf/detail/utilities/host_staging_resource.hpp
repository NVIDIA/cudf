/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/pinned_memory.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstddef>

namespace cudf::detail {

/**
 * @brief Adapts a pinned resource for CPU-written staging buffers.
 *
 * Allocations are immediately usable by the CPU, independent of pending work on the supplied
 * stream. Deallocation remains stream-ordered so asynchronous copies can finish before storage
 * is reused. Holds an owning copy of the upstream resource.
 */
class host_staging_resource {
 public:
  /**
   * @brief Construct using the configured pinned resource or an explicit upstream resource.
   *
   * @param upstream Pinned resource used for allocation and deallocation
   */
  explicit host_staging_resource(
    rmm::host_device_async_resource_ref upstream = cudf::get_pinned_memory_resource())
    : upstream_{upstream}
  {
  }

  /** @brief Allocate CPU-ready storage. */
  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return upstream_.allocate_sync(bytes, alignment);
  }

  /** @brief Allocate CPU-ready storage without waiting for the supplied stream. */
  void* allocate(cuda::stream_ref, std::size_t bytes, std::size_t alignment)
  {
    return allocate_sync(bytes, alignment);
  }

  /** @brief Release storage synchronously. */
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
  {
    upstream_.deallocate_sync(ptr, bytes, alignment);
  }

  /** @brief Release storage after prior work on the supplied stream. */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    upstream_.deallocate(stream, ptr, bytes, alignment);
  }

  bool operator==(host_staging_resource const&) const = default;

  friend void get_property(host_staging_resource const&, cuda::mr::host_accessible) noexcept {}
  friend void get_property(host_staging_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  cuda::mr::any_resource<cuda::mr::host_accessible, cuda::mr::device_accessible> upstream_;
};

}  // namespace cudf::detail
