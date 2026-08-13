/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/cuda_device.hpp>

#include <atomic>
#include <iostream>

namespace cudf::prefetch {

namespace detail {

std::atomic_bool& enabled()
{
  static std::atomic_bool value;
  return value;
}

std::atomic_bool& debug()
{
  static std::atomic_bool value;
  return value;
}

cudaError_t prefetch_noexcept(void const* ptr,
                              std::size_t size,
                              rmm::cuda_stream_view stream,
                              rmm::cuda_device_id device_id) noexcept
{
  if (!detail::enabled()) { return cudaSuccess; }

  // Don't try to prefetch nullptrs or empty data. Sometimes libcudf has column
  // views that use nullptrs with a nonzero size as an optimization.
  if (ptr == nullptr) {
    if (detail::debug()) { std::cerr << "Skipping prefetch of nullptr" << std::endl; }
    return cudaSuccess;
  }
  if (size == 0) {
    if (detail::debug()) { std::cerr << "Skipping prefetch of size 0" << std::endl; }
    return cudaSuccess;
  }
  if (detail::debug()) {
    std::cerr << "Prefetching " << size << " bytes at location " << ptr << std::endl;
  }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
  cudaMemLocation location{
    (device_id.value() == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice,
    {device_id.value()}};
  constexpr int flags = 0;
  auto result         = cudaMemPrefetchAsync(ptr, size, location, flags, stream.value());
#else
  auto result = cudaMemPrefetchAsync(ptr, size, device_id.value(), stream.value());
#endif
  // Need to flush the CUDA error so that the context is not corrupted.
  if (result == cudaErrorInvalidValue) { cudaGetLastError(); }
  return result;
}

void prefetch(void const* ptr,
              std::size_t size,
              rmm::cuda_stream_view stream,
              rmm::cuda_device_id device_id)
{
  auto result = prefetch_noexcept(ptr, size, stream, device_id);
  // Ignore cudaErrorInvalidValue because that will be raised if prefetching is
  // attempted on unmanaged memory.
  if ((result != cudaErrorInvalidValue) && (result != cudaSuccess)) {
    std::cerr << "Prefetch failed" << std::endl;
    CUDF_CUDA_TRY(result);
  }
}

}  // namespace detail

void enable() noexcept { detail::enabled() = true; }

void disable() noexcept { detail::enabled() = false; }

void enable_debugging() noexcept { detail::debug() = true; }

void disable_debugging() noexcept { detail::debug() = false; }
}  // namespace cudf::prefetch
