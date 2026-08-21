/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/host_memory.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>

namespace cudf {

namespace {

// Inlined from RMM internals after public MR definitions moved to source files:
// https://github.com/rapidsai/rmm/pull/2416
void* aligned_host_allocate(std::size_t bytes, std::size_t alignment)
{
  assert(rmm::is_supported_alignment(alignment));

  // allocate memory for bytes, plus potential alignment correction,
  // plus store of the correction offset
  std::size_t padded_allocation_size{bytes + alignment + sizeof(std::ptrdiff_t)};
  char* const original = static_cast<char*>(::operator new(padded_allocation_size));

  // account for storage of offset immediately prior to the aligned pointer
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  void* aligned{original + sizeof(std::ptrdiff_t)};

  // std::align modifies `aligned` to point to the first aligned location
  std::align(alignment, bytes, aligned, padded_allocation_size);

  // Compute the offset between the original and aligned pointers
  std::ptrdiff_t const offset = static_cast<char*>(aligned) - original;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  *(static_cast<std::ptrdiff_t*>(aligned) - 1) = offset;

  return aligned;
}

void aligned_host_deallocate(void* ptr,
                             [[maybe_unused]] std::size_t bytes,
                             [[maybe_unused]] std::size_t alignment) noexcept
{
  assert(rmm::is_supported_alignment(alignment));

  if (ptr != nullptr) {
    // Get offset from the location immediately prior to the aligned pointer
    // NOLINTNEXTLINE
    std::ptrdiff_t const offset = *(reinterpret_cast<std::ptrdiff_t*>(ptr) - 1);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    void* const original = static_cast<char*>(ptr) - offset;

    ::operator delete(original);
  }
}

/**
 * @brief Stream-ordered pinned host memory pool backed by the CUDA native memory pool API.
 *
 * Uses `cudaMallocFromPoolAsync` / `cudaFreeAsync` with a pool created for
 * `cudaMemAllocationTypePinned + cudaMemLocationTypeHost`.  This provides two benefits over the
 * old `rmm::mr::pool_memory_resource<pinned_host_memory_resource>` approach:
 *
 *   1. **No sync on allocation.**  The CUDA runtime guarantees that recycled blocks have no
 *      outstanding GPU accesses, so `rmm_host_allocator::allocate()` can skip
 *      `stream.synchronize()`.
 *   2. **Stream-ordered free.**  Callers can destroy a `host_vector` immediately after enqueuing
 *      a device-to-host copy; the runtime defers the actual reclaim until the stream reaches the
 *      free point.
 *
 * The pool grows on demand.  `release_threshold` controls how many bytes the pool retains before
 * returning memory to the OS (mirrors the old max-pool-size semantics).
 */
class cuda_host_pinned_pool_memory_resource {
 public:
  explicit cuda_host_pinned_pool_memory_resource(std::size_t release_threshold)
  {
    cudaMemPoolProps props{};
    props.allocType     = cudaMemAllocationTypePinned;
    props.handleTypes   = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeHost;
    props.location.id   = 0;  // id is ignored for cudaMemLocationTypeHost
    CUDF_CUDA_TRY(cudaMemPoolCreate(&pool_, &props));

    // Keep up to release_threshold bytes in the pool before releasing to the OS.
    CUDF_CUDA_TRY(
      cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &release_threshold));

    // Unlike cudaMallocHost, host pools are not device-accessible by default.
    int current_device = 0;
    CUDF_CUDA_TRY(cudaGetDevice(&current_device));
    cudaMemAccessDesc access{};
    access.location.type = cudaMemLocationTypeDevice;
    access.location.id   = current_device;
    access.flags         = cudaMemAccessFlagsProtReadWrite;
    CUDF_CUDA_TRY(cudaMemPoolSetAccess(pool_, &access, 1));

    CUDF_LOG_INFO("CUDA host pinned pool created, release threshold = %zu bytes, device = %d",
                  release_threshold,
                  current_device);
  }

  // The pool handle is an opaque value, so copies are cheap.  Copies share the same underlying
  // pool; operator== distinguishes them.  We intentionally do NOT destroy the pool in the
  // destructor: cuda::mr::resource_ref stores a copy of the resource via type erasure, and
  // calling cudaMemPoolDestroy at process exit can race with CUDA teardown (same reasoning as
  // the raw pool_ pointer in the old pinned_pool_with_fallback_memory_resource).
  cuda_host_pinned_pool_memory_resource(cuda_host_pinned_pool_memory_resource const&) = default;
  cuda_host_pinned_pool_memory_resource& operator=(cuda_host_pinned_pool_memory_resource const&) =
    default;

  // clang-tidy will complain about these get_property friends because they are completely
  // unused at runtime and only exist for tag introspection by CCCL, so we ignore linting.
  // This masks a real issue if we ever want to compile with clang, though, which is that the
  // function will actually be compiled out by clang.  The same goes for the other get_property
  // definitions in this file.
  friend void get_property(cuda_host_pinned_pool_memory_resource const&,  // NOLINT
                           cuda::mr::host_accessible) noexcept
  {
  }

  friend void get_property(cuda_host_pinned_pool_memory_resource const&,  // NOLINT
                           cuda::mr::device_accessible) noexcept
  {
  }

  friend void get_property(cuda_host_pinned_pool_memory_resource const&,  // NOLINT
                           cudf::detail::stream_ordered_host_accessible_t) noexcept
  {
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    // cudaMallocFromPoolAsync guarantees at least 256-byte alignment; cudf never requests more.
    void* ptr      = nullptr;
    auto const err = cudaMallocFromPoolAsync(&ptr, bytes, pool_, stream.get());
    if (err != cudaSuccess) { throw std::bad_alloc(); }
    return ptr;
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  [[maybe_unused]] std::size_t bytes,
                  [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    [[maybe_unused]] auto err = cudaFreeAsync(ptr, stream.get());
  }

  bool operator==(cuda_host_pinned_pool_memory_resource const& other) const noexcept
  {
    return pool_ == other.pool_;
  }

  bool operator!=(cuda_host_pinned_pool_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  cudaMemPool_t pool_{};
};

static_assert(cuda::mr::resource_with<cuda_host_pinned_pool_memory_resource,
                                      cuda::mr::device_accessible,
                                      cuda::mr::host_accessible,
                                      cudf::detail::stream_ordered_host_accessible_t>,
              "CUDA host pinned pool mr must be host/device accessible and stream-ordered");

CUDF_EXPORT cuda_host_pinned_pool_memory_resource& make_default_pinned_mr(
  std::optional<size_t> config_size)
{
  static cuda_host_pinned_pool_memory_resource mr = [config_size]() {
    auto const initial_size = [&config_size]() -> size_t {
      if (auto const env_val = getenv("LIBCUDF_PINNED_POOL_SIZE"); env_val != nullptr) {
        return std::atol(env_val);
      }

      if (config_size.has_value()) { return *config_size; }

      auto const total = rmm::available_device_memory().second;
      // 0.5% of total device memory, capped at 64 MB
      return std::min(total / 200, size_t{64} * 1024 * 1024);
    }();

    auto const release_threshold = [&initial_size]() -> size_t {
      if (auto const env_val = getenv("LIBCUDF_PINNED_POOL_MAX_SIZE"); env_val != nullptr) {
        return std::atol(env_val);
      }
      return initial_size * 16;
    }();

    return cuda_host_pinned_pool_memory_resource{release_threshold};
  }();

  return mr;
}

CUDF_EXPORT std::mutex& host_mr_mutex()
{
  static std::mutex map_lock;
  return map_lock;
}

// Must be called with the host_mr_mutex mutex held
CUDF_EXPORT rmm::host_device_async_resource_ref& make_host_mr(
  std::optional<pinned_mr_options> const& opts, bool* did_configure = nullptr)
{
  static rmm::host_device_async_resource_ref* mr_ref = nullptr;
  bool configured                                    = false;
  if (mr_ref == nullptr) {
    configured = true;
    auto& pool = make_default_pinned_mr(opts ? opts->pool_size : std::nullopt);
    static rmm::host_device_async_resource_ref pool_ref{pool};
    mr_ref = &pool_ref;
  }

  if (did_configure != nullptr) { *did_configure = configured; }

  return *mr_ref;
}

// Must be called with the host_mr_mutex mutex held
CUDF_EXPORT rmm::host_device_async_resource_ref& host_mr()
{
  static rmm::host_device_async_resource_ref mr_ref = make_host_mr(std::nullopt);
  return mr_ref;
}

// Returns a typed ref that carries stream_ordered_host_accessible_t — used internally so that
// rmm_host_allocator can skip stream.synchronize() in allocate().
CUDF_EXPORT cudf::detail::stream_ordered_host_device_async_resource_ref& stream_ordered_host_mr()
{
  static cudf::detail::stream_ordered_host_device_async_resource_ref mr_ref =
    make_default_pinned_mr(std::nullopt);
  return mr_ref;
}

class new_delete_memory_resource {
 public:
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    try {
      return aligned_host_allocate(bytes, alignment);
    } catch (std::bad_alloc const& e) {
      CUDF_FAIL("Failed to allocate memory: " + std::string{e.what()}, rmm::out_of_memory);
    }
  }

  void* allocate([[maybe_unused]] cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    aligned_host_deallocate(ptr, bytes, alignment);
  }

  void deallocate([[maybe_unused]] cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate_sync(ptr, bytes, alignment);
  }

  bool operator==(new_delete_memory_resource const& other) const { return true; }

  bool operator!=(new_delete_memory_resource const& other) const { return !operator==(other); }

  // NOLINTBEGIN
  friend void get_property(new_delete_memory_resource const&, cuda::mr::host_accessible) noexcept {}
  // NOLINTEND
};

static_assert(cuda::mr::resource_with<new_delete_memory_resource, cuda::mr::host_accessible>,
              "Pageable pool mr must be accessible from the host");

}  // namespace

rmm::host_device_async_resource_ref set_pinned_memory_resource(
  rmm::host_device_async_resource_ref mr)
{
  std::scoped_lock lock{host_mr_mutex()};
  auto last_mr = host_mr();
  host_mr()    = mr;
  return last_mr;
}

rmm::host_device_async_resource_ref get_pinned_memory_resource()
{
  std::scoped_lock lock{host_mr_mutex()};
  return host_mr();
}

bool config_default_pinned_memory_resource(pinned_mr_options const& opts)
{
  std::scoped_lock lock{host_mr_mutex()};
  auto did_configure = false;
  make_host_mr(opts, &did_configure);
  return did_configure;
}

CUDF_EXPORT auto& kernel_pinned_copy_threshold()
{
  // use cudaMemcpyAsync for all pinned copies
  static std::atomic<size_t> threshold =
    cudf::detail::getenv_or("LIBCUDF_KERNEL_PINNED_COPY_THRESHOLD", 0);
  return threshold;
}

void set_kernel_pinned_copy_threshold(size_t threshold)
{
  kernel_pinned_copy_threshold() = threshold;
}

size_t get_kernel_pinned_copy_threshold() { return kernel_pinned_copy_threshold(); }

CUDF_EXPORT auto& allocate_host_as_pinned_threshold()
{
  // use pageable memory for all host allocations
  static std::atomic<size_t> threshold =
    cudf::detail::getenv_or("LIBCUDF_ALLOCATE_HOST_AS_PINNED_THRESHOLD", 0);
  return threshold;
}

void set_allocate_host_as_pinned_threshold(size_t threshold)
{
  allocate_host_as_pinned_threshold() = threshold;
}

size_t get_allocate_host_as_pinned_threshold() { return allocate_host_as_pinned_threshold(); }

namespace detail {

CUDF_EXPORT rmm::host_async_resource_ref get_pageable_memory_resource()
{
  static new_delete_memory_resource mr{};
  static rmm::host_async_resource_ref mr_ref{mr};
  return mr_ref;
}

CUDF_EXPORT stream_ordered_host_device_async_resource_ref
get_stream_ordered_pinned_memory_resource()
{
  return stream_ordered_host_mr();
}

}  // namespace detail

}  // namespace cudf
