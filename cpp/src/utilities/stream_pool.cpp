/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

namespace cudf::detail {

// Maximum number of streams a single thread's pool will create, for a single device. Sized to cover
// the largest number of streams requested by a single `fork_streams` call in libcudf, which is the
// number of distinct parquet decode kernels (see `decode_kernel_mask`).
//
// This is a per-thread bound, not a process-wide one, so the streams an application holds scale
// with the number of threads that call into libcudf. Pools only grow on demand and are recycled
// when a thread exits, so the steady-state total is bounded by the peak number of concurrent
// threads rather than by the number of threads created.
std::size_t constexpr STREAM_POOL_SIZE = 32;

// FIXME: "borrowed" from rmm...remove when this stream pool is moved there
#ifdef NDEBUG
#define CUDF_ASSERT_CUDA_SUCCESS(_call) \
  do {                                  \
    (_call);                            \
  } while (0);
#else
#define CUDF_ASSERT_CUDA_SUCCESS(_call)                                         \
  do {                                                                          \
    cudaError_t const status__ = (_call);                                       \
    if (status__ != cudaSuccess) {                                              \
      std::cerr << "CUDA Error detected. " << cudaGetErrorName(status__) << " " \
                << cudaGetErrorString(status__) << std::endl;                   \
    }                                                                           \
    /* NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay) */   \
    assert(status__ == cudaSuccess);                                            \
  } while (0)
#endif

/**
 * @brief RAII struct to wrap a cuda event and ensure its proper destruction.
 */
struct cuda_event {
  cuda_event() { CUDF_CUDA_TRY(cudaEventCreateWithFlags(&e_, cudaEventDisableTiming)); }
  virtual ~cuda_event() { CUDF_ASSERT_CUDA_SUCCESS(cudaEventDestroy(e_)); }

  // Moveable but not copyable.
  cuda_event(cuda_event const&)            = delete;
  cuda_event& operator=(cuda_event const&) = delete;

  cuda_event(cuda_event&&)            = default;
  cuda_event& operator=(cuda_event&&) = default;

  operator cudaEvent_t() { return e_; }

 private:
  cudaEvent_t e_{};
};

namespace {

// FIXME: these will be available in rmm soon
inline int get_num_cuda_devices()
{
  rmm::cuda_device_id::value_type num_dev{};
  CUDF_CUDA_TRY(cudaGetDeviceCount(&num_dev));
  return num_dev;
}

rmm::cuda_device_id get_current_cuda_device()
{
  int device_id = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device_id));
  return rmm::cuda_device_id{device_id};
}

/**
 * @brief Returns the maximum number of streams a single thread's pool will hold.
 */
std::size_t stream_pool_size()
{
  static std::size_t const size =
    std::max<std::size_t>(1, getenv_or("LIBCUDF_STREAM_POOL_SIZE", STREAM_POOL_SIZE));
  return size;
}

/**
 * @brief Returns a cudaEvent_t for the current thread.
 *
 * The returned event is valid for the current device.
 *
 * @return A cudaEvent_t unique to the current thread and valid on the current device.
 */
cudaEvent_t event_for_thread()
{
  // The program may crash if this function is called from the main thread and user application
  // subsequently calls cudaDeviceReset().
  // As a workaround, here we intentionally disable RAII and leak cudaEvent_t.
  thread_local static std::vector<cuda_event*> thread_events(get_num_cuda_devices());
  auto const device_id = get_current_cuda_device();
  if (not thread_events[device_id.value()]) { thread_events[device_id.value()] = new cuda_event(); }
  return *thread_events[device_id.value()];
}

}  // namespace

/**
 * @brief Implementation of `cuda_stream_pool` that creates streams on demand.
 *
 * Instances are owned by a single thread at a time, so no synchronization is needed. The pool
 * never shrinks; it grows to the largest number of streams requested so far, up to
 * `stream_pool_size()`.
 */
class growing_cuda_stream_pool : public cuda_stream_pool {
  std::vector<cuda::stream> _streams;
  std::size_t _next_stream{0};

  /**
   * @brief Creates streams until the pool can serve `count` streams with room to spare.
   *
   * Twice the requested count is created so that consecutive requests are served from different
   * streams. Nested requests, such as decompression forking streams while its caller is using
   * forked streams of its own, then avoid colliding with the streams they are nested inside,
   * except where the rotation wraps around a pool that has reached `stream_pool_size()`.
   */
  void grow_to(std::size_t count)
  {
    // A pool is only ever used with the device it was created on current, so the streams it creates
    // belong to that device. `cuda::stream` is always non-blocking.
    auto const device = cuda::device_ref{get_current_cuda_device().value()};
    auto const target = std::min(2 * count, stream_pool_size());
    while (_streams.size() < target) {
      _streams.emplace_back(device);
    }
  }

 public:
  cuda::stream_ref get_stream() override { return get_streams(1).front(); }

  cuda::stream_ref get_stream(stream_id_type stream_id) override
  {
    // The id maps to the same stream on every call: growing for `stream_id` leaves the pool either
    // larger than `stream_id` or at exactly `stream_pool_size()`, so the modulus below is fixed.
    grow_to(stream_id + 1);
    return _streams[stream_id % _streams.size()];
  }

  std::vector<cuda::stream_ref> get_streams(std::size_t count) override
  {
    grow_to(count);
    auto const first = std::exchange(_next_stream, _next_stream + count);
    auto streams     = std::vector<cuda::stream_ref>();
    streams.reserve(count);
    for (std::size_t i = 0; i < count; i++) {
      streams.emplace_back(_streams[(first + i) % _streams.size()]);
    }
    return streams;
  }
};

/**
 * @brief Implementation of `cuda_stream_pool` that always returns `cudf::get_default_stream()`
 */
class debug_cuda_stream_pool : public cuda_stream_pool {
 public:
  cuda::stream_ref get_stream() override { return cudf::get_default_stream(); }
  cuda::stream_ref get_stream(stream_id_type stream_id) override
  {
    return cudf::get_default_stream();
  }

  std::vector<cuda::stream_ref> get_streams(std::size_t count) override
  {
    return std::vector<cuda::stream_ref>(count, cudf::get_default_stream());
  }
};

cuda_stream_pool* create_global_cuda_stream_pool()
{
  if (getenv("LIBCUDF_USE_DEBUG_STREAM_POOL")) return new debug_cuda_stream_pool();
  return new growing_cuda_stream_pool();
}

namespace {

/**
 * @brief Free lists of pools that are not currently owned by any thread, one list per device.
 *
 * Pools are recycled instead of destroyed so that applications which create and destroy many
 * threads do not accumulate streams. The registry is intentionally leaked so that its lifetime
 * covers the `thread_local` destructors that push pools back into it.
 *
 * Pools cannot move between devices; a stream is bound to the device that was current when the
 * stream was created, so each device has its own list.
 */
class stream_pool_registry {
  std::mutex _mutex;
  std::vector<std::vector<cuda_stream_pool*>> _free_pools;

 public:
  stream_pool_registry() : _free_pools(get_num_cuda_devices()) {}

  /**
   * @brief Takes a pool for `device_id`, reusing a retired one if there is one available.
   */
  cuda_stream_pool* acquire(rmm::cuda_device_id device_id)
  {
    {
      std::lock_guard<std::mutex> const lock(_mutex);
      auto& free_pools = _free_pools[device_id.value()];
      if (not free_pools.empty()) {
        auto* pool = free_pools.back();
        free_pools.pop_back();
        return pool;
      }
    }
    return create_global_cuda_stream_pool();
  }

  /**
   * @brief Returns a pool so that another thread can reuse it.
   *
   * Called from a `thread_local` destructor, so it must not call into CUDA; destroying streams
   * here would race with driver teardown when the main thread exits.
   */
  void release(rmm::cuda_device_id device_id, cuda_stream_pool* pool) noexcept
  {
    std::lock_guard<std::mutex> const lock(_mutex);
    _free_pools[device_id.value()].push_back(pool);
  }
};

stream_pool_registry& pool_registry()
{
  static auto* registry = new stream_pool_registry();
  return *registry;
}

/**
 * @brief Owns the calling thread's pool for each device, and retires them when the thread exits.
 */
class thread_stream_pools {
  std::vector<cuda_stream_pool*> _pools;

 public:
  thread_stream_pools() : _pools(get_num_cuda_devices(), nullptr) {}

  ~thread_stream_pools()
  {
    for (rmm::cuda_device_id::value_type device = 0; std::cmp_less(device, _pools.size());
         device++) {
      if (_pools[device] != nullptr) {
        pool_registry().release(rmm::cuda_device_id{device}, _pools[device]);
      }
    }
  }

  thread_stream_pools(thread_stream_pools const&)            = delete;
  thread_stream_pools& operator=(thread_stream_pools const&) = delete;
  thread_stream_pools(thread_stream_pools&&)                 = delete;
  thread_stream_pools& operator=(thread_stream_pools&&)      = delete;

  cuda_stream_pool& pool_for(rmm::cuda_device_id device_id)
  {
    auto*& pool = _pools[device_id.value()];
    if (pool == nullptr) { pool = pool_registry().acquire(device_id); }
    return *pool;
  }
};

}  // namespace

/**
 * @brief Returns a reference to the calling thread's stream pool for the current device.
 * @return `cuda_stream_pool` owned by the current thread and valid on the current device.
 */
cuda_stream_pool& thread_cuda_stream_pool()
{
  thread_local thread_stream_pools pools;
  return pools.pool_for(get_current_cuda_device());
}

std::vector<cuda::stream_ref> fork_streams(cuda::stream_ref stream, std::size_t count)
{
  auto const streams = thread_cuda_stream_pool().get_streams(count);
  auto const event   = event_for_thread();
  CUDF_CUDA_TRY(cudaEventRecord(event, stream.get()));
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(strm.get(), event, 0));
  });
  return streams;
}

void join_streams(host_span<cuda::stream_ref const> streams, cuda::stream_ref stream)
{
  auto const event = event_for_thread();
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaEventRecord(event, strm.get()));
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.get(), event, 0));
  });
}

}  // namespace cudf::detail
