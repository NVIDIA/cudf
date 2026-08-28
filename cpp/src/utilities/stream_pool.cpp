/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda/devices>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace cudf::detail {

// Maximum number of streams a single thread's pool will create, for a single device. Sized to cover
// the largest request libcudf makes with a fixed bound, which is one stream per distinct parquet
// decode kernel (see `decode_kernel_mask`). Host (de)compression, the JSON reader and others
// instead scale their request with the host worker count, so raising `LIBCUDF_NUM_HOST_WORKERS`
// past this leaves those requests sharing streams.
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
 * @brief Returns the configured maximum number of streams a single pool will hold.
 *
 * The environment is read once, so that pools created on worker threads do not race a concurrent
 * `setenv`.
 */
std::size_t configured_max_pool_size()
{
  static std::size_t const size =
    std::max<std::size_t>(1, getenv_or("LIBCUDF_STREAM_POOL_SIZE", STREAM_POOL_SIZE));
  return size;
}

}  // namespace

/**
 * @brief Implementation of `cuda_stream_pool` that creates streams on demand.
 *
 * Instances are owned by a single thread at a time, so no synchronization is needed. The pool
 * never shrinks; it grows to the largest number of streams requested so far, up to `_max_size`.
 */
class growing_cuda_stream_pool : public cuda_stream_pool {
  std::vector<cuda::stream> _streams;
  std::size_t _next_stream{0};
  std::size_t const _max_size{configured_max_pool_size()};

  /**
   * @brief Creates streams until the pool holds `size` of them, or has reached `_max_size`.
   */
  void grow_to(std::size_t size)
  {
    auto const device = cuda::device_ref{get_current_cuda_device().value()};
    auto const target = std::min(size, _max_size);
    while (_streams.size() < target) {
      // `cuda::stream` creates non-blocking streams.
      _streams.emplace_back(device);
    }
  }

 public:
  cuda::stream_ref get_stream() override { return get_streams(1).front(); }

  std::vector<cuda::stream_ref> get_streams(std::size_t count) override
  {
    // Growing to twice the requested count leaves room for consecutive requests to return
    // different streams.
    grow_to(2 * count);
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

  std::vector<cuda::stream_ref> get_streams(std::size_t count) override
  {
    return std::vector<cuda::stream_ref>(count, cudf::get_default_stream());
  }
};

cuda_stream_pool* create_cuda_stream_pool()
{
  if (getenv("LIBCUDF_USE_DEBUG_STREAM_POOL")) return new debug_cuda_stream_pool();
  return new growing_cuda_stream_pool();
}

namespace {

/**
 * @brief Free lists of pools and events that are not currently owned by any thread, one list of
 * each per device.
 *
 * They are recycled instead of destroyed so that applications which create and destroy many
 * threads do not accumulate streams and events. The registry is intentionally leaked so that its
 * lifetime covers the `thread_local` destructors that push into it.
 *
 * Neither pools nor events can move between devices; both are bound to the device that was current
 * when they were created, so each device has its own lists.
 */
class stream_pool_registry {
  std::mutex _mutex;
  std::vector<std::vector<cuda_stream_pool*>> _free_pools;
  std::vector<std::vector<cuda_event*>> _free_events;

 public:
  stream_pool_registry() : _free_pools(get_num_cuda_devices()), _free_events(get_num_cuda_devices())
  {
  }

  /**
   * @brief Takes a pool for `device_id`, reusing a retired one if there is one available.
   */
  cuda_stream_pool* acquire_pool(rmm::cuda_device_id device_id)
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
    return create_cuda_stream_pool();
  }

  /**
   * @brief Takes an event for `device_id`, reusing a retired one if there is one available.
   *
   * Events are never destroyed: the program may crash if one is destroyed after the application
   * calls `cudaDeviceReset()`.
   */
  cuda_event* acquire_event(rmm::cuda_device_id device_id)
  {
    {
      std::lock_guard<std::mutex> const lock(_mutex);
      auto& free_events = _free_events[device_id.value()];
      if (not free_events.empty()) {
        auto* event = free_events.back();
        free_events.pop_back();
        return event;
      }
    }
    return new cuda_event();
  }

  /**
   * @brief Returns a pool so that another thread can reuse it.
   *
   * Called from a `thread_local` destructor, so it must not call into CUDA; destroying streams
   * here would race with driver teardown when the main thread exits.
   */
  void release_pool(rmm::cuda_device_id device_id, cuda_stream_pool* pool) noexcept
  {
    std::lock_guard<std::mutex> const lock(_mutex);
    _free_pools[device_id.value()].push_back(pool);
  }

  /**
   * @brief Returns an event so that another thread can reuse it.
   *
   * Called from a `thread_local` destructor, so it must not call into CUDA, for the same reason as
   * `release_pool`.
   */
  void release_event(rmm::cuda_device_id device_id, cuda_event* event) noexcept
  {
    std::lock_guard<std::mutex> const lock(_mutex);
    _free_events[device_id.value()].push_back(event);
  }
};

stream_pool_registry& pool_registry()
{
  static auto* registry = new stream_pool_registry();
  return *registry;
}

/**
 * @brief Owns the calling thread's pool and event for each device, and retires them when the
 * thread exits.
 */
class thread_stream_resources {
  std::vector<cuda_stream_pool*> _pools;
  std::vector<cuda_event*> _events;

 public:
  thread_stream_resources()
    : _pools(get_num_cuda_devices(), nullptr), _events(get_num_cuda_devices(), nullptr)
  {
  }

  ~thread_stream_resources()
  {
    for (rmm::cuda_device_id::value_type device = 0; std::cmp_less(device, _pools.size());
         device++) {
      auto const device_id = rmm::cuda_device_id{device};
      if (_pools[device] != nullptr) { pool_registry().release_pool(device_id, _pools[device]); }
      if (_events[device] != nullptr) { pool_registry().release_event(device_id, _events[device]); }
    }
  }

  thread_stream_resources(thread_stream_resources const&)            = delete;
  thread_stream_resources& operator=(thread_stream_resources const&) = delete;
  thread_stream_resources(thread_stream_resources&&)                 = delete;
  thread_stream_resources& operator=(thread_stream_resources&&)      = delete;

  cuda_stream_pool& pool_for(rmm::cuda_device_id device_id)
  {
    auto*& pool = _pools[device_id.value()];
    if (pool == nullptr) { pool = pool_registry().acquire_pool(device_id); }
    return *pool;
  }

  cuda_event& event_for(rmm::cuda_device_id device_id)
  {
    auto*& event = _events[device_id.value()];
    if (event == nullptr) { event = pool_registry().acquire_event(device_id); }
    return *event;
  }
};

thread_stream_resources& current_thread_resources()
{
  thread_local thread_stream_resources resources;
  return resources;
}

/**
 * @brief Returns a cudaEvent_t the calling thread can use on the current device.
 *
 * The event is reused by every fork and join the thread performs on that device, and is recycled
 * for another thread when this one exits.
 */
cudaEvent_t event_for_thread()
{
  return current_thread_resources().event_for(get_current_cuda_device());
}

}  // namespace

/**
 * @brief Returns a reference to the calling thread's stream pool for the current device.
 * @return `cuda_stream_pool` owned by the current thread and valid on the current device.
 */
cuda_stream_pool& current_cuda_stream_pool()
{
  return current_thread_resources().pool_for(get_current_cuda_device());
}

std::vector<cuda::stream_ref> fork_streams(cuda::stream_ref stream, std::size_t count)
{
  auto const streams = current_cuda_stream_pool().get_streams(count);
  auto const event   = event_for_thread();
  CUDF_CUDA_TRY(cudaEventRecord(event, stream.get()));
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(strm.get(), event, 0));
  });
  return streams;
}

void join_streams(std::span<cuda::stream_ref const> streams, cuda::stream_ref stream)
{
  auto const event = event_for_thread();
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaEventRecord(event, strm.get()));
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.get(), event, 0));
  });
}

}  // namespace cudf::detail
