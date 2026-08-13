/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_pool.hpp>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <vector>

namespace cudf::detail {

// TODO: what is a good number here. what's the penalty for making it larger?
// Dave Baranec rule of thumb was max_streams_needed * num_concurrent_threads,
// where num_concurrent_threads was estimated to be 4. so using 32 will allow
// for 8 streams per thread, which should be plenty (decoding will be up to 4
// kernels when delta_byte_array decoding is added). rmm::cuda_stream_pool
// defaults to 16.
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
 * @brief Implementation of `cuda_stream_pool` that wraps an `rmm::cuda_stram_pool`.
 */
class rmm_cuda_stream_pool : public cuda_stream_pool {
  rmm::cuda_stream_pool _pool;

 public:
  rmm_cuda_stream_pool() : _pool{STREAM_POOL_SIZE, rmm::cuda_stream::flags::non_blocking} {}
  rmm::cuda_stream_view get_stream() override { return _pool.get_stream(); }
  rmm::cuda_stream_view get_stream(stream_id_type stream_id) override
  {
    return _pool.get_stream(stream_id);
  }

  std::vector<rmm::cuda_stream_view> get_streams(std::size_t count) override
  {
    auto streams = std::vector<rmm::cuda_stream_view>();
    for (uint32_t i = 0; i < count; i++) {
      streams.emplace_back(_pool.get_stream());
    }
    return streams;
  }

  [[nodiscard]] std::size_t get_stream_pool_size() const override { return STREAM_POOL_SIZE; }
};

/**
 * @brief Implementation of `cuda_stream_pool` that always returns `cudf::get_default_stream()`
 */
class debug_cuda_stream_pool : public cuda_stream_pool {
 public:
  rmm::cuda_stream_view get_stream() override { return cudf::get_default_stream(); }
  rmm::cuda_stream_view get_stream(stream_id_type stream_id) override
  {
    return cudf::get_default_stream();
  }

  std::vector<rmm::cuda_stream_view> get_streams(std::size_t count) override
  {
    return std::vector<rmm::cuda_stream_view>(count, cudf::get_default_stream());
  }

  [[nodiscard]] std::size_t get_stream_pool_size() const override { return 1UL; }
};

cuda_stream_pool* create_global_cuda_stream_pool()
{
  if (getenv("LIBCUDF_USE_DEBUG_STREAM_POOL")) return new debug_cuda_stream_pool();
  return new rmm_cuda_stream_pool();
}

/**
 * @brief Returns a reference to the global stream pool for the current device.
 * @return `cuda_stream_pool` valid on the current device.
 */
cuda_stream_pool& global_cuda_stream_pool()
{
  // using bare pointers here to deliberately allow them to leak. otherwise we wind up with
  // seg faults trying to destroy stream objects after the context has shut down.
  static std::vector<cuda_stream_pool*> pools(get_num_cuda_devices());
  static std::mutex mutex;
  auto const device_id = get_current_cuda_device();

  std::lock_guard<std::mutex> const lock(mutex);
  if (pools[device_id.value()] == nullptr) {
    pools[device_id.value()] = create_global_cuda_stream_pool();
  }
  return *pools[device_id.value()];
}

std::vector<rmm::cuda_stream_view> fork_streams(rmm::cuda_stream_view stream, std::size_t count)
{
  auto const streams = global_cuda_stream_pool().get_streams(count);
  auto const event   = event_for_thread();
  CUDF_CUDA_TRY(cudaEventRecord(event, stream));
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(strm, event, 0));
  });
  return streams;
}

void join_streams(host_span<rmm::cuda_stream_view const> streams, rmm::cuda_stream_view stream)
{
  auto const event = event_for_thread();
  std::for_each(streams.begin(), streams.end(), [&](auto& strm) {
    CUDF_CUDA_TRY(cudaEventRecord(event, strm));
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream, event, 0));
  });
}

}  // namespace cudf::detail
