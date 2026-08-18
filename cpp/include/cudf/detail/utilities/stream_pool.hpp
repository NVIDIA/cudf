/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream_ref>

#include <cstddef>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Interface for a pool of CUDA streams.
 *
 * Implementations are not required to be thread safe. A pool is owned by a single thread at a time,
 * which is how `thread_cuda_stream_pool()` hands them out, so an implementation may keep
 * unsynchronized state. Sharing one pool between threads requires external synchronization.
 */
class cuda_stream_pool {
 public:
  virtual ~cuda_stream_pool()                          = default;
  cuda_stream_pool(cuda_stream_pool const&)            = delete;
  cuda_stream_pool(cuda_stream_pool&&)                 = delete;
  cuda_stream_pool& operator=(cuda_stream_pool const&) = delete;
  cuda_stream_pool& operator=(cuda_stream_pool&&)      = delete;

  /**
   * @brief Get a single stream from the pool.
   *
   * @note Use `get_streams` to obtain multiple streams. Repeated single-stream requests are not
   * guaranteed to return different streams.
   *
   * @return Stream view.
   */
  virtual cuda::stream_ref get_stream() = 0;

  /**
   * @brief Get a set of `cuda_stream_view` objects from the pool.
   *
   * The returned streams are distinct unless `count` is greater than the maximum number of streams
   * the pool provides, in which case streams are repeated.
   *
   * Consecutive calls are served from different streams where the pool is large enough, so a
   * nested call generally does not return streams that its caller is already using. This is not
   * guaranteed: once the pool has reached its maximum size the assignment wraps around, so a
   * request for more than half the pool can overlap with the streams the caller holds.
   *
   * @param count The number of stream views to return.
   * @return Vector containing `count` stream views.
   */
  virtual std::vector<cuda::stream_ref> get_streams(std::size_t count) = 0;

 protected:
  cuda_stream_pool() = default;
};

/**
 * @brief Initialize global stream pool.
 */
cuda_stream_pool* create_global_cuda_stream_pool();

/**
 * @brief Get the calling thread's stream pool for the current device.
 *
 * Each thread has its own pool for each device it uses, so concurrent threads are handed distinct
 * streams. The maximum number of streams a pool provides can be configured with the
 * `LIBCUDF_STREAM_POOL_SIZE` environment variable.
 *
 * The returned streams stay valid for the lifetime of the process and may be used from any thread.
 * Once the thread that obtained them exits its pool is recycled, so another thread can be handed
 * the same streams; holding on to them past that point gives up the isolation the pool provides.
 */
cuda_stream_pool& thread_cuda_stream_pool();

/**
 * @brief Get the calling thread's stream pool for the current device.
 *
 * @deprecated Renamed to `thread_cuda_stream_pool` now that the pool is per thread and per device.
 */
[[deprecated("Use thread_cuda_stream_pool instead.")]]  //
inline cuda_stream_pool&
global_cuda_stream_pool()
{
  return thread_cuda_stream_pool();
}

/**
 * @brief Acquire a set of `cuda_stream_view` objects and synchronize them to an event on another
 * stream.
 *
 * By default the calling thread's stream pool is used to obtain the streams, so streams are not
 * shared with concurrently forking threads. The only other implementation at present is a debugging
 * version that always returns the stream returned by `cudf::get_default_stream()`. To use this
 * debugging version, set the environment variable `LIBCUDF_USE_DEBUG_STREAM_POOL`.
 *
 * The returned streams stay valid after the calling thread exits, but its pool is recycled at that
 * point, so they may then be handed to another thread as well.
 *
 * Example usage:
 * @code{.cpp}
 * auto stream = cudf::get_default_stream();
 * auto const num_streams = 2;
 * // do work on stream
 * // allocate streams and wait for an event on stream before executing on any of streams
 * auto streams = cudf::detail::fork_stream(stream, num_streams);
 * // do work on streams[0] and streams[1]
 * // wait for event on streams before continuing to do work on stream
 * cudf::detail::join_streams(streams, stream);
 * @endcode
 *
 * @param stream Stream that the returned streams will wait on.
 * @param count The number of `cuda_stream_view` objects to return.
 * @return Vector containing `count` stream views.
 */
[[nodiscard]] std::vector<cuda::stream_ref> fork_streams(cuda::stream_ref stream,
                                                         std::size_t count);

/**
 * @brief Synchronize a stream to an event on a set of streams.
 *
 * @param streams Streams to wait on.
 * @param stream Joined stream that synchronizes with the waited-on streams.
 */
void join_streams(host_span<cuda::stream_ref const> streams, cuda::stream_ref stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
