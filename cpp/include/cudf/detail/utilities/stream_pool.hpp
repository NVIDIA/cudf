/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/stream>

#include <cstddef>
#include <span>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Interface for a pool of CUDA streams.
 *
 * Implementations are not required to be thread safe. A pool is owned by a single thread at a time,
 * which is how `current_cuda_stream_pool()` hands them out, so an implementation may keep
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
   * @return Stream reference.
   */
  virtual cuda::stream_ref get_stream() = 0;

  /**
   * @brief Get a vector of `cuda::stream_ref` objects from the pool.
   *
   * The returned streams are distinct unless `count` is greater than the maximum number of streams
   * the pool provides, in which case streams are repeated.
   *
   * @param count The number of stream references to return.
   * @return Vector containing `count` stream references.
   */
  virtual std::vector<cuda::stream_ref> get_streams(std::size_t count) = 0;

 protected:
  cuda_stream_pool() = default;
};

/**
 * @brief Create a stream pool for a thread to use with one device.
 *
 * Overridden by the stream identification test utilities to substitute a pool that always returns
 * the default stream.
 *
 * @return An owning pointer to a new pool.
 */
cuda_stream_pool* create_cuda_stream_pool();

/**
 * @brief Get the stream pool the calling thread should use for the current device.
 *
 * Each thread currently has its own pool for each device it uses, so concurrent threads are handed
 * distinct streams. The maximum number of streams a pool provides can be configured with the
 * `LIBCUDF_STREAM_POOL_SIZE` environment variable.
 *
 * The returned streams stay valid for the lifetime of the process and may be used from any thread.
 * Once the thread that obtained them exits its pool is recycled, so another thread can be handed
 * the same streams; holding on to them past that point gives up the isolation the pool provides.
 *
 * @return Reference to the calling thread's stream pool for the current device.
 */
cuda_stream_pool& current_cuda_stream_pool();

/**
 * @brief Get the stream pool the calling thread should use for the current device.
 *
 * @deprecated Renamed to `current_cuda_stream_pool`, which does not imply a process-wide pool.
 *
 * @return Reference to the calling thread's stream pool for the current device.
 */
[[deprecated("Use current_cuda_stream_pool instead.")]]  //
inline cuda_stream_pool&
global_cuda_stream_pool()
{
  return current_cuda_stream_pool();
}

/**
 * @brief Acquire a vector of `cuda::stream_ref` objects and synchronize them to an event on
 * another stream.
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
 * @param count The number of `cuda::stream_ref` objects to return.
 * @return Vector containing `count` stream references.
 */
[[nodiscard]] std::vector<cuda::stream_ref> fork_streams(cuda::stream_ref stream,
                                                         std::size_t count);

/**
 * @brief Synchronize a stream to an event on each of a group of streams.
 *
 * @param streams Streams to wait on.
 * @param stream Joined stream that synchronizes with the waited-on streams.
 */
void join_streams(std::span<cuda::stream_ref const> streams, cuda::stream_ref stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
