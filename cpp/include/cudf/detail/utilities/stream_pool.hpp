/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail {

class cuda_stream_pool {
 public:
  // matching type used in rmm::cuda_stream_pool::get_stream(stream_id)
  using stream_id_type = std::size_t;

  virtual ~cuda_stream_pool()                          = default;
  cuda_stream_pool(cuda_stream_pool const&)            = delete;
  cuda_stream_pool(cuda_stream_pool&&)                 = delete;
  cuda_stream_pool& operator=(cuda_stream_pool const&) = delete;
  cuda_stream_pool& operator=(cuda_stream_pool&&)      = delete;

  /**
   * @brief Get a `cuda_stream_view` of a stream in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return Stream view.
   */
  virtual rmm::cuda_stream_view get_stream() = 0;

  /**
   * @brief Get a `cuda_stream_view` of the stream associated with `stream_id`.
   *
   * Equivalent values of `stream_id` return a `cuda_stream_view` to the same underlying stream.
   * This function is thread safe with respect to other calls to the same function.
   *
   * @param stream_id Unique identifier for the desired stream
   * @return Requested stream view.
   */
  virtual rmm::cuda_stream_view get_stream(stream_id_type stream_id) = 0;

  /**
   * @brief Get a set of `cuda_stream_view` objects from the pool.
   *
   * The returned streams are distinct unless `count` is greater than the value returned by
   * `get_stream_pool_size()`, in which case streams are repeated.
   *
   * Consecutive calls are served from different streams where the pool is large enough, so a
   * nested call generally does not return streams that its caller is already using. This is not
   * guaranteed: once the pool has reached its maximum size the assignment wraps around, so a
   * request for more than half the pool can overlap with the streams the caller holds.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @param count The number of stream views to return.
   * @return Vector containing `count` stream views.
   */
  virtual std::vector<rmm::cuda_stream_view> get_streams(std::size_t count) = 0;

  /**
   * @brief Get the maximum number of unique stream objects the pool can provide.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return the maximum number of stream objects in the pool
   */
  [[nodiscard]] virtual std::size_t get_stream_pool_size() const = 0;

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
 * streams. Pools are created empty and grow on demand up to a maximum that can be configured with
 * the `LIBCUDF_STREAM_POOL_SIZE` environment variable.
 *
 * The returned streams may be used from any thread, but must not be used after the thread that
 * obtained them has exited: a pool is recycled for reuse by another thread at that point.
 */
cuda_stream_pool& global_cuda_stream_pool();

/**
 * @brief Acquire a set of `cuda_stream_view` objects and synchronize them to an event on another
 * stream.
 *
 * By default the calling thread's stream pool is used to obtain the streams, so streams are not
 * shared with concurrently forking threads. The only other implementation at present is a debugging
 * version that always returns the stream returned by `cudf::get_default_stream()`. To use this
 * debugging version, set the environment variable `LIBCUDF_USE_DEBUG_STREAM_POOL`.
 *
 * The returned streams must not be used after the calling thread has exited.
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
[[nodiscard]] std::vector<rmm::cuda_stream_view> fork_streams(rmm::cuda_stream_view stream,
                                                              std::size_t count);

/**
 * @brief Synchronize a stream to an event on a set of streams.
 *
 * @param streams Streams to wait on.
 * @param stream Joined stream that synchronizes with the waited-on streams.
 */
void join_streams(host_span<rmm::cuda_stream_view const> streams, rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
