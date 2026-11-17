/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @brief Convenience factories for creating device buffers from host spans
 * @file buffer_factories.hpp
 */

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/buffer>
#include <cuda/stream>

#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

namespace cudf::detail {

/**
 * @brief Asynchronously construct a `device_buffer` and set all elements to zero.
 *
 * @note This function does not synchronize `stream`.
 *
 * @tparam T The type of the data to copy
 * @param size The number of elements in the created buffer
 * @param stream The stream on which to allocate memory and perform the memset
 * @param mr The memory resource to use for allocating the returned device_buffer
 * @return A device_buffer containing zeros
 */
template <typename T>
cuda::device_buffer<T> make_zeroed_device_buffer_async(std::size_t size,
                                                       cuda::stream_ref stream,
                                                       cudf::memory_resources mr)
{
  cuda::device_buffer<T> ret(stream, mr.get_output_mr(), size, cuda::no_init);
  if (size != 0) { CUDF_CUDA_TRY(cudaMemsetAsync(ret.data(), 0, size * sizeof(T), stream.get())); }
  return ret;
}

/**
 * @brief Synchronously construct a `device_buffer` and set all elements to zero.
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam T The type of the data to copy
 * @param size The number of elements in the created buffer
 * @param stream The stream on which to allocate memory and perform the memset
 * @param mr The memory resource to use for allocating the returned device_buffer
 * @return A device_buffer containing zeros
 */
template <typename T>
cuda::device_buffer<T> make_zeroed_device_buffer(std::size_t size,
                                                 cuda::stream_ref stream,
                                                 cudf::memory_resources mr)
{
  auto ret = make_zeroed_device_buffer_async<T>(size, stream, mr);
  cudf::detail::sync_stream(stream);
  return ret;
}

/**
 * @brief Asynchronously construct a `device_buffer` containing a deep copy of data from a
 * `host_span`
 *
 * @note This function does not synchronize `stream`.
 *
 * @tparam T The type of the data to copy (may be const-qualified)
 * @param source_data The host_span of data to deep copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_buffer
 * @return A device_buffer containing the copied data
 */
template <typename T>
cuda::device_buffer<std::remove_cv_t<T>> make_device_buffer_async(host_span<T> source_data,
                                                                  cuda::stream_ref stream,
                                                                  cudf::memory_resources mr)
{
  using value_type = std::remove_cv_t<T>;
  cuda::device_buffer<value_type> ret(
    stream, mr.get_output_mr(), source_data.size(), cuda::no_init);
  cuda_memcpy_async<value_type>(ret, host_span<value_type const>{source_data}, stream);
  return ret;
}

/**
 * @brief Asynchronously construct a `device_buffer` containing a deep copy of data from a host
 * container
 *
 * @note This function does not synchronize `stream`.
 *
 * @tparam Container The type of the container to copy from
 * @param c The input host container from which to copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_buffer
 * @return A device_buffer containing the copied data
 */
template <typename Container>
cuda::device_buffer<typename Container::value_type> make_device_buffer_async(
  Container const& c, cuda::stream_ref stream, cudf::memory_resources mr)
  requires(std::is_convertible_v<Container, host_span<typename Container::value_type const>>)
{
  return make_device_buffer_async(host_span<typename Container::value_type const>{c}, stream, mr);
}

/**
 * @brief Asynchronously construct a `device_buffer` from a `std::vector`
 *
 * @note This function does not synchronize `stream`.
 *
 * @tparam T The type of the data to copy
 * @tparam Allocator The allocator type of the std::vector
 * @param source_data The std::vector of data to deep copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_buffer
 * @return A device_buffer containing the copied data
 */
template <typename T, typename Allocator>
cuda::device_buffer<T> make_device_buffer_async(std::vector<T, Allocator> const& source_data,
                                                cuda::stream_ref stream,
                                                cudf::memory_resources mr)
{
  return make_device_buffer_async(host_span<T const>{source_data}, stream, mr);
}

/**
 * @brief Asynchronously construct a `device_buffer` containing a deep copy of data from a
 * `std::span`
 *
 * @note This function does not synchronize `stream`.
 *
 * @tparam T The type of the data to copy (may be const-qualified)
 * @tparam Extent The extent of the std::span
 * @param source_data The std::span of data to deep copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_buffer
 * @return A device_buffer containing the copied data
 */
template <typename T, std::size_t Extent>
cuda::device_buffer<std::remove_cv_t<T>> make_device_buffer_async(std::span<T, Extent> source_data,
                                                                  cuda::stream_ref stream,
                                                                  cudf::memory_resources mr)
{
  return make_device_buffer_async(host_span{source_data.data(), source_data.size()}, stream, mr);
}

/**
 * @brief Synchronously construct a `device_buffer` containing a deep copy of data from a
 * `host_span`
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam T The type of the data to copy
 * @param source_data The host_span of data to deep copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_buffer
 * @return A device_buffer containing the copied data
 */
template <typename T>
cuda::device_buffer<T> make_device_buffer(host_span<T const> source_data,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  auto ret = make_device_buffer_async(source_data, stream, mr);
  cudf::detail::sync_stream(stream);
  return ret;
}

/**
 * @brief Synchronously construct a `device_buffer` containing a deep copy of data from a host
 * container
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam Container The type of the container to copy from
 * @param c The input host container from which to copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_buffer
 * @return A device_buffer containing the copied data
 */
template <typename Container>
cuda::device_buffer<typename Container::value_type> make_device_buffer(Container const& c,
                                                                       cuda::stream_ref stream,
                                                                       cudf::memory_resources mr)
  requires(std::is_convertible_v<Container, host_span<typename Container::value_type const>>)
{
  return make_device_buffer(host_span<typename Container::value_type const>{c}, stream, mr);
}

/**
 * @brief Synchronously construct a `device_buffer` from a `std::vector`
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam T The type of the data to copy
 * @tparam Allocator The allocator type of the std::vector
 * @param source_data The std::vector of data to deep copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_buffer
 * @return A device_buffer containing the copied data
 */
template <typename T, typename Allocator>
cuda::device_buffer<T> make_device_buffer(std::vector<T, Allocator> const& source_data,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  return make_device_buffer(host_span<T const>{source_data}, stream, mr);
}

/**
 * @brief Synchronously construct a `device_buffer` containing a deep copy of data from a
 * `std::span`
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam T The type of the data to copy (may be const-qualified)
 * @tparam Extent The extent of the std::span
 * @param source_data The std::span of data to deep copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_buffer
 * @return A device_buffer containing the copied data
 */
template <typename T, std::size_t Extent>
cuda::device_buffer<std::remove_cv_t<T>> make_device_buffer(std::span<T, Extent> source_data,
                                                            cuda::stream_ref stream,
                                                            cudf::memory_resources mr)
{
  return make_device_buffer(host_span{source_data.data(), source_data.size()}, stream, mr);
}

}  // namespace cudf::detail
