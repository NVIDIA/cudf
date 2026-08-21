/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

enum class host_memory_kind : uint8_t { PINNED, PAGEABLE };

void cuda_memcpy_async_impl(
  void* dst, void const* src, size_t size, host_memory_kind kind, rmm::cuda_stream_view stream);

/**
 * @brief Wrapper around cudaMemcpyBatchAsync
 *
 * Uses `cudaMemcpyBatchAsync` on CUDA 13.0+ with `cudaMemcpySrcAccessOrderStream`, which defers
 * reading the source buffers until the stream reaches each copy. The source buffers must therefore
 * remain valid until the stream has executed the copies; for device memory this is naturally
 * satisfied, but for host memory the caller must ensure the source is not freed before the stream
 * is synchronized.
 *
 * A batch uses `cudaMemcpyFlagPreferOverlapWithCompute` when every copy is 128 KiB or less. If
 * any copy is larger, the batch uses `cudaMemcpyFlagDefault`.
 *
 * @param dsts Host pointer to a list of destination pointers.
 * @param srcs Host pointer to a list of source pointers.
 * @param sizes Host pointer to a list of sizes.
 * @param count Size of dsts, srcs, sizes arrays
 * @param stream CUDA stream on which copies are enqueued
 *
 * @note if \p stream is the default stream, this function will fallback to `cudaMemcpyAsync` for
 * each copy.
 */
[[nodiscard]] cudaError_t memcpy_batch_async(void* const* dsts,
                                             void const* const* srcs,
                                             std::size_t const* sizes,
                                             std::size_t count,
                                             rmm::cuda_stream_view stream);

/**
 * @brief Asynchronously copies a single buffer, wrapping `memcpy_batch_async`.
 *
 * Carries the same source-lifetime requirement as `memcpy_batch_async`: the source buffer must
 * remain valid until the stream has executed the copy.
 *
 * Prefer `cudf::detail::cuda_memcpy_async` for host/device copies involving typed spans.
 * Use this function for device-to-device copies or when a raw `void*` interface is required.
 * The copy direction is inferred from the pointer types (`cudaMemcpyDefault`).
 *
 * @param dst Destination memory address
 * @param src Source memory address
 * @param count Size in bytes to copy
 * @param stream CUDA stream on which the copy is enqueued
 * @return cudaError_t CUDA error code
 */
[[nodiscard]] cudaError_t memcpy_async(void* dst,
                                       void const* src,
                                       size_t count,
                                       rmm::cuda_stream_view stream);

/**
 * @brief Asynchronously copies data from host to device memory.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination device memory
 * @param src Source host memory
 * @param stream CUDA stream used for the copy
 */
template <typename T>
void cuda_memcpy_async(device_span<T> dst, host_span<T const> src, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(dst.size() == src.size(), "Mismatched sizes in cuda_memcpy_async");
  auto const is_pinned = src.is_device_accessible();
  cuda_memcpy_async_impl(dst.data(),
                         src.data(),
                         src.size_bytes(),
                         is_pinned ? host_memory_kind::PINNED : host_memory_kind::PAGEABLE,
                         stream);
}

/**
 * @brief Asynchronously copies data from device to host memory.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination host memory
 * @param src Source device memory
 * @param stream CUDA stream used for the copy
 */
template <typename T>
void cuda_memcpy_async(host_span<T> dst, device_span<T const> src, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(dst.size() == src.size(), "Mismatched sizes in cuda_memcpy_async");
  auto const is_pinned = dst.is_device_accessible();
  cuda_memcpy_async_impl(dst.data(),
                         src.data(),
                         src.size_bytes(),
                         is_pinned ? host_memory_kind::PINNED : host_memory_kind::PAGEABLE,
                         stream);
}

/**
 * @brief Synchronously copies data from host to device memory.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination device memory
 * @param src Source host memory
 * @param stream CUDA stream used for the copy
 */
template <typename T>
void cuda_memcpy(device_span<T> dst, host_span<T const> src, rmm::cuda_stream_view stream)
{
  cuda_memcpy_async(dst, src, stream);
  stream.synchronize();
}

/**
 * @brief Synchronously copies data from device to host memory.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination host memory
 * @param src Source device memory
 * @param stream CUDA stream used for the copy
 */
template <typename T>
void cuda_memcpy(host_span<T> dst, device_span<T const> src, rmm::cuda_stream_view stream)
{
  cuda_memcpy_async(dst, src, stream);
  stream.synchronize();
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
