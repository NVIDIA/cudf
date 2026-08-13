/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

#include <algorithm>
#include <ranges>
#include <vector>

namespace cudf::detail {

namespace {

// Simple kernel to copy between device buffers
CUDF_KERNEL void copy_kernel(char const* __restrict__ src, char* __restrict__ dst, size_t n)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx < n) { dst[idx] = src[idx]; }
}

void copy_pinned(void* dst, void const* src, std::size_t size, rmm::cuda_stream_view stream)
{
  if (size == 0) return;

  if (size < get_kernel_pinned_copy_threshold()) {
    int const block_size = 256;
    auto const grid_size = cudf::util::div_rounding_up_safe<size_t>(size, block_size);
    // We are explicitly launching the kernel here instead of calling a thrust function because the
    // thrust function can potentially call cudaMemcpyAsync instead of using a kernel
    copy_kernel<<<grid_size, block_size, 0, stream.value()>>>(
      static_cast<char const*>(src), static_cast<char*>(dst), size);
  } else {
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(dst, src, size, stream));
  }
}

void copy_pageable(void* dst, void const* src, std::size_t size, rmm::cuda_stream_view stream)
{
  if (size == 0) return;

  CUDF_CUDA_TRY(cudf::detail::memcpy_async(dst, src, size, stream));
}

};  // namespace

cudaError_t memcpy_batch_async(void* const* dsts,
                               void const* const* srcs,
                               std::size_t const* sizes,
                               std::size_t count,
                               rmm::cuda_stream_view stream)
{
// Uses cudaMemcpyBatchAsync for CUDA 13.0+ to avoid driver-side locking overhead.
// cudaMemcpyBatchAsync does not support the default stream.
#if CUDART_VERSION >= 13000
  if (!stream.is_default()) {
    // Filter out invalid copies (nullptr dst/src or size==0);
    // cudaMemcpyBatchAsync does not support these inputs
    auto is_invalid = [&](auto i) {
      return dsts[i] == nullptr || srcs[i] == nullptr || sizes[i] == 0;
    };
    std::vector<void*> valid_dsts;
    std::vector<void const*> valid_srcs;
    std::vector<std::size_t> valid_sizes;

    if (std::ranges::any_of(std::ranges::views::iota(std::size_t{0}, count), is_invalid)) {
      valid_dsts.reserve(count);
      valid_srcs.reserve(count);
      valid_sizes.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        if (dsts[i] != nullptr && srcs[i] != nullptr && sizes[i] != 0) {
          valid_dsts.push_back(dsts[i]);
          valid_srcs.push_back(srcs[i]);
          valid_sizes.push_back(sizes[i]);
        }
      }
      if (valid_dsts.empty()) { return cudaSuccess; }
      dsts  = valid_dsts.data();
      srcs  = valid_srcs.data();
      sizes = valid_sizes.data();
      count = valid_dsts.size();
    }

    cudaMemcpyAttributes attrs = {.srcAccessOrder = cudaMemcpySrcAccessOrderStream,
                                  .flags          = cudaMemcpyFlagPreferOverlapWithCompute};
    std::size_t attrs_idxs     = 0;
    return cudaMemcpyBatchAsync(dsts, srcs, sizes, count, &attrs, &attrs_idxs, 1, stream.value());
  }
#endif  // CUDART_VERSION >= 13000
  for (std::size_t i = 0; i < count; ++i) {
    cudaError_t status =
      cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream.value());
    if (status != cudaSuccess) { return status; }
  }
  return cudaSuccess;
}

cudaError_t memcpy_async(void* dst, void const* src, size_t count, rmm::cuda_stream_view stream)
{
  if (count == 0) { return cudaSuccess; }

  // Use batch API with size 1 to prefer cudaMemcpyBatchAsync over
  // cudaMemcpyAsync. The batched API is more efficient.
  return memcpy_batch_async(&dst, &src, &count, 1, stream);
}

void cuda_memcpy_async_impl(
  void* dst, void const* src, size_t size, host_memory_kind kind, rmm::cuda_stream_view stream)
{
  if (kind == host_memory_kind::PINNED) {
    copy_pinned(dst, src, size, stream);
  } else if (kind == host_memory_kind::PAGEABLE) {
    copy_pageable(dst, src, size, stream);
  } else {
    CUDF_FAIL("Unsupported host memory kind");
  }
}

}  // namespace cudf::detail
