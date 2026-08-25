/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream_ref>

#include <algorithm>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Get the number of multiprocessors on the device
 */
cudf::size_type num_multiprocessors();

/**
 * @brief Check if the current device has integrated memory.
 */
[[nodiscard]] bool has_integrated_memory();

/**
 * @brief Get the number of elements that can be processed per thread.
 *
 * @param[in] kernel The kernel for which the elements per thread needs to be assessed
 * @param[in] total_size Number of elements
 * @param[in] block_size Expected block size
 * @param[in] max_per_thread Maximum number of elements per thread
 *
 * @return cudf::size_type Elements per thread that can be processed for given specification.
 */
template <typename Kernel>
cudf::size_type elements_per_thread(Kernel kernel,
                                    cudf::size_type total_size,
                                    cudf::size_type block_size,
                                    cudf::size_type max_per_thread = 32)
{
  CUDF_FUNC_RANGE();

  // calculate theoretical occupancy
  int max_blocks = 0;
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block_size, 0));

  int per_thread = total_size / (max_blocks * num_multiprocessors() * block_size);
  return std::clamp(per_thread, 1, max_per_thread);
}

/**
 * @brief Synchronize a CUDA stream.
 *
 * `cuda::stream_ref::sync()` uses the CUDA driver API, which requires the stream's CUDA context to
 * be current on the calling thread. Use the runtime API for host-facing synchronization points that
 * may be called from threads without a current CUDA context.
 */
inline void sync_stream(cuda::stream_ref stream)
{
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
