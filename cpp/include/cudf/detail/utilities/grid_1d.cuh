/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief A kernel grid configuration construction gadget for simple
 * one-dimensional kernels, with protection against integer overflow.
 */
class grid_1d {
 public:
  thread_index_type const num_threads_per_block;
  thread_index_type const num_blocks;

  /**
   * @param overall_num_elements The number of elements the kernel needs to
   * handle/process, in its main, one-dimensional/linear input (e.g. one or more
   * cuDF columns)
   * @param num_threads_per_block The grid block size, determined according to
   * the kernel's specific features (amount of shared memory necessary, SM
   * functional units use pattern etc.); this can't be determined
   * generically/automatically (as opposed to the number of blocks)
   * @param elements_per_thread Typically, a single kernel thread processes more
   * than a single element; this affects the number of threads the grid must
   * contain
   */
  grid_1d(thread_index_type overall_num_elements,
          thread_index_type num_threads_per_block,
          thread_index_type elements_per_thread = 1);

  /**
   * @brief Returns the global thread index in a 1D grid.
   *
   * The returned index is unique across the entire grid.
   *
   * @param thread_id The thread index within the block
   * @param block_id The block index within the grid
   * @param num_threads_per_block The number of threads per block
   * @return thread_index_type The global thread index
   */
  __device__ static constexpr thread_index_type global_thread_id(
    thread_index_type thread_id,
    thread_index_type block_id,
    thread_index_type num_threads_per_block)
  {
    return thread_id + block_id * num_threads_per_block;
  }

  /**
   * @brief Returns the global thread index of the current thread in a 1D grid.
   *
   * @return thread_index_type The global thread index
   */
  static __device__ thread_index_type global_thread_id()
  {
    return global_thread_id(threadIdx.x, blockIdx.x, blockDim.x);
  }

  /**
   * @brief Returns the global thread index of the current thread in a 1D grid.
   *
   * @tparam num_threads_per_block The number of threads per block
   *
   * @return thread_index_type The global thread index
   */
  template <thread_index_type num_threads_per_block>
  static __device__ thread_index_type global_thread_id()
  {
    return global_thread_id(threadIdx.x, blockIdx.x, num_threads_per_block);
  }

  /**
   * @brief Returns the stride of a 1D grid.
   *
   * The returned stride is the total number of threads in the grid.
   *
   * @param num_threads_per_block The number of threads per block
   * @param num_blocks_per_grid Number of blocks in the grid
   * @return thread_index_type The grid stride
   */
  __device__ static constexpr thread_index_type grid_stride(thread_index_type num_threads_per_block,
                                                            thread_index_type num_blocks_per_grid)
  {
    return num_threads_per_block * num_blocks_per_grid;
  }

  /**
   * @brief Returns the stride of the current 1D grid.
   *
   * @return thread_index_type The number of threads in the grid.
   */
  static __device__ thread_index_type grid_stride() { return grid_stride(blockDim.x, gridDim.x); }

  /**
   * @brief Returns the stride of the current 1D grid.
   *
   * @tparam num_threads_per_block The number of threads per block
   *
   * @return thread_index_type The number of threads in the grid.
   */
  template <thread_index_type num_threads_per_block>
  static __device__ thread_index_type grid_stride()
  {
    return grid_stride(num_threads_per_block, gridDim.x);
  }
};

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
