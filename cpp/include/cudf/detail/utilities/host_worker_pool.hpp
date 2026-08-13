/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda_runtime.h>

#include <BS_thread_pool.hpp>

#include <cstddef>
#include <memory>

namespace cudf::detail {

/**
 * @brief Sentinel value indicating a thread not assigned to any pool level.
 *
 * This is the initial value of `thread_pool_level` for all threads. Once a thread
 * executes its first task from a pool, `thread_pool_level` is set to that pool's level
 * and never changes back to this value.
 */
constexpr int THREAD_POOL_LEVEL_NONE = -1;
static_assert(THREAD_POOL_LEVEL_NONE == -1, "THREAD_POOL_LEVEL_NONE must be -1");

/**
 * @brief Thread-local variable indicating which pool level this thread belongs to.
 *
 * Used by `host_worker_pool()` to route tasks to the correct nesting level.
 */
CUDF_EXPORT extern thread_local int thread_pool_level;

/**
 * @brief Set the OS thread name for a host worker pool thread.
 *
 * Called once per worker thread the first time it executes a task, so that
 * profilers (nsys, `top -H`, `/proc/<pid>/task/<tid>/comm`) show a meaningful
 * name like `cudf-hwp-L1` instead of the parent process name. POSIX only;
 * a no-op on platforms without `pthread_setname_np`.
 *
 * @param level Pool level assigned to this thread.
 */
CUDF_EXPORT void set_thread_name_for_pool_level(int level);

/**
 * @brief Thread pool wrapper that marks its threads with ownership.
 *
 * This wrapper ensures that threads know which pool they belong to, enabling
 * automatic routing to the correct nesting level.
 */
class CUDF_EXPORT hierarchical_thread_pool {
  BS::thread_pool pool_;
  int level_;

 public:
  /**
   * @brief Construct a new tiered thread pool.
   *
   * @param num_threads Number of threads in the pool
   * @param level Pool level
   */
  hierarchical_thread_pool(std::size_t num_threads, int level);

  /**
   * @brief Submit task and mark the executing thread with ownership.
   *
   * When a thread first executes a task, it's marked with the pool level.
   * This ownership persists for the lifetime of the thread.
   *
   * @tparam F Callable type
   * @param task Task to execute
   * @return Future for the task result
   */
  template <typename F>
  auto submit_task(F&& task)
  {
    // Wrap task in shared_ptr so lambda can call it without being mutable.
    // This is required because BS::thread_pool stores the lambda and calls
    // it from a const context.
    auto task_ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(task));

    // Capture the submitting thread's CUDA device so worker threads operate on
    // the correct device. Without this, worker threads default to device 0
    int device_id = 0;
    cudaGetDevice(&device_id);

    return pool_.submit_task([task_ptr, level = level_, device_id]() {
      // Mark this thread as owned by this pool's level (happens once per thread)
      if (thread_pool_level == THREAD_POOL_LEVEL_NONE) {
        thread_pool_level = level;
        set_thread_name_for_pool_level(level);
      }

      cudaSetDevice(device_id);

      return (*task_ptr)();
    });
  }

  /**
   * @brief Get the number of threads in this pool.
   */
  [[nodiscard]] std::size_t get_thread_count() const { return pool_.get_thread_count(); }
};

/**
 * @brief Retrieves the appropriate thread pool based on the calling thread's context.
 *
 * The returned pool is always different from the calling thread's pool.
 *
 * @return Reference to the thread pool
 */
hierarchical_thread_pool& host_worker_pool();
}  // namespace cudf::detail
