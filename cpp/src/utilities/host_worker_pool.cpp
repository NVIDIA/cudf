/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/utilities/error.hpp>

#include <pthread.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <utility>
#include <vector>

namespace cudf::detail {

// Thread-local variable tracking which pool owns this thread
thread_local int thread_pool_level = THREAD_POOL_LEVEL_NONE;

// Constructor for hierarchical_thread_pool
hierarchical_thread_pool::hierarchical_thread_pool(std::size_t num_threads, int level)
  : pool_(num_threads), level_(level)
{
}

void set_thread_name_for_pool_level(int level)
{
  // Linux caps thread names (comm) at 15 chars + NUL. Keep the name short.
  std::array<char, 16> name{};
  std::snprintf(name.data(), name.size(), "cudf-hwp-L%d", level);
  pthread_setname_np(pthread_self(), name.data());
}

namespace {

// Dynamic pool storage - grows as needed
std::vector<std::unique_ptr<hierarchical_thread_pool>> g_pools;
// Reader-writer lock for pool access
std::shared_mutex g_pools_mutex;

[[nodiscard]] std::size_t pool_size()
{
  static std::size_t const default_pool_size =
    std::min<std::size_t>(16, std::thread::hardware_concurrency() / 4);
  return cudf::detail::getenv_or("LIBCUDF_NUM_HOST_WORKERS", default_pool_size);
}

/**
 * @brief Get or create pool for specific level (dynamic creation).
 */
hierarchical_thread_pool& pool(int level)
{
  {
    // Shared lock is sufficient for read operations
    std::shared_lock<std::shared_mutex> read_lock(g_pools_mutex);
    if (std::cmp_less(level, g_pools.size()) && g_pools[level]) { return *g_pools[level]; }
  }

  // Exclusive lock is required for write operations
  std::unique_lock<std::shared_mutex> write_lock(g_pools_mutex);

  // Double-check after acquiring write lock
  if (std::cmp_less(level, g_pools.size()) && g_pools[level]) { return *g_pools[level]; }

  // Create and add the pool to the vector
  CUDF_EXPECTS(std::cmp_equal(level, g_pools.size()),
               "Invalid pool level, should only increase by 1");
  g_pools.emplace_back(std::make_unique<hierarchical_thread_pool>(pool_size(), level));
  return *g_pools.back();
}

}  // anonymous namespace

hierarchical_thread_pool& host_worker_pool()
{
  // Use the pool at the next level up from the calling thread's level
  return pool(thread_pool_level + 1);
}

}  // namespace cudf::detail
