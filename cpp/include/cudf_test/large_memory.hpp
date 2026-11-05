/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <string>

namespace cudf::test {

/**
 * @brief Returns why a test needing `bytes` of device memory should be skipped, or an empty string
 * if it can run.
 *
 * Large-memory tests are skipped unless `GTEST_CUDF_LARGE_MEMORY_TESTS=1`, and then only run if
 * the current CUDA device has at least `bytes` of total memory.
 */
inline std::string large_memory_skip_reason(std::size_t bytes)
{
  auto const* env = std::getenv("GTEST_CUDF_LARGE_MEMORY_TESTS");
  if (env == nullptr or std::string{env} != "1") {
    return "Large-memory tests are disabled, set GTEST_CUDF_LARGE_MEMORY_TESTS=1 to enable";
  }
  std::size_t free{};
  std::size_t total{};
  CUDF_CUDA_TRY(cudaMemGetInfo(&free, &total));
  if (total < bytes) {
    return "Requires " + std::to_string(bytes >> 20) +
           " MiB of device memory, but the device has " + std::to_string(total >> 20) + " MiB";
  }
  return {};
}

}  // namespace cudf::test

/// Skips the current test unless large-memory tests are enabled and the device has `bytes` memory.
#define CUDF_TEST_REQUIRE_DEVICE_MEMORY_OR_SKIP(bytes)                         \
  do {                                                                         \
    if (auto const skip_reason_ = cudf::test::large_memory_skip_reason(bytes); \
        not skip_reason_.empty()) {                                            \
      GTEST_SKIP() << skip_reason_;                                            \
    }                                                                          \
  } while (0)
