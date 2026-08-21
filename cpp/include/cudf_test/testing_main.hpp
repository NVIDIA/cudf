/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/cxxopts.hpp>
#include <cudf_test/stream_checking_resource_adaptor.hpp>

#include <cudf/context.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/binning_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <iostream>

namespace CUDF_EXPORT cudf {
namespace test {

struct config {
  std::string rmm_mode;
  std::string stream_mode;
  std::string stream_error_mode;
};

// Wrapper that adds host_accessible + device_accessible properties to a pool_memory_resource.
// pool_memory_resource doesn't propagate the host_accessible property from its pinned upstream
// through its type, so this shim is needed to satisfy the host_device_async_resource_ref
// required by set_pinned_memory_resource.
struct pinned_pool {
  rmm::mr::pinned_host_memory_resource pinned_mr;
  rmm::mr::pool_memory_resource pool_mr;

  explicit pinned_pool(std::size_t pool_size)
    : pool_mr{rmm::device_async_resource_ref{pinned_mr}, pool_size}
  {
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return pool_mr.allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }
  void deallocate_sync(void* p,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    pool_mr.deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, p, bytes, alignment);
  }
  void* allocate(cuda::stream_ref s,
                 std::size_t bytes,
                 std::size_t a = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return pool_mr.allocate(s, bytes, a);
  }
  void deallocate(cuda::stream_ref s,
                  void* p,
                  std::size_t bytes,
                  std::size_t a = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    pool_mr.deallocate(s, p, bytes, a);
  }
  bool operator==(pinned_pool const& o) const noexcept { return &pool_mr == &o.pool_mr; }
  bool operator!=(pinned_pool const& o) const noexcept { return &pool_mr != &o.pool_mr; }
  friend void get_property(pinned_pool const&, cuda::mr::device_accessible) noexcept {}
  friend void get_property(pinned_pool const&, cuda::mr::host_accessible) noexcept {}
};

/// MR factory functions
inline auto make_cuda() { return rmm::mr::cuda_memory_resource{}; }

inline auto make_async() { return rmm::mr::cuda_async_memory_resource{}; }

inline auto make_managed() { return rmm::mr::managed_memory_resource{}; }

inline auto make_pool()
{
  auto const [free, total] = rmm::available_device_memory();
  auto const min_alloc =
    rmm::align_down(std::min(free, total / 10), rmm::CUDA_ALLOCATION_ALIGNMENT);
  return rmm::mr::pool_memory_resource{rmm::mr::cuda_memory_resource{}, min_alloc};
}

inline auto make_arena() { return rmm::mr::arena_memory_resource{rmm::mr::cuda_memory_resource{}}; }

inline auto make_binning()
{
  // Add a binning_memory_resource with fixed-size bins of sizes 256, 512, 1024, 2048 and 4096KiB
  // Larger allocations will use the pool resource
  return rmm::mr::binning_memory_resource{make_pool(), 18, 22};
}

/**
 * @brief Creates a memory resource for the unit test environment
 * given the name of the allocation mode.
 *
 * The returned resource instance must be kept alive for the duration of
 * the tests. Attaching the resource to a TestEnvironment causes
 * issues since the environment objects are not destroyed until
 * after the runtime is shutdown.
 *
 * @throw cudf::logic_error if the `allocation_mode` is unsupported.
 *
 * @param allocation_mode String identifies which resource type.
 *        Accepted types are "pool", "cuda", "async", "arena", "binning", and "managed".
 * @return Memory resource instance
 */
inline cuda::mr::any_resource<cuda::mr::device_accessible> create_memory_resource(
  std::string const& allocation_mode)
{
  if (allocation_mode == "binning") return make_binning();
  if (allocation_mode == "cuda") return make_cuda();
  if (allocation_mode == "async") return make_async();
  if (allocation_mode == "pool") return make_pool();
  if (allocation_mode == "arena") return make_arena();
  if (allocation_mode == "managed") return make_managed();
  CUDF_FAIL("Invalid RMM allocation mode: " + allocation_mode);
}

}  // namespace test
}  // namespace CUDF_EXPORT cudf

/**
 * @brief Parses the cuDF test command line options.
 *
 * Currently only supports 'rmm_mode' string parameter, which set the rmm
 * allocation mode. The default value of the parameter is 'pool'.
 * Environment variable 'CUDF_TEST_RMM_MODE' can also be used to set the rmm
 * allocation mode. If both are set, the value of 'rmm_mode' string parameter
 * takes precedence.
 *
 * @return Parsing results in the form of unordered map
 */
inline auto parse_cudf_test_opts(int argc, char** argv)
{
  try {
    cxxopts::Options options(argv[0], " - cuDF tests command line options");
    char const* env_rmm_mode = std::getenv("GTEST_CUDF_RMM_MODE");  // Overridden by CLI options
    char const* env_stream_mode =
      std::getenv("GTEST_CUDF_STREAM_MODE");  // Overridden by CLI options
    char const* env_stream_error_mode =
      std::getenv("GTEST_CUDF_STREAM_ERROR_MODE");  // Overridden by CLI options
    auto default_rmm_mode          = env_rmm_mode ? env_rmm_mode : "async";
    auto default_stream_mode       = env_stream_mode ? env_stream_mode : "default";
    auto default_stream_error_mode = env_stream_error_mode ? env_stream_error_mode : "error";
    options.allow_unrecognised_options().add_options()(
      "rmm_mode",
      "RMM allocation mode",
      cxxopts::value<std::string>()->default_value(default_rmm_mode));
    // `new_cudf_default` means that cudf::get_default_stream has been patched,
    // so we raise errors anywhere that a CUDA default stream is observed
    // instead of cudf::get_default_stream(). This corresponds to compiling
    // identify_stream_usage with STREAM_MODE_TESTING=OFF (must do both at the
    // same time).
    // `new_testing_default` means that cudf::test::get_default_stream has been
    // patched, so we raise errors anywhere that _any_ other stream is
    // observed. This corresponds to compiling identify_stream_usage with
    // STREAM_MODE_TESTING=ON (must do both at the same time).
    options.allow_unrecognised_options().add_options()(
      "stream_mode",
      "Whether to use a non-default stream",
      cxxopts::value<std::string>()->default_value(default_stream_mode));
    options.allow_unrecognised_options().add_options()(
      "stream_error_mode",
      "Whether to error or print to stdout when a non-default stream is observed and stream_mode "
      "is not \"default\"",
      cxxopts::value<std::string>()->default_value(default_stream_error_mode));
    return options.parse(argc, argv);
  } catch (cxxopts::OptionException const& e) {
    CUDF_FAIL("Error parsing command line options");
  }
}

/**
 * @brief Sets up the memory resource for the test run.
 *
 * @param config Command line options returned by parse_cudf_test_opts
 */
inline auto make_memory_resource_adaptor(cudf::test::config const& config)
{
  auto resource = cudf::test::create_memory_resource(config.rmm_mode);
  cudf::set_current_device_resource(resource);
  return resource;
}

/**
 * @brief Sets up stream mode memory resource adaptor
 *
 * The resource adaptor is only set as the current device resource if the
 * stream mode is enabled.
 *
 * @param config Command line options returned by parse_cudf_test_opts
 */
inline void make_stream_mode_adaptor(cudf::test::config const& config)
{
  if ((config.stream_mode == "new_cudf_default") || (config.stream_mode == "new_testing_default")) {
    auto const error_on_invalid_stream = (config.stream_error_mode == "error");
    auto const check_default_stream    = (config.stream_mode == "new_cudf_default");
    auto resource                      = cudf::reset_current_device_resource();
    auto adaptor                       = cudf::test::stream_checking_resource_adaptor(
      resource, error_on_invalid_stream, check_default_stream);
    cudf::set_current_device_resource(adaptor);
  }
}

/**
 * @brief Should be called in every test program that uses rmm allocators.
 * Parses the command line to customize test behavior, like the allocation mode
 * used for creating the default memory resource.
 *
 */
inline void init_cudf_test(int argc, char** argv, cudf::test::config const& config_override = {})
{
  auto const cmd_opts       = parse_cudf_test_opts(argc, argv);
  cudf::test::config config = config_override;
  if (config.rmm_mode.empty()) { config.rmm_mode = cmd_opts["rmm_mode"].as<std::string>(); }

  if (config.stream_mode.empty()) {
    config.stream_mode = cmd_opts["stream_mode"].as<std::string>();
  }

  if (config.stream_error_mode.empty()) {
    config.stream_error_mode = cmd_opts["stream_error_mode"].as<std::string>();
  }

  make_memory_resource_adaptor(config);
  make_stream_mode_adaptor(config);
}

/**
 * @brief Macro that defines main function for gtest programs that use rmm
 *
 * This `main` function is a wrapper around the google test generated `main`,
 * maintaining the original functionality.
 */
#define CUDF_TEST_PROGRAM_MAIN()                                                                 \
  int main(int argc, char** argv)                                                                \
  {                                                                                              \
    cudf::initialize();                                                                          \
    ::testing::InitGoogleTest(&argc, argv);                                                      \
    init_cudf_test(argc, argv);                                                                  \
    if (std::getenv("GTEST_CUDF_MEMORY_PEAK")) {                                                 \
      auto mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());   \
      cudf::set_current_device_resource(mr);                                                     \
      auto rc = RUN_ALL_TESTS();                                                                 \
      std::cout << "Peak memory usage " << mr.get_bytes_counter().peak << " bytes" << std::endl; \
      cudf::teardown();                                                                          \
      rmm::mr::reset_current_device_resource();                                                  \
      return rc;                                                                                 \
    } else {                                                                                     \
      auto rc = RUN_ALL_TESTS();                                                                 \
      cudf::teardown();                                                                          \
      rmm::mr::reset_current_device_resource();                                                  \
      return rc;                                                                                 \
    }                                                                                            \
  }
