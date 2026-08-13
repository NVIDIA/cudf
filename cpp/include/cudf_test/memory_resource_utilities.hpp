/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>
#include <functional>
#include <utility>

namespace CUDF_EXPORT cudf {
namespace test {

/**
 * @brief Exception-safe owner for a temporary current device resource.
 *
 * The installed resource and the previous resource are held by owning type-erased resource values.
 * Destruction restores the previous resource, including during stack unwinding. Because the current
 * resource is device-global state, scopes must not overlap concurrent work that changes or uses the
 * current resource.
 */
class scoped_current_device_resource {
 public:
  /**
   * @brief Install `resource` as the current device resource for this object's lifetime.
   *
   * @param resource Owning type-erased resource value to install
   */
  explicit scoped_current_device_resource(
    cuda::mr::any_resource<cuda::mr::device_accessible> resource);

  ~scoped_current_device_resource();

  scoped_current_device_resource(scoped_current_device_resource const&)            = delete;
  scoped_current_device_resource& operator=(scoped_current_device_resource const&) = delete;
  scoped_current_device_resource(scoped_current_device_resource&&)                 = delete;
  scoped_current_device_resource& operator=(scoped_current_device_resource&&)      = delete;

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> _previous;
};

/** @brief Expected relationship between live and total output-resource allocations. */
enum class output_allocation_expectation {
  EXACT,         ///< Total output-resource bytes equal the live result size
  AT_LEAST_LIVE  ///< Total output-resource bytes may include released allocations
};

/** @brief Expected temporary-resource allocation activity. */
enum class temporary_allocation_expectation {
  NONE,  ///< No temporary-resource allocation is expected
  SOME   ///< At least one temporary-resource allocation is expected
};

/** @brief Allocation expectations shared by reusable memory-resource tests. */
struct memory_resource_expectations {
  output_allocation_expectation output{output_allocation_expectation::EXACT};
  temporary_allocation_expectation temporary{temporary_allocation_expectation::NONE};
};

/**
 * @brief Reusable memory-resource instrumentation for API routing tests.
 *
 * This harness owns independent statistics adaptors for setup, output, and temporary allocations.
 * It also owns a resource that throws on allocation and can be installed as the current resource
 * around only the API call under test. Counter assertion helpers synchronize the supplied stream
 * before inspecting allocation state.
 */
class memory_resource_test_harness {
 public:
  /**
   * @brief Construct instrumentation over `upstream`.
   *
   * @param upstream Resource used by each independent statistics adaptor
   */
  explicit memory_resource_test_harness(
    rmm::device_async_resource_ref upstream = cudf::get_current_device_resource_ref());

  /** @brief Return the statistics resource used to construct test inputs and expected results. */
  [[nodiscard]] rmm::mr::statistics_resource_adaptor& setup_mr() noexcept;

  /** @brief Return the statistics resource used for API output allocations. */
  [[nodiscard]] rmm::mr::statistics_resource_adaptor& output_mr() noexcept;

  /** @brief Return the statistics resource used for API temporary allocations. */
  [[nodiscard]] rmm::mr::statistics_resource_adaptor& temporary_mr() noexcept;

  /**
   * @brief Return explicit output and temporary resources without consulting the current resource.
   */
  [[nodiscard]] cudf::memory_resources resources() noexcept;

  /**
   * @brief Install a current resource that throws on allocation for the returned scope's lifetime.
   *
   * Keep the returned scope limited to the API invocation and stream synchronization. Construct
   * inputs and validate results outside this scope.
   *
   * @return Scope that restores the prior current resource on destruction
   */
  [[nodiscard]] scoped_current_device_resource fail_on_current_device_resource_use();

  /** @brief Synchronize `stream` before leaving a failing-resource API scope. */
  void synchronize(cuda::stream_ref stream = cudf::test::get_default_stream()) const;

  /** @brief Assert that output allocations are live after synchronizing `stream`. */
  void expect_output_allocations_live(
    cuda::stream_ref stream = cudf::test::get_default_stream()) const;

  /** @brief Assert that temporary allocations were made after synchronizing `stream`. */
  void expect_temporary_allocation_activity(
    cuda::stream_ref stream = cudf::test::get_default_stream()) const;

  /** @brief Assert that no temporary allocations remain live after synchronizing `stream`. */
  void expect_temporary_allocations_released(
    cuda::stream_ref stream = cudf::test::get_default_stream()) const;

  /**
   * @brief Assert output ownership and temporary allocation activity.
   *
   * @param expected_output_bytes Number of bytes owned by the live result
   * @param expectations Expected output and temporary allocation behavior
   * @param stream Stream to synchronize before inspecting counters
   */
  void expect_resource_usage(std::size_t expected_output_bytes,
                             memory_resource_expectations expectations = {},
                             cuda::stream_ref stream = cudf::test::get_default_stream()) const;

  /**
   * @brief Assert that no output or temporary allocations remain after synchronizing `stream`.
   *
   * Setup allocations are intentionally excluded because test inputs may remain alive while result
   * allocation lifetimes are checked.
   */
  void expect_no_live_allocations(cuda::stream_ref stream = cudf::test::get_default_stream()) const;

 private:
  rmm::mr::statistics_resource_adaptor _setup_mr;
  rmm::mr::statistics_resource_adaptor _output_mr;
  rmm::mr::statistics_resource_adaptor _temporary_mr;
  rmm::mr::callback_memory_resource _failing_mr;
};

/**
 * @brief Verify that an owning result uses one explicitly supplied output resource.
 *
 * `factory` receives a statistics resource and returns a column wrapper. The helper releases the
 * wrapper into an owning column and compares its `alloc_size()` with the resource counters.
 * The current device resource is used only as the upstream allocator for the statistics resource;
 * the statistics resource itself is passed explicitly to `factory`. Returned allocations that
 * bypass the supplied statistics resource are therefore excluded from its counters and cause the
 * ownership checks to fail.
 *
 * @tparam Factory Callable type used to construct the result
 * @param factory Callable invoked with the output resource
 * @param output_expectation Expected relationship between live and total output bytes
 * @param stream Stream to synchronize before inspecting allocation counters
 */
template <typename Factory>
void expect_output_uses_resource(
  Factory&& factory,
  output_allocation_expectation output_expectation = output_allocation_expectation::EXACT,
  cuda::stream_ref stream                          = cudf::test::get_default_stream())
{
  auto output_mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());

  {
    auto result = std::invoke(std::forward<Factory>(factory), output_mr).release();
    stream.sync();
    auto const output_bytes = output_mr.get_bytes_counter();
    auto const result_bytes = result->alloc_size();

    EXPECT_EQ(result_bytes, static_cast<std::size_t>(output_bytes.value));
    if (output_expectation == output_allocation_expectation::EXACT) {
      EXPECT_EQ(result_bytes, static_cast<std::size_t>(output_bytes.total));
    } else {
      EXPECT_GE(output_bytes.total, output_bytes.value);
    }
  }

  stream.sync();
  EXPECT_EQ(output_mr.get_bytes_counter().value, 0);
}

/**
 * @brief Verify output and temporary routing for an owning result.
 *
 * `factory` receives distinct output and temporary resources and returns a column wrapper. The
 * helper releases the wrapper into an owning column and compares its `alloc_size()` with the
 * resource counters.
 *
 * @tparam Factory Callable type used to construct the result
 * @param factory Callable invoked with distinct resources
 * @param expectations Expected output and temporary allocation behavior
 * @param stream Stream to synchronize before inspecting allocation counters
 */
template <typename Factory>
void expect_output_uses_distinct_resources(
  Factory&& factory,
  memory_resource_expectations expectations = {},
  cuda::stream_ref stream                   = cudf::test::get_default_stream())
{
  auto harness = memory_resource_test_harness{};

  {
    auto result = std::invoke(std::forward<Factory>(factory), harness.resources()).release();
    harness.expect_resource_usage(result->alloc_size(), expectations, stream);
  }

  harness.expect_no_live_allocations(stream);
}

/**
 * @brief Invoke an API with explicit resources and validate its allocation routing.
 *
 * The current resource throws on allocation only while `factory` runs and `stream` synchronizes.
 * Allocation accounting and `validate_result` run after the prior current resource is restored.
 * `result_size` makes this helper usable with columns, tables, buffers, and compound result types.
 *
 * @tparam Factory Callable accepting `cudf::memory_resources` and returning an owning result
 * @tparam ResultSize Callable returning the result's output allocation size in bytes
 * @tparam ResultValidator Callable that validates the result outside the failing-resource scope
 * @param harness Reusable setup, output, temporary, and failing-resource instrumentation
 * @param factory API invocation accepting the harness resources
 * @param result_size Callable returning the number of output bytes owned by the result
 * @param validate_result Callable that validates the result after restoring the current resource
 * @param expectations Expected output and temporary allocation behavior
 * @param stream Stream used by the API invocation
 */
template <typename Factory, typename ResultSize, typename ResultValidator>
void expect_api_uses_memory_resources(memory_resource_test_harness& harness,
                                      Factory&& factory,
                                      ResultSize&& result_size,
                                      ResultValidator&& validate_result,
                                      memory_resource_expectations expectations = {},
                                      cuda::stream_ref stream = cudf::test::get_default_stream())
{
  {
    auto result = [&]() {
      auto current_scope = harness.fail_on_current_device_resource_use();
      auto api_result    = std::invoke(std::forward<Factory>(factory), harness.resources());
      harness.synchronize(stream);
      return api_result;
    }();

    auto const output_bytes =
      static_cast<std::size_t>(std::invoke(std::forward<ResultSize>(result_size), result));
    harness.expect_resource_usage(output_bytes, expectations, stream);
    std::invoke(std::forward<ResultValidator>(validate_result), result);
  }

  harness.expect_no_live_allocations(stream);
}

}  // namespace test
}  // namespace CUDF_EXPORT cudf
