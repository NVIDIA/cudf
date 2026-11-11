/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/error.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/buffer>

#include <gtest/gtest-spi.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using cudf::test::scoped_current_device_resource;

static rmm::device_async_resource_ref get_output_mr(cudf::memory_resources resources)
{
  return resources.get_output_mr();
}

static_assert(std::is_nothrow_copy_constructible_v<cudf::memory_resources>);
static_assert(std::is_nothrow_move_constructible_v<cudf::memory_resources>);
static_assert(noexcept(std::declval<cudf::memory_resources const&>().get_output_mr()));
static_assert(noexcept(std::declval<cudf::memory_resources const&>().get_temporary_mr()));

TEST(MemoryResourcesTest, OneResourceUsesCurrentResourceAtConstruction)
{
  auto upstream       = cudf::get_current_device_resource_ref();
  auto output_mr      = rmm::mr::statistics_resource_adaptor{upstream};
  auto temporary_mr   = rmm::mr::statistics_resource_adaptor{upstream};
  auto replacement_mr = rmm::mr::statistics_resource_adaptor{upstream};

  scoped_current_device_resource temporary_scope{temporary_mr};
  cudf::memory_resources resources{output_mr};

  EXPECT_TRUE(resources.get_output_mr() == rmm::device_async_resource_ref{output_mr});
  EXPECT_TRUE(resources.get_temporary_mr() == rmm::device_async_resource_ref{temporary_mr});

  {
    scoped_current_device_resource replacement_scope{replacement_mr};
    cudf::memory_resources replacement_resources{output_mr};
    EXPECT_TRUE(replacement_resources.get_temporary_mr() ==
                rmm::device_async_resource_ref{replacement_mr});
  }
}

TEST(MemoryResourcesTest, TwoResourcesIgnoreCurrentResource)
{
  auto upstream     = cudf::get_current_device_resource_ref();
  auto output_mr    = rmm::mr::statistics_resource_adaptor{upstream};
  auto temporary_mr = rmm::mr::statistics_resource_adaptor{upstream};
  auto unrelated_mr = rmm::mr::statistics_resource_adaptor{upstream};

  scoped_current_device_resource current_scope{unrelated_mr};
  cudf::memory_resources resources{output_mr, temporary_mr};

  EXPECT_TRUE(resources.get_output_mr() == rmm::device_async_resource_ref{output_mr});
  EXPECT_TRUE(resources.get_temporary_mr() == rmm::device_async_resource_ref{temporary_mr});
  EXPECT_FALSE(resources.get_temporary_mr() == cudf::get_current_device_resource_ref());
}

TEST(MemoryResourcesTest, ResourceObjectAndRefCompatibility)
{
  auto output_mr    = rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
  auto temporary_mr = rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
  auto output_ref   = rmm::device_async_resource_ref{output_mr};
  auto temporary_ref = rmm::device_async_resource_ref{temporary_mr};

  static_assert(std::is_convertible_v<decltype(output_mr)&, cudf::memory_resources>);
  static_assert(std::is_convertible_v<rmm::device_async_resource_ref, cudf::memory_resources>);
  static_assert(
    std::is_constructible_v<cudf::memory_resources, decltype(output_mr)&, decltype(temporary_mr)&>);

  cudf::memory_resources from_objects{output_mr, temporary_mr};
  cudf::memory_resources from_refs{output_ref, temporary_ref};
  auto copied = from_objects;

  EXPECT_TRUE(from_refs.get_output_mr() == output_ref);
  EXPECT_TRUE(from_refs.get_temporary_mr() == temporary_ref);
  EXPECT_TRUE(copied.get_output_mr() == output_ref);
  EXPECT_TRUE(copied.get_temporary_mr() == temporary_ref);
  EXPECT_TRUE(get_output_mr(output_mr) == output_ref);
  EXPECT_TRUE(get_output_mr(output_ref) == output_ref);
}

TEST(MemoryResourceTestHarness, RestoresCurrentResourceDuringStackUnwinding)
{
  auto original       = cudf::get_current_device_resource_ref();
  auto replacement_mr = rmm::mr::statistics_resource_adaptor{original};

  EXPECT_THROW(
    {
      scoped_current_device_resource current_scope{replacement_mr};
      EXPECT_TRUE(cudf::get_current_device_resource_ref() ==
                  rmm::device_async_resource_ref{replacement_mr});
      throw std::runtime_error{"test exception"};
    },
    std::runtime_error);

  EXPECT_TRUE(cudf::get_current_device_resource_ref() == original);
}

TEST(MemoryResourceTestHarness, FailingCurrentResourceDetectsFallbackAndRestores)
{
  auto harness  = cudf::test::memory_resource_test_harness{};
  auto stream   = cudf::test::get_default_stream();
  auto original = cudf::get_current_device_resource_ref();

  {
    auto current_scope = harness.fail_on_current_device_resource_use();
    EXPECT_THROW((cuda::device_buffer<std::byte>{
                   stream, cudf::get_current_device_resource_ref(), 1, cuda::no_init}),
                 rmm::bad_alloc);
    harness.synchronize(stream);
  }

  EXPECT_TRUE(cudf::get_current_device_resource_ref() == original);
  EXPECT_NO_THROW((cuda::device_buffer<std::byte>{
    stream, cudf::get_current_device_resource_ref(), 1, cuda::no_init}));
}

TEST(MemoryResourceTestHarness, ExplicitResourcesWorkWithFailingCurrentResource)
{
  auto harness = cudf::test::memory_resource_test_harness{};
  auto stream  = cudf::test::get_default_stream();

  {
    auto current_scope = harness.fail_on_current_device_resource_use();
    auto resources     = harness.resources();
    auto output =
      cuda::device_buffer<std::byte>{stream, resources.get_output_mr(), 64, cuda::no_init};
    auto temporary =
      cuda::device_buffer<std::byte>{stream, resources.get_temporary_mr(), 32, cuda::no_init};
    harness.synchronize(stream);
  }

  harness.expect_no_live_allocations(stream);
}

TEST(MemoryResourceTestHarness, TracksSetupOutputAndTemporaryLifetimes)
{
  auto harness = cudf::test::memory_resource_test_harness{};
  auto stream  = cudf::test::get_default_stream();
  auto setup   = cuda::device_buffer<std::byte>{
    stream, rmm::device_async_resource_ref{harness.setup_mr()}, 16, cuda::no_init};

  {
    auto resources = harness.resources();
    auto output =
      cuda::device_buffer<std::byte>{stream, resources.get_output_mr(), 64, cuda::no_init};
    {
      auto temporary =
        cuda::device_buffer<std::byte>{stream, resources.get_temporary_mr(), 32, cuda::no_init};
    }

    harness.expect_output_allocations_live(stream);
    harness.expect_temporary_allocation_activity(stream);
    harness.expect_temporary_allocations_released(stream);
  }

  harness.expect_no_live_allocations(stream);
  EXPECT_GT(harness.setup_mr().get_bytes_counter().value, 0);
}

namespace {

std::vector<unsigned char> copy_to_host(void const* ptr, std::size_t bytes, cuda::stream_ref stream)
{
  std::vector<unsigned char> host(bytes);
  CUDF_CUDA_TRY(cudaMemcpyAsync(host.data(), ptr, bytes, cudaMemcpyDefault, stream.get()));
  stream.sync();
  return host;
}

std::pair<std::vector<unsigned char>, std::vector<unsigned char>>
temporary_bytes_on_allocate_and_release(cudf::test::temporary_memory_checks checks)
{
  constexpr std::size_t bytes = 64;
  auto const stream           = cudf::test::get_default_stream();
  auto current                = cudf::get_current_device_resource_ref();
  std::vector<unsigned char> released;
  auto recording_mr = rmm::mr::callback_memory_resource{
    [current](cuda::stream_ref s, std::size_t size, std::size_t alignment, void*) mutable {
      return current.allocate(s, size, alignment);
    },
    [current, &released](
      cuda::stream_ref s, void* ptr, std::size_t size, std::size_t alignment, void*) mutable {
      released = copy_to_host(ptr, size, s);
      current.deallocate(s, ptr, size, alignment);
    }};
  std::vector<unsigned char> allocated;
  {
    auto harness   = cudf::test::memory_resource_test_harness{recording_mr, checks};
    auto temporary = cuda::device_buffer<std::byte>{
      stream, harness.resources().get_temporary_mr(), bytes, cuda::no_init};
    allocated = copy_to_host(temporary.data(), bytes, stream);
    CUDF_CUDA_TRY(cudaMemsetAsync(temporary.data(), 0, bytes, stream.get()));
  }
  return {allocated, released};
}

bool all_poisoned(std::vector<unsigned char> const& bytes)
{
  return !bytes.empty() && std::all_of(bytes.begin(), bytes.end(), [](auto b) {
    return b == cudf::test::poison_byte;
  });
}

}  // namespace

TEST(MemoryResourceTestHarness, PoisonsTemporaryMemoryOnAllocateAndRelease)
{
  auto const [allocated, released] =
    temporary_bytes_on_allocate_and_release(cudf::test::temporary_memory_checks::POISON);
  EXPECT_TRUE(all_poisoned(allocated));
  EXPECT_TRUE(all_poisoned(released));
}

TEST(MemoryResourceTestHarness, DoesNotPoisonTemporaryMemoryByDefault)
{
  auto const [allocated, released] =
    temporary_bytes_on_allocate_and_release(cudf::test::temporary_memory_checks::NONE);
  EXPECT_EQ(released, std::vector<unsigned char>(released.size(), 0));
  EXPECT_FALSE(released.empty());
}

namespace {

std::vector<std::string> nonfatal_failure_messages(std::function<void()> const& statement)
{
  ::testing::TestPartResultArray results;
  {
    ::testing::ScopedFakeTestPartResultReporter reporter{
      ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results};
    statement();
  }
  std::vector<std::string> messages;
  for (int i = 0; i < results.size(); ++i) {
    messages.emplace_back(results.GetTestPartResult(i).message());
  }
  return messages;
}

bool any_contains(std::vector<std::string> const& messages, std::string const& text)
{
  return std::any_of(messages.begin(), messages.end(), [&](auto const& message) {
    return message.find(text) != std::string::npos;
  });
}

std::vector<std::string> buffer_api_failures(bool scratch_on_output, bool result_on_temporary)
{
  return nonfatal_failure_messages([&] {
    auto harness = cudf::test::memory_resource_test_harness{};
    auto stream  = cudf::test::get_default_stream();
    cudf::test::expect_api_uses_memory_resources(
      harness,
      [&](cudf::memory_resources mr) {
        auto scratch = cuda::device_buffer<std::byte>{
          stream,
          scratch_on_output ? mr.get_output_mr() : mr.get_temporary_mr(),
          32,
          cuda::no_init};
        return cuda::device_buffer<std::byte>{
          stream,
          result_on_temporary ? mr.get_temporary_mr() : mr.get_output_mr(),
          64,
          cuda::no_init};
      },
      [](auto const& result) { return result.size(); },
      [](auto const&) {},
      {cudf::test::output_allocation_expectation::EXACT,
       cudf::test::temporary_allocation_expectation::SOME},
      stream);
  });
}

}  // namespace

TEST(MemoryResourceTestHarness, AcceptsCorrectlyRoutedAllocations)
{
  EXPECT_TRUE(buffer_api_failures(false, false).empty());
}

TEST(MemoryResourceTestHarness, DetectsTemporaryAllocatedFromOutputResource)
{
  auto const failures = buffer_api_failures(true, false);
  EXPECT_FALSE(failures.empty());
  EXPECT_TRUE(any_contains(failures, "output_bytes.total"));
}

TEST(MemoryResourceTestHarness, DetectsResultAllocatedFromTemporaryResource)
{
  auto const failures = buffer_api_failures(false, true);
  EXPECT_FALSE(failures.empty());
  EXPECT_TRUE(any_contains(failures, "temporary_bytes.value"));
}
