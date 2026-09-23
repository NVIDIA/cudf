/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <stdexcept>
#include <type_traits>
#include <utility>

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
    EXPECT_THROW(rmm::device_buffer(1, stream), rmm::bad_alloc);
    harness.synchronize(stream);
  }

  EXPECT_TRUE(cudf::get_current_device_resource_ref() == original);
  EXPECT_NO_THROW(rmm::device_buffer(1, stream));
}

TEST(MemoryResourceTestHarness, ExplicitResourcesWorkWithFailingCurrentResource)
{
  auto harness = cudf::test::memory_resource_test_harness{};
  auto stream  = cudf::test::get_default_stream();

  {
    auto current_scope = harness.fail_on_current_device_resource_use();
    auto resources     = harness.resources();
    auto output        = rmm::device_buffer(64, stream, resources.get_output_mr());
    auto temporary     = rmm::device_buffer(32, stream, resources.get_temporary_mr());
    harness.synchronize(stream);
  }

  harness.expect_no_live_allocations(stream);
}

TEST(MemoryResourceTestHarness, TracksSetupOutputAndTemporaryLifetimes)
{
  auto harness = cudf::test::memory_resource_test_harness{};
  auto stream  = cudf::test::get_default_stream();
  auto setup   = rmm::device_buffer(16, stream, harness.setup_mr());

  {
    auto resources = harness.resources();
    auto output    = rmm::device_buffer(64, stream, resources.get_output_mr());
    {
      auto temporary = rmm::device_buffer(32, stream, resources.get_temporary_mr());
    }

    harness.expect_output_allocations_live(stream);
    harness.expect_temporary_allocation_activity(stream);
    harness.expect_temporary_allocations_released(stream);
  }

  harness.expect_no_live_allocations(stream);
  EXPECT_GT(harness.setup_mr().get_bytes_counter().value, 0);
}
