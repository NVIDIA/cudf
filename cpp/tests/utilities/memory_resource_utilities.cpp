/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/error.hpp>

#include <cuda/stream_ref>

#include <cstddef>
#include <utility>

namespace cudf::test {

scoped_current_device_resource::scoped_current_device_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> resource)
  : _previous{cudf::set_current_device_resource(std::move(resource))}
{
}

scoped_current_device_resource::~scoped_current_device_resource()
{
  std::ignore = cudf::set_current_device_resource(std::move(_previous));
}

memory_resource_test_harness::memory_resource_test_harness(rmm::device_async_resource_ref upstream)
  : _setup_mr{upstream},
    _output_mr{upstream},
    _temporary_mr{upstream},
    _failing_mr{[](std::size_t, cuda::stream_ref, void*) -> void* {
                  throw rmm::bad_alloc{"Unexpected allocation from the current device resource"};
                },
                [](void*, std::size_t, cuda::stream_ref, void*) {}}
{
}

rmm::mr::statistics_resource_adaptor& memory_resource_test_harness::setup_mr() noexcept
{
  return _setup_mr;
}

rmm::mr::statistics_resource_adaptor& memory_resource_test_harness::output_mr() noexcept
{
  return _output_mr;
}

rmm::mr::statistics_resource_adaptor& memory_resource_test_harness::temporary_mr() noexcept
{
  return _temporary_mr;
}

cudf::memory_resources memory_resource_test_harness::resources() noexcept
{
  return cudf::memory_resources{_output_mr, _temporary_mr};
}

scoped_current_device_resource memory_resource_test_harness::fail_on_current_device_resource_use()
{
  return scoped_current_device_resource{_failing_mr};
}

void memory_resource_test_harness::synchronize(cuda::stream_ref stream) const { stream.sync(); }

void memory_resource_test_harness::expect_output_allocations_live(cuda::stream_ref stream) const
{
  synchronize(stream);
  EXPECT_GT(_output_mr.get_bytes_counter().value, 0);
}

void memory_resource_test_harness::expect_temporary_allocation_activity(
  cuda::stream_ref stream) const
{
  synchronize(stream);
  EXPECT_GT(_temporary_mr.get_bytes_counter().total, 0);
}

void memory_resource_test_harness::expect_temporary_allocations_released(
  cuda::stream_ref stream) const
{
  synchronize(stream);
  EXPECT_EQ(_temporary_mr.get_bytes_counter().value, 0);
}

void memory_resource_test_harness::expect_resource_usage(std::size_t expected_output_bytes,
                                                         memory_resource_expectations expectations,
                                                         cuda::stream_ref stream) const
{
  synchronize(stream);
  auto const output_bytes    = _output_mr.get_bytes_counter();
  auto const temporary_bytes = _temporary_mr.get_bytes_counter();

  EXPECT_EQ(expected_output_bytes, static_cast<std::size_t>(output_bytes.value));
  if (expectations.output == output_allocation_expectation::EXACT) {
    EXPECT_EQ(expected_output_bytes, static_cast<std::size_t>(output_bytes.total));
  } else {
    EXPECT_GE(output_bytes.total, output_bytes.value);
  }

  EXPECT_EQ(temporary_bytes.value, 0);
  if (expectations.temporary == temporary_allocation_expectation::SOME) {
    EXPECT_GT(temporary_bytes.total, 0);
  } else {
    EXPECT_EQ(temporary_bytes.total, 0);
  }
}

void memory_resource_test_harness::expect_no_live_allocations(cuda::stream_ref stream) const
{
  synchronize(stream);
  EXPECT_EQ(_output_mr.get_bytes_counter().value, 0);
  EXPECT_EQ(_temporary_mr.get_bytes_counter().value, 0);
}

}  // namespace cudf::test
