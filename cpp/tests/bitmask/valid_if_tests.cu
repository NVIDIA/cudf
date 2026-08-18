/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>

struct ValidIfTest : public cudf::test::BaseFixture {};

namespace {
struct odds_valid {
  __host__ __device__ bool operator()(cudf::size_type i) { return i % 2; }
};
struct all_valid {
  __host__ __device__ bool operator()(cudf::size_type i) { return true; }
};
struct all_null {
  __host__ __device__ bool operator()(cudf::size_type i) { return false; }
};
}  // namespace

TEST_F(ValidIfTest, EmptyRange)
{
  auto actual        = cudf::detail::valid_if(cuda::counting_iterator<cudf::size_type>{0},
                                       cuda::counting_iterator<cudf::size_type>{0},
                                       odds_valid{},
                                       cudf::get_default_stream(),
                                       cudf::get_current_device_resource_ref());
  auto const& buffer = actual.first;
  EXPECT_EQ(0u, buffer.size());
  EXPECT_EQ(nullptr, buffer.data());
  EXPECT_EQ(0, actual.second);
}

TEST_F(ValidIfTest, ExplicitMemoryResourcesEmptyRange)
{
  auto harness = cudf::test::memory_resource_test_harness{this->mr()};
  auto stream  = cudf::get_default_stream();

  auto actual = [&] {
    auto current_scope = harness.fail_on_current_device_resource_use();
    auto result        = cudf::detail::valid_if(cuda::counting_iterator<cudf::size_type>{0},
                                         cuda::counting_iterator<cudf::size_type>{0},
                                         odds_valid{},
                                         stream,
                                         harness.resources());
    harness.synchronize(stream);
    return result;
  }();

  EXPECT_EQ(0u, actual.first.size());
  EXPECT_EQ(nullptr, actual.first.data());
  EXPECT_EQ(0, actual.second);
  harness.expect_no_live_allocations(stream);
  EXPECT_EQ(0, harness.output_mr().get_bytes_counter().total);
  EXPECT_EQ(0, harness.temporary_mr().get_bytes_counter().total);
}

TEST_F(ValidIfTest, InvalidRange)
{
  EXPECT_THROW(cudf::detail::valid_if(cuda::counting_iterator<cudf::size_type>{1},
                                      cuda::counting_iterator<cudf::size_type>{0},
                                      odds_valid{},
                                      cudf::get_default_stream(),
                                      cudf::get_current_device_resource_ref()),
               cudf::logic_error);
}

TEST_F(ValidIfTest, OddsValid)
{
  auto iter     = cudf::detail::make_counting_transform_iterator(0, odds_valid{});
  auto expected = cudf::test::detail::make_null_mask(iter, iter + 10000);
  auto actual   = cudf::detail::valid_if(cuda::counting_iterator<cudf::size_type>{0},
                                       cuda::counting_iterator<cudf::size_type>{10000},
                                       odds_valid{},
                                       cudf::get_default_stream(),
                                       cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.first.data(), actual.first.data(), expected.first.size());
  EXPECT_EQ(5000, actual.second);
  EXPECT_EQ(expected.second, actual.second);
}

TEST_F(ValidIfTest, ExplicitMemoryResourceControl)
{
  auto harness              = cudf::test::memory_resource_test_harness{this->mr()};
  auto stream               = cudf::get_default_stream();
  auto comparison_resources = cudf::memory_resources{harness.setup_mr(), harness.setup_mr()};
  auto iter                 = cudf::detail::make_counting_transform_iterator(0, odds_valid{});
  auto expected =
    cudf::test::detail::make_null_mask(iter, iter + 10000, stream, comparison_resources);

  {
    auto actual = [&] {
      auto current_scope = harness.fail_on_current_device_resource_use();
      auto result        = cudf::detail::valid_if(cuda::counting_iterator<cudf::size_type>{0},
                                           cuda::counting_iterator<cudf::size_type>{10000},
                                           odds_valid{},
                                           stream,
                                           harness.resources());
      harness.synchronize(stream);
      return result;
    }();

    harness.expect_output_allocations_live(stream);
    harness.expect_temporary_allocation_activity(stream);
    harness.expect_temporary_allocations_released(stream);
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.first.data(),
                                   actual.first.data(),
                                   expected.first.size(),
                                   stream,
                                   comparison_resources);
    EXPECT_EQ(5000, actual.second);
    EXPECT_EQ(expected.second, actual.second);
  }

  harness.expect_no_live_allocations(stream);
}

TEST_F(ValidIfTest, AllValid)
{
  auto iter     = cudf::detail::make_counting_transform_iterator(0, all_valid{});
  auto expected = cudf::test::detail::make_null_mask(iter, iter + 10000);
  auto actual   = cudf::detail::valid_if(cuda::counting_iterator<cudf::size_type>{0},
                                       cuda::counting_iterator<cudf::size_type>{10000},
                                       all_valid{},
                                       cudf::get_default_stream(),
                                       cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.first.data(), actual.first.data(), expected.first.size());
  EXPECT_EQ(0, actual.second);
  EXPECT_EQ(expected.second, actual.second);
}

TEST_F(ValidIfTest, AllNull)
{
  auto iter     = cudf::detail::make_counting_transform_iterator(0, all_null{});
  auto expected = cudf::test::detail::make_null_mask(iter, iter + 10000);
  auto actual   = cudf::detail::valid_if(cuda::counting_iterator<cudf::size_type>{0},
                                       cuda::counting_iterator<cudf::size_type>{10000},
                                       all_null{},
                                       cudf::get_default_stream(),
                                       cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.first.data(), actual.first.data(), expected.first.size());
  EXPECT_EQ(10000, actual.second);
  EXPECT_EQ(expected.second, actual.second);
}
