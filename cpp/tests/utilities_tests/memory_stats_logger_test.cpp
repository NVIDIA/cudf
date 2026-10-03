/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file memory_stats_logger_test.cpp
 * @brief Verify memory-statistics resets preserve the captured resource and restart peak tracking.
 *
 * Checks rejection of live allocations, including zero-byte allocations, and upstream restoration.
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cstddef>

TEST(MemoryStatsLoggerTest, ResetPreservesCapturedResourceAndStartsFreshMeasurement)
{
  auto const upstream = cudf::get_current_device_resource_ref();
  auto const stream   = cudf::test::get_default_stream();

  {
    cudf::memory_stats_logger logger;
    auto captured_mr = cudf::get_current_device_resource_ref();
    EXPECT_FALSE(captured_mr == upstream);
    EXPECT_EQ(logger.peak_memory_usage(), 0);

    constexpr std::size_t large_bytes = 4096;
    constexpr std::size_t small_bytes = 1024;
    {
      rmm::device_buffer allocation{large_bytes, stream, captured_mr};
      stream.sync();
      EXPECT_EQ(logger.peak_memory_usage(), large_bytes);

      EXPECT_THROW(logger.reset_counters(), cudf::logic_error);
      EXPECT_EQ(logger.peak_memory_usage(), large_bytes);
      EXPECT_TRUE(cudf::get_current_device_resource_ref() == captured_mr);
    }
    stream.sync();
    EXPECT_EQ(logger.peak_memory_usage(), large_bytes);

    ASSERT_NO_THROW(logger.reset_counters());
    EXPECT_EQ(logger.peak_memory_usage(), 0);
    EXPECT_TRUE(cudf::get_current_device_resource_ref() == captured_mr);

    {
      rmm::device_buffer allocation{small_bytes, stream, captured_mr};
      stream.sync();
      EXPECT_EQ(logger.peak_memory_usage(), small_bytes);
      EXPECT_LT(logger.peak_memory_usage(), large_bytes);
    }
    stream.sync();
    EXPECT_EQ(logger.peak_memory_usage(), small_bytes);
    EXPECT_NO_THROW(logger.reset_counters());
    EXPECT_EQ(logger.peak_memory_usage(), 0);
  }

  EXPECT_TRUE(cudf::get_current_device_resource_ref() == upstream);
}

TEST(MemoryStatsLoggerTest, ResetRejectsLiveZeroByteAllocation)
{
  auto const upstream = cudf::get_current_device_resource_ref();
  auto const stream   = cudf::test::get_default_stream();

  {
    cudf::memory_stats_logger logger;
    auto captured_mr = cudf::get_current_device_resource_ref();

    // Bypass device_buffer, which skips zero-byte allocations, to exercise the allocation counter.
    auto* allocation = captured_mr.allocate(stream, 0, alignof(std::max_align_t));
    stream.sync();
    EXPECT_EQ(logger.peak_memory_usage(), 0);
    EXPECT_THROW(logger.reset_counters(), cudf::logic_error);
    EXPECT_EQ(logger.peak_memory_usage(), 0);

    captured_mr.deallocate(stream, allocation, 0, alignof(std::max_align_t));
    stream.sync();
    EXPECT_NO_THROW(logger.reset_counters());
    EXPECT_EQ(logger.peak_memory_usage(), 0);
  }

  EXPECT_TRUE(cudf::get_current_device_resource_ref() == upstream);
}
