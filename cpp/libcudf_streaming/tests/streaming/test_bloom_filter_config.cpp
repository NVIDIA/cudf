/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>

#include <cudf_streaming/bloom_filter.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

TEST(BloomFilterTest, AlignsStorageSize)
{
  EXPECT_EQ(cudf_streaming::bloom_filter::aligned_size(31), 0);
  EXPECT_EQ(cudf_streaming::bloom_filter::aligned_size(32), 32);
  EXPECT_EQ(cudf_streaming::bloom_filter::aligned_size(65), 64);
}

TEST(BloomFilterTest, RequiresAlignedStorageSize)
{
  auto make_filter = [](std::size_t filter_size) {
    return cudf_streaming::bloom_filter{std::shared_ptr<rapidsmpf::streaming::Context>{},
                                        std::shared_ptr<rapidsmpf::Communicator>{},
                                        0,
                                        filter_size};
  };

  EXPECT_THROW(make_filter(0), std::logic_error);
  EXPECT_THROW(make_filter(65), std::logic_error);
  EXPECT_THROW(make_filter(cudf_streaming::bloom_filter::aligned_size(
                 std::numeric_limits<std::size_t>::max())),
               std::logic_error);
  EXPECT_NO_THROW(make_filter(64));
}

}  // namespace
