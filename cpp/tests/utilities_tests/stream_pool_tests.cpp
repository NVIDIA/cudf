/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/stream_pool.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <latch>
#include <thread>
#include <unordered_set>
#include <vector>

class StreamPoolTest : public cudf::test::BaseFixture {};

namespace {

std::vector<cudaStream_t> get_hashable_streams(std::size_t count)
{
  auto const streams = cudf::detail::current_cuda_stream_pool().get_streams(count);
  auto values        = std::vector<cudaStream_t>{};
  std::transform(streams.begin(), streams.end(), std::back_inserter(values), [](auto stream) {
    return stream.get();
  });
  return values;
}

}  // namespace

TEST_F(StreamPoolTest, ConcurrentThreadsGetDistinctStreams)
{
  auto constexpr num_requests = 20;
  auto constexpr num_streams  = 8;

  // Repeated requests make a shared round-robin counter very likely to hand the same stream to both
  // threads. Each thread takes its pool before the latch: a pool is created on first use and
  // returned to the free list when its thread exits, so a thread that made its first request after
  // the other one exited would adopt the retired pool and see the same streams.
  auto collect = [](std::unordered_set<cudaStream_t>& out, std::latch& ready) {
    auto const first = get_hashable_streams(num_streams);
    out.insert(first.begin(), first.end());
    ready.arrive_and_wait();
    for (auto request = 0; request < num_requests; request++) {
      auto const streams = get_hashable_streams(num_streams);
      out.insert(streams.begin(), streams.end());
    }
  };

  std::unordered_set<cudaStream_t> first_streams;
  std::unordered_set<cudaStream_t> second_streams;
  std::latch ready{2};

  std::thread first(collect, std::ref(first_streams), std::ref(ready));
  std::thread second(collect, std::ref(second_streams), std::ref(ready));
  first.join();
  second.join();

  EXPECT_FALSE(first_streams.empty());
  EXPECT_TRUE(std::none_of(first_streams.begin(), first_streams.end(), [&](auto stream) {
    return second_streams.contains(stream);
  }));
}

TEST_F(StreamPoolTest, RequestLargerThanPoolRepeatsStreams)
{
  auto constexpr count = 128;

  auto const streams = get_hashable_streams(count);
  EXPECT_EQ(streams.size(), count);

  // Request this large is served by repeating streams instead of growing the pool
  auto const unique = std::unordered_set<cudaStream_t>(streams.begin(), streams.end());
  EXPECT_LT(unique.size(), count);
}

TEST_F(StreamPoolTest, PoolIsReusedAfterThreadExits)
{
  // A request larger than max pool size to get all streams in the pool
  std::unordered_set<cudaStream_t> first_thread_streams;
  std::thread([&] {
    auto const streams = get_hashable_streams(128);
    first_thread_streams.insert(streams.begin(), streams.end());
  }).join();

  std::vector<cudaStream_t> second_thread_streams;
  std::thread([&] { second_thread_streams = get_hashable_streams(4); }).join();

  EXPECT_FALSE(second_thread_streams.empty());
  // A thread that adopted the retired pool can only be handed streams from it
  EXPECT_TRUE(std::all_of(second_thread_streams.begin(),
                          second_thread_streams.end(),
                          [&](auto stream) { return first_thread_streams.contains(stream); }));
}
