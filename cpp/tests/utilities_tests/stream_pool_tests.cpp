/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/stream_pool.hpp>

#include <cuda/stream_ref>

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

// A thread has no use for more than 64 streams, so a request of this size exceeds the maximum of
// any pool worth configuring and cycles through every stream in it.
auto constexpr more_streams_than_any_pool = 128;

std::vector<cudaStream_t> get_streams_from_pool(std::size_t count)
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

  // Both threads request repeatedly so that a shared round-robin counter would be very likely to
  // hand the same stream to both of them. The latch makes them overlap, which is the case under
  // test: a thread that started after the other exited would correctly adopt the retired pool and
  // observe the very streams this test requires to be disjoint.
  auto collect = [](std::unordered_set<cudaStream_t>& out, std::latch& ready) {
    ready.arrive_and_wait();
    for (auto request = 0; request < num_requests; request++) {
      auto const streams = get_streams_from_pool(num_streams);
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
  // The pool is capped, so a request this large is served by repeating streams instead of creating
  // one stream per request.
  auto const streams = get_streams_from_pool(more_streams_than_any_pool);
  EXPECT_EQ(streams.size(), more_streams_than_any_pool);

  auto const unique = std::unordered_set<cudaStream_t>(streams.begin(), streams.end());
  EXPECT_LT(unique.size(), more_streams_than_any_pool);
}

TEST_F(StreamPoolTest, PoolIsReusedAfterThreadExits)
{
  // A request larger than max pool size to get all streams in the pool
  std::unordered_set<cudaStream_t> first_thread_streams;
  std::thread([&] {
    auto const streams = get_streams_from_pool(more_streams_than_any_pool);
    first_thread_streams.insert(streams.begin(), streams.end());
  }).join();

  std::vector<cudaStream_t> second_thread_streams;
  std::thread([&] { second_thread_streams = get_streams_from_pool(4); }).join();

  EXPECT_FALSE(second_thread_streams.empty());
  // A thread that adopted the retired pool can only be handed streams from it
  EXPECT_TRUE(std::all_of(second_thread_streams.begin(),
                          second_thread_streams.end(),
                          [&](auto stream) { return first_thread_streams.contains(stream); }));
}
