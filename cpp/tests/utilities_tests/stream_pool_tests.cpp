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

std::vector<cudaStream_t> get_streams_from_pool(std::size_t count)
{
  auto const streams = cudf::detail::thread_cuda_stream_pool().get_streams(count);
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
  auto constexpr count = 256;

  auto const streams = get_streams_from_pool(count);
  EXPECT_EQ(streams.size(), count);

  auto const unique = std::unordered_set<cudaStream_t>(streams.begin(), streams.end());
  EXPECT_LT(unique.size(), count);
}

TEST_F(StreamPoolTest, PoolIsReusedAfterThreadExits)
{
  std::unordered_set<cudaStream_t> first_thread_streams;
  std::thread first([&] {
    auto const streams = get_streams_from_pool(4);
    first_thread_streams.insert(streams.begin(), streams.end());
  });
  first.join();

  // Requesting more streams than the pool can hold cycles through all of them, so a thread that
  // adopted its predecessor's retired pool observes every stream that predecessor used. One that
  // created a fresh pool would observe entirely different streams.
  std::unordered_set<cudaStream_t> second_thread_streams;
  std::thread second([&] {
    auto const streams = get_streams_from_pool(256);
    second_thread_streams.insert(streams.begin(), streams.end());
  });
  second.join();

  EXPECT_FALSE(first_thread_streams.empty());
  EXPECT_TRUE(std::all_of(first_thread_streams.begin(),
                          first_thread_streams.end(),
                          [&](auto stream) { return second_thread_streams.contains(stream); }));
}
