/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <algorithm>
#include <iterator>
#include <latch>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

class StreamPoolTest : public cudf::test::BaseFixture {};

namespace {

std::vector<cudaStream_t> fork_and_collect(std::size_t count)
{
  auto const streams = cudf::detail::fork_streams(cudf::get_default_stream(), count);
  auto values        = std::vector<cudaStream_t>{};
  std::transform(streams.begin(), streams.end(), std::back_inserter(values), [](auto stream) {
    return stream.value();
  });
  cudf::detail::join_streams(streams, cudf::get_default_stream());
  return values;
}

}  // namespace

TEST_F(StreamPoolTest, ConcurrentThreadsGetDistinctStreams)
{
  auto constexpr num_streams = 8;
  auto constexpr num_forks   = 20;

  // Both threads fork repeatedly so that a shared round-robin counter would be very likely to
  // hand the same stream to both of them.
  auto collect = [](std::unordered_set<cudaStream_t>& out, std::latch& ready) {
    ready.arrive_and_wait();
    for (auto fork = 0; fork < num_forks; fork++) {
      auto const streams = fork_and_collect(num_streams);
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

  auto const pool_size = cudf::detail::global_cuda_stream_pool().get_stream_pool_size();
  EXPECT_GE(first_streams.size(), num_streams);
  EXPECT_LE(first_streams.size(), pool_size);
  EXPECT_TRUE(std::none_of(first_streams.begin(), first_streams.end(), [&](auto stream) {
    return second_streams.contains(stream);
  }));
}

TEST_F(StreamPoolTest, NestedForkDoesNotReuseOuterStreams)
{
  auto constexpr outer_count = 4;
  auto constexpr inner_count = 2;

  std::vector<cudaStream_t> outer;
  std::vector<cudaStream_t> inner;

  std::thread worker([&]() {
    auto const outer_streams = cudf::detail::fork_streams(cudf::get_default_stream(), outer_count);
    // Fork again while the outer streams are still in use, as decompression does while its caller
    // is working on forked streams of its own.
    inner = fork_and_collect(inner_count);
    std::transform(
      outer_streams.begin(), outer_streams.end(), std::back_inserter(outer), [](auto stream) {
        return stream.value();
      });
    cudf::detail::join_streams(outer_streams, cudf::get_default_stream());
  });
  worker.join();

  EXPECT_TRUE(std::none_of(inner.begin(), inner.end(), [&](auto stream) {
    return std::find(outer.begin(), outer.end(), stream) != outer.end();
  }));
}

TEST_F(StreamPoolTest, PoolGrowsToBoundedHighWaterMark)
{
  auto constexpr count = 4;
  auto constexpr forks = 10;

  std::unordered_set<cudaStream_t> all_streams;
  std::vector<std::vector<cudaStream_t>> per_fork;

  std::thread worker([&]() {
    for (auto fork = 0; fork < forks; fork++) {
      per_fork.push_back(fork_and_collect(count));
      all_streams.insert(per_fork.back().begin(), per_fork.back().end());
    }
  });
  worker.join();

  // Consecutive requests are served from different streams, ...
  for (auto fork = 1; fork < forks; fork++) {
    auto const& previous = per_fork[fork - 1];
    EXPECT_TRUE(std::none_of(per_fork[fork].begin(), per_fork[fork].end(), [&](auto stream) {
      return std::find(previous.begin(), previous.end(), stream) != previous.end();
    }));
  }

  // ... but repeated requests cycle through the streams already created rather than making more.
  auto early = std::unordered_set<cudaStream_t>{};
  for (auto fork = 0; fork < forks / 2; fork++) {
    early.insert(per_fork[fork].begin(), per_fork[fork].end());
  }
  for (auto fork = forks / 2; fork < forks; fork++) {
    EXPECT_TRUE(std::all_of(per_fork[fork].begin(), per_fork[fork].end(), [&](auto stream) {
      return early.contains(stream);
    }));
  }
  EXPECT_LE(all_streams.size(), cudf::detail::global_cuda_stream_pool().get_stream_pool_size());
}

TEST_F(StreamPoolTest, RequestLargerThanPoolRepeatsStreams)
{
  auto const pool_size = cudf::detail::global_cuda_stream_pool().get_stream_pool_size();
  auto const count     = pool_size + 4;

  auto const streams = fork_and_collect(count);
  EXPECT_EQ(streams.size(), count);

  auto const unique = std::unordered_set<cudaStream_t>(streams.begin(), streams.end());
  EXPECT_EQ(unique.size(), pool_size);
}

TEST_F(StreamPoolTest, PoolIsReusedAfterThreadExits)
{
  auto constexpr num_streams = 4;

  // Each thread cycles through its whole pool, so an adopting thread observes exactly the streams
  // its predecessor used. A thread that created a fresh pool would observe entirely new ones.
  auto cycle_pool = [](std::unordered_set<cudaStream_t>& out) {
    for (auto fork = 0; fork < 4; fork++) {
      auto const streams = fork_and_collect(num_streams);
      out.insert(streams.begin(), streams.end());
    }
  };

  std::unordered_set<cudaStream_t> first_thread_streams;
  std::thread first(cycle_pool, std::ref(first_thread_streams));
  first.join();

  std::unordered_set<cudaStream_t> second_thread_streams;
  std::thread second(cycle_pool, std::ref(second_thread_streams));
  second.join();

  EXPECT_EQ(first_thread_streams, second_thread_streams);
}
