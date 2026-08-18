/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>

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

std::vector<cudaStream_t> values_of(std::vector<cuda::stream_ref> const& streams)
{
  auto values = std::vector<cudaStream_t>{};
  std::transform(streams.begin(), streams.end(), std::back_inserter(values), [](auto stream) {
    return stream.get();
  });
  return values;
}

std::vector<cudaStream_t> fork_and_collect(std::size_t count)
{
  auto const streams = cudf::detail::fork_streams(cudf::get_default_stream(), count);
  auto const values  = values_of(streams);
  cudf::detail::join_streams(streams, cudf::get_default_stream());
  return values;
}

}  // namespace

TEST_F(StreamPoolTest, ConcurrentThreadsGetDistinctStreams)
{
  auto constexpr num_forks = 20;
  auto const pool_size     = cudf::detail::global_cuda_stream_pool().get_stream_pool_size();
  auto const num_streams   = std::min<std::size_t>(8, pool_size);

  // Both threads fork repeatedly so that a shared round-robin counter would be very likely to
  // hand the same stream to both of them.
  auto collect = [num_streams](std::unordered_set<cudaStream_t>& out, std::latch& ready) {
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

  // Disjointness is only guaranteed while the two requests together fit in the pool; beyond that
  // the rotation wraps, as `get_streams` documents.
  auto const pool_size = cudf::detail::global_cuda_stream_pool().get_stream_pool_size();
  if (pool_size < outer_count + inner_count) {
    GTEST_SKIP() << "Pool of " << pool_size << " streams is too small to fork " << outer_count
                 << " and then " << inner_count;
  }

  std::vector<cudaStream_t> outer;
  std::vector<cudaStream_t> inner;

  std::thread worker([&]() {
    auto const outer_streams = cudf::detail::fork_streams(cudf::get_default_stream(), outer_count);
    // Fork again while the outer streams are still in use, as decompression does while its caller
    // is working on forked streams of its own.
    inner = fork_and_collect(inner_count);
    outer = values_of(outer_streams);
    cudf::detail::join_streams(outer_streams, cudf::get_default_stream());
  });
  worker.join();

  EXPECT_TRUE(std::none_of(inner.begin(), inner.end(), [&](auto stream) {
    return std::find(outer.begin(), outer.end(), stream) != outer.end();
  }));
}

TEST_F(StreamPoolTest, PoolGrowsToBoundedHighWaterMark)
{
  auto const pool_size = cudf::detail::global_cuda_stream_pool().get_stream_pool_size();
  if (pool_size < 4) {
    GTEST_SKIP() << "Pool of " << pool_size << " streams is too small to cycle";
  }

  // Request half the pool at a time so that consecutive requests can be disjoint, and fork enough
  // times to cycle any pool the worker might adopt. Pools are recycled between threads, so the size
  // of the one this thread is handed is not known here.
  auto const count = pool_size / 2;
  auto const forks = 2 * pool_size / count + 2;

  std::unordered_set<cudaStream_t> all_streams;
  std::vector<std::vector<cudaStream_t>> per_fork;

  std::thread worker([&]() {
    for (auto fork = 0u; fork < forks; fork++) {
      per_fork.push_back(fork_and_collect(count));
      all_streams.insert(per_fork.back().begin(), per_fork.back().end());
    }
  });
  worker.join();

  // Consecutive requests are served from different streams, ...
  for (auto fork = 1u; fork < forks; fork++) {
    auto const& previous = per_fork[fork - 1];
    EXPECT_TRUE(std::none_of(per_fork[fork].begin(), per_fork[fork].end(), [&](auto stream) {
      return std::find(previous.begin(), previous.end(), stream) != previous.end();
    }));
  }

  // ... but the pool stops growing: once it has been cycled, later requests only return streams
  // that were handed out before.
  auto seen = std::unordered_set<cudaStream_t>{};
  for (auto fork = 0u; fork + 1 < forks; fork++) {
    seen.insert(per_fork[fork].begin(), per_fork[fork].end());
  }
  EXPECT_TRUE(std::all_of(per_fork.back().begin(), per_fork.back().end(), [&](auto stream) {
    return seen.contains(stream);
  }));
  EXPECT_LE(all_streams.size(), pool_size);
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
  auto const pool_size       = cudf::detail::global_cuda_stream_pool().get_stream_pool_size();

  auto collect = [](std::unordered_set<cudaStream_t>& out, std::size_t forks) {
    for (auto fork = 0u; fork < forks; fork++) {
      auto const streams = fork_and_collect(num_streams);
      out.insert(streams.begin(), streams.end());
    }
  };

  std::unordered_set<cudaStream_t> first_thread_streams;
  std::thread first(collect, std::ref(first_thread_streams), 2);
  first.join();

  // The second thread cycles the whole pool, so if it adopted the retired pool it observes
  // everything its predecessor used. A thread that created a fresh pool would observe entirely
  // different streams, since the rotation offset carries over but the streams themselves would not.
  std::unordered_set<cudaStream_t> second_thread_streams;
  std::thread second(collect, std::ref(second_thread_streams), 2 * pool_size / num_streams + 2);
  second.join();

  EXPECT_FALSE(first_thread_streams.empty());
  EXPECT_TRUE(std::all_of(first_thread_streams.begin(),
                          first_thread_streams.end(),
                          [&](auto stream) { return second_thread_streams.contains(stream); }));
}

TEST_F(StreamPoolTest, EachDeviceHasItsOwnPool)
{
  auto num_devices = 0;
  CUDF_CUDA_TRY(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) { GTEST_SKIP() << "Requires more than one visible device"; }

  auto constexpr count = 2;
  auto const first     = values_of(cudf::detail::global_cuda_stream_pool().get_streams(count));

  auto second = std::vector<cudaStream_t>{};
  {
    // Streams are bound to the device that was current when they were created, so the pool for
    // another device must not hand out the streams of this one.
    rmm::cuda_set_device_raii const device{rmm::cuda_device_id{1}};
    second = values_of(cudf::detail::global_cuda_stream_pool().get_streams(count));
  }

  EXPECT_EQ(first.size(), count);
  EXPECT_EQ(second.size(), count);
  EXPECT_TRUE(std::none_of(second.begin(), second.end(), [&](auto stream) {
    return std::find(first.begin(), first.end(), stream) != first.end();
  }));
}
