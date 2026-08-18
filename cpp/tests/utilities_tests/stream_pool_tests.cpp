/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>

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
  auto const pool_size     = cudf::detail::thread_cuda_stream_pool().get_stream_pool_size();
  auto const num_streams   = std::min<std::size_t>(8, pool_size);

  // Both threads fork repeatedly so that a shared round-robin counter would be very likely to
  // hand the same stream to both of them. The latch keeps them alive at the same time; otherwise
  // the second thread could adopt the pool the first one retired and pass trivially.
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

TEST_F(StreamPoolTest, RequestLargerThanPoolRepeatsStreams)
{
  auto const pool_size = cudf::detail::thread_cuda_stream_pool().get_stream_pool_size();
  auto const count     = pool_size + 4;

  auto const streams = fork_and_collect(count);
  EXPECT_EQ(streams.size(), count);

  auto const unique = std::unordered_set<cudaStream_t>(streams.begin(), streams.end());
  EXPECT_EQ(unique.size(), pool_size);
}

TEST_F(StreamPoolTest, PoolIsReusedAfterThreadExits)
{
  auto constexpr num_streams = 4;
  auto const pool_size       = cudf::detail::thread_cuda_stream_pool().get_stream_pool_size();

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
