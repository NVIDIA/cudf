/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <cuda/iterator>

#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>

template <typename T>
struct BatchedMemcpyTest : public cudf::test::BaseFixture {};

TEST(BatchedMemcpyTest, BasicTest)
{
  using T1 = int64_t;

  // Device init
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Buffer lengths (in number of elements)
  std::vector<std::size_t> const h_lens{
    50000, 4, 1000, 0, 250000, 1, 100, 8000, 0, 1, 100, 1000, 10000, 100000, 0, 1, 100000};

  // Total number of buffers
  auto const num_buffs = h_lens.size();

  // Exclusive sum of buffer lengths for pointers
  std::vector<std::size_t> h_lens_excl_sum(num_buffs);
  std::exclusive_scan(h_lens.begin(), h_lens.end(), h_lens_excl_sum.begin(), 0);

  // Corresponding buffer sizes (in bytes)
  std::vector<std::size_t> h_sizes_bytes;
  h_sizes_bytes.reserve(num_buffs);
  std::transform(
    h_lens.cbegin(), h_lens.cend(), std::back_inserter(h_sizes_bytes), [&](auto& size) {
      return size * sizeof(T1);
    });

  // Initialize random engine
  auto constexpr seed = 0xcead;
  std::mt19937 engine{seed};
  using uniform_distribution =
    typename std::conditional_t<std::is_same_v<T1, bool>,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point_v<T1>,
                                                   std::uniform_real_distribution<T1>,
                                                   std::uniform_int_distribution<T1>>>;
  uniform_distribution dist{};

  // Generate a src vector of random data vectors
  std::vector<std::vector<T1>> h_sources;
  h_sources.reserve(num_buffs);
  std::transform(h_lens.begin(), h_lens.end(), std::back_inserter(h_sources), [&](auto size) {
    std::vector<T1> data(size);
    std::generate_n(data.begin(), size, [&]() { return T1{dist(engine)}; });
    return data;
  });
  // Copy the vectors to device
  std::vector<rmm::device_uvector<T1>> h_device_vecs;
  h_device_vecs.reserve(h_sources.size());
  std::transform(
    h_sources.begin(), h_sources.end(), std::back_inserter(h_device_vecs), [stream, mr](auto& vec) {
      return cudf::detail::make_device_uvector_async(vec, stream, mr);
    });
  // Pointers to the source vectors
  std::vector<T1*> h_src_ptrs;
  h_src_ptrs.reserve(h_sources.size());
  std::transform(
    h_device_vecs.begin(), h_device_vecs.end(), std::back_inserter(h_src_ptrs), [](auto& vec) {
      return static_cast<T1*>(vec.data());
    });
  // Copy the source data pointers to device
  auto d_src_ptrs = cudf::detail::make_device_uvector_async(h_src_ptrs, stream, mr);

  // Total number of elements in all buffers
  auto const total_buff_len = std::accumulate(h_lens.cbegin(), h_lens.cend(), 0);

  // Create one giant buffer for destination
  auto d_dst_data = cudf::detail::make_zeroed_device_uvector_async<T1>(total_buff_len, stream, mr);
  // Pointers to destination buffers within the giant destination buffer
  std::vector<T1*> h_dst_ptrs(num_buffs);
  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{num_buffs},
                [&](auto i) { return h_dst_ptrs[i] = d_dst_data.data() + h_lens_excl_sum[i]; });
  // Copy destination data pointers to device
  auto d_dst_ptrs = cudf::detail::make_device_uvector_async(h_dst_ptrs, stream, mr);

  // Copy buffer size iterators (in bytes) to device
  auto d_sizes_bytes = cudf::detail::make_device_uvector_async(h_sizes_bytes, stream, mr);

  // Run the batched memcpy
  cudf::detail::batched_memcpy_async(
    d_src_ptrs.begin(), d_dst_ptrs.begin(), d_sizes_bytes.begin(), num_buffs, stream);

  // Expected giant destination buffer after the memcpy
  std::vector<T1> expected_buffer;
  expected_buffer.reserve(total_buff_len);
  std::for_each(h_sources.cbegin(), h_sources.cend(), [&expected_buffer](auto& source) {
    expected_buffer.insert(expected_buffer.end(), source.begin(), source.end());
  });

  // Copy over the result destination buffer to host and synchronize the stream
  auto result_dst_buffer =
    cudf::detail::make_std_vector<T1>(cudf::device_span<T1>(d_dst_data), stream);

  // Check if both vectors are equal
  EXPECT_TRUE(
    std::equal(expected_buffer.begin(), expected_buffer.end(), result_dst_buffer.begin()));
}
