/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/reduction/bloom_filter.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cudf_streaming/detail/device_bloom_filter.hpp>

#include <rmm/device_scalar.hpp>

#include <cuco/bloom_filter_ref.cuh>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

using policy_type = cudf::arrow_filter_policy<cuco::identity_hash<std::uint64_t>>;

__global__ void block_index_kernel(std::uint32_t upper_hash,
                                   std::size_t num_blocks,
                                   std::uint32_t* result)
{
  *result = policy_type{}.block_index(upper_hash, cuco::extent<std::size_t>{num_blocks});
}

TEST(BloomFilterPolicyTest, UsesBlocksBeyondFormerArrowLimit)
{
  constexpr auto arrow_max_blocks = std::size_t{4'194'304};
  constexpr auto num_blocks       = arrow_max_blocks + 1;
  constexpr auto upper_hash       = std::numeric_limits<std::uint32_t>::max();
  auto const stream               = cudf::get_default_stream();
  rmm::device_scalar<std::uint32_t> index{0, stream};

  block_index_kernel<<<1, 1, 0, stream.value()>>>(upper_hash, num_blocks, index.data());
  CUDF_CHECK_CUDA(stream.value());

  EXPECT_EQ(index.value(stream), arrow_max_blocks);
}

TEST(DeviceBloomFilterTest, RejectsStorageBeyondPolicyLimit)
{
  using filter_ref_type     = cuco::bloom_filter_ref<std::uint64_t,
                                                     cuco::extent<std::size_t>,
                                                     cuco::thread_scope_device,
                                                     policy_type>;
  constexpr auto block_size = sizeof(filter_ref_type::filter_block_type);
  constexpr auto too_large  = (policy_type::max_filter_blocks + std::size_t{1}) * block_size;
  auto const stream         = cudf::get_default_stream();

  EXPECT_THROW(cudf_streaming::detail::device_bloom_filter::storage(
                 too_large, stream, cudf::get_current_device_resource_ref()),
               std::logic_error);
}

TEST(DeviceBloomFilterTest, RequiresAlignedStorageSize)
{
  constexpr auto unaligned_size = std::size_t{65};
  constexpr auto aligned_size   = std::size_t{64};
  auto const stream             = cudf::get_default_stream();

  EXPECT_THROW(cudf_streaming::detail::device_bloom_filter::storage(
                 unaligned_size, stream, cudf::get_current_device_resource_ref()),
               std::logic_error);

  auto storage = cudf_streaming::detail::device_bloom_filter::storage(
    aligned_size, stream, cudf::get_current_device_resource_ref());

  EXPECT_EQ(storage->size(), aligned_size);

  auto const filter = cudf_streaming::detail::device_bloom_filter{aligned_size, 0, storage->data()};
  EXPECT_EQ(filter.size(), aligned_size);
}

}  // namespace
