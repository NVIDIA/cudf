/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>

// cuco headers have sign-conversion issues; suppress for the host compiler
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <cudf/reduction/bloom_filter.cuh>

#include <cuco/bloom_filter_ref.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_streaming/detail/device_bloom_filter.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_transform.cuh>
#include <cuda/std/tuple>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>

namespace cudf_streaming::detail {

namespace {
using key_type = std::uint64_t;

using bloom_filter_policy_type =
  cudf::arrow_bloom_filter_policy<key_type, cuco::identity_hash<key_type>>;
using bloom_filter_ref_type = cuco::bloom_filter_ref<key_type,
                                                     cuco::extent<std::size_t>,
                                                     cuco::thread_scope_device,
                                                     bloom_filter_policy_type>;
using storage_type          = bloom_filter_ref_type::filter_block_type;

std::size_t num_blocks(std::size_t filter_size)
{
  RAPIDSMPF_EXPECTS(filter_size >= sizeof(storage_type),
                    "Bloom filter storage must contain at least one filter block");
  RAPIDSMPF_EXPECTS(filter_size == device_bloom_filter::aligned_size(filter_size),
                    "Bloom filter storage size must be a multiple of the filter block size");
  auto const blocks = filter_size / sizeof(storage_type);
  RAPIDSMPF_EXPECTS(blocks <= bloom_filter_policy_type::max_filter_blocks,
                    "Bloom filter storage exceeds the maximum size supported by its policy");
  return blocks;
}

}  // namespace

device_bloom_filter::device_bloom_filter(std::size_t filter_size, std::uint64_t seed, void* storage)
  : num_blocks_{num_blocks(filter_size)}, seed_{seed}, storage_{storage}
{
  RAPIDSMPF_EXPECTS(
    reinterpret_cast<std::uintptr_t>(storage_) % std::alignment_of_v<storage_type> == 0,
    "Allocation for bloom filter is not aligned.");
}

device_bloom_filter const device_bloom_filter::view(std::size_t filter_size,
                                                    std::uint64_t seed,
                                                    void const* storage)
{
  // const-cast is safe because the returned object is also const and therefore can't
  // call methods that throw away constness.
  return device_bloom_filter(filter_size, seed, const_cast<void*>(storage));
}

std::unique_ptr<rmm::device_buffer> device_bloom_filter::storage(std::size_t filter_size,
                                                                 rmm::cuda_stream_view stream,
                                                                 rmm::device_async_resource_ref mr)
{
  return std::make_unique<rmm::device_buffer>(
    num_blocks(filter_size) * sizeof(storage_type), std::alignment_of_v<storage_type>, stream, mr);
}

void device_bloom_filter::add(cudf::table_view const& values_to_hash,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  auto filter_ref = bloom_filter_ref_type{
    static_cast<storage_type*>(storage_), num_blocks_, cuco::thread_scope_device, {}};
  auto hashes    = cudf::hashing::xxhash_64(values_to_hash, seed_, stream, mr);
  auto hash_view = hashes->view();
  RAPIDSMPF_EXPECTS(hash_view.type().id() == cudf::type_to_id<key_type>(),
                    "Hash values do not have correct type");
  filter_ref.add_async(hash_view.begin<key_type>(), hash_view.end<key_type>(), stream);
}

void device_bloom_filter::merge(device_bloom_filter const& other, rmm::cuda_stream_view stream)
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  RAPIDSMPF_EXPECTS(num_blocks_ == other.num_blocks_, "Mismatching number of blocks in filters");
  auto ref_this = bloom_filter_ref_type{
    static_cast<storage_type*>(storage_), num_blocks_, cuco::thread_scope_device, {}};
  auto ref_other = bloom_filter_ref_type{
    static_cast<storage_type*>(other.storage_), num_blocks_, cuco::thread_scope_device, {}};
  ref_this.merge_async(ref_other, stream);
}

rmm::device_uvector<bool> device_bloom_filter::contains(cudf::table_view const& values,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr) const
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  auto filter_ref = bloom_filter_ref_type{
    static_cast<storage_type*>(storage_), num_blocks_, cuco::thread_scope_device, {}};
  auto hashes = cudf::hashing::xxhash_64(values, seed_, stream, mr);
  auto view   = hashes->view();
  rmm::device_uvector<bool> result{static_cast<std::size_t>(view.size()), stream, mr};
  filter_ref.contains_async(view.begin<key_type>(), view.end<key_type>(), result.begin(), stream);
  return result;
}

std::size_t device_bloom_filter::aligned_size(std::size_t size) noexcept
{
  return rmm::align_down(size, std::alignment_of_v<storage_type>);
}

std::size_t device_bloom_filter::max_size() noexcept
{
  return bloom_filter_policy_type::max_filter_blocks * sizeof(storage_type);
}

void* device_bloom_filter::data() noexcept { return storage_; }

void const* device_bloom_filter::data() const noexcept { return storage_; }

std::size_t device_bloom_filter::size() const noexcept
{
  return num_blocks_ * sizeof(storage_type);
}

}  // namespace cudf_streaming::detail
