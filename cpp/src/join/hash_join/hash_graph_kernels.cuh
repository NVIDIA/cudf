/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "hash_graph.cuh"

#include <cudf/detail/utilities/dispatchers.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/std/algorithm.max.h>
#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <cuda/stream>

#include <utility>

namespace cudf::detail {

constexpr thread_index_type hash_graph_block_size = 256;

template <typename Hasher>
CUDF_KERNEL void hash_graph_build_count_kernel(size_type num_rows,
                                               bitmask_type const* valid_rows,
                                               size_type* counts,
                                               hash_value_type* hashes,
                                               cuda::std::uint32_t num_buckets,
                                               Hasher hasher)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    if (valid_rows != nullptr && !cudf::bit_is_set(valid_rows, index)) { continue; }
    auto const hash = hasher(index);
    if (hashes != nullptr) { hashes[index] = hash; }
    cuda::atomic_ref<size_type, cuda::thread_scope_device>{
      counts[hash_graph_bucket(hash, num_buckets)]}
      .fetch_add(1, cuda::memory_order_relaxed);
  }
}

template <typename Hasher>
CUDF_KERNEL void hash_graph_build_fill_kernel(size_type num_rows,
                                              bitmask_type const* valid_rows,
                                              size_type* offsets,
                                              hash_graph_record* records,
                                              hash_value_type const* hashes,
                                              cuda::std::uint32_t num_buckets,
                                              Hasher hasher)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    if (valid_rows != nullptr && !cudf::bit_is_set(valid_rows, index)) { continue; }
    auto const hash = hashes != nullptr ? hashes[index] : hasher(index);
    auto const end =
      cuda::atomic_ref<size_type, cuda::thread_scope_device>{
        offsets[hash_graph_bucket(hash, num_buckets)]}
        .fetch_sub(1, cuda::memory_order_relaxed);
    records[end - 1] = {hash, index};
  }
}

template <bool IsOuter, bool ComputeSum, typename Equal, typename Hasher>
CUDF_KERNEL void hash_graph_probe_count_kernel(size_type num_rows,
                                               bitmask_type const* valid_rows,
                                               hash_graph_probe_state* probe_states,
                                               size_type* match_counts,
                                               cuda::std::uint64_t* total_matches,
                                               cuda::std::uint32_t* matched_rows,
                                               cuda::std::uint64_t* matched_build_rows,
                                               hash_graph_ref graph,
                                               Equal equal,
                                               Hasher hasher)
{
  auto const stride                    = grid_1d::grid_stride();
  cuda::std::uint64_t thread_matches   = 0;
  cuda::std::uint64_t thread_new_flags = 0;
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    auto count       = size_type{0};
    auto state       = hash_graph_no_match;
    if (valid_rows == nullptr || cudf::bit_is_set(valid_rows, index)) {
      auto const key    = hash_graph_record{hasher(index), index};
      auto const bucket = hash_graph_bucket(key.first, graph.num_buckets);
      auto const end    = graph.offsets[bucket + 1];
      auto other_key    = false;
      for (auto position = graph.offsets[bucket]; position < end; ++position) {
        auto const record = graph.records[position];
        if (record.first != key.first) { continue; }
        if (!equal(key, record)) {
          other_key = true;
          continue;
        }
        if (count == 0) { state = static_cast<hash_graph_probe_state>(position); }
        ++count;
        // Only full joins consume the matched-row tally, and `matched_rows` is null for every
        // other kind.
        if constexpr (IsOuter) {
          if (matched_rows != nullptr) {
            auto expected = cuda::std::uint32_t{0};
            thread_new_flags +=
              cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{
                matched_rows[record.second]}
                .compare_exchange_strong(
                  expected, cuda::std::uint32_t{1}, cuda::memory_order_relaxed);
          }
        }
      }
      if (count != 0 && other_key) { state |= hash_graph_compare_rows; }
    }
    if (probe_states != nullptr) { probe_states[index] = state; }
    auto const match_count = IsOuter ? cuda::std::max(count, size_type{1}) : count;
    if (match_counts != nullptr) { match_counts[index] = match_count; }
    if constexpr (ComputeSum) { thread_matches += static_cast<cuda::std::uint64_t>(match_count); }
  }

  if constexpr (IsOuter) {
    if (thread_new_flags != 0) {
      cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device>{*matched_build_rows}
        .fetch_add(thread_new_flags, cuda::memory_order_relaxed);
    }
  }

  if constexpr (ComputeSum) {
    using block_reduce = cub::BlockReduce<cuda::std::uint64_t, hash_graph_block_size>;
    __shared__ typename block_reduce::TempStorage temp_storage;
    auto const block_matches = block_reduce(temp_storage).Sum(thread_matches);
    if (threadIdx.x == 0) {
      cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device>{*total_matches}.fetch_add(
        block_matches, cuda::memory_order_relaxed);
    }
  }
}

template <bool IsOuter, typename Equal>
CUDF_KERNEL void hash_graph_retrieve_kernel(size_type num_rows,
                                            hash_graph_probe_state const* probe_states,
                                            cuda::std::int64_t const* output_offsets,
                                            hash_graph_record const* records,
                                            size_type left_index_offset,
                                            size_type* left_indices,
                                            size_type* right_indices,
                                            Equal equal)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index      = static_cast<size_type>(row);
    auto output           = output_offsets[index];
    auto const output_end = output_offsets[index + 1];
    if (output == output_end) { continue; }
    auto const state = probe_states[index];
    if constexpr (IsOuter) {
      if (state == hash_graph_no_match) {
        left_indices[output]  = index + left_index_offset;
        right_indices[output] = JoinNoMatch;
        continue;
      }
    }
    // The matches of a key are interleaved with the other keys of its bucket. Resume at the
    // first match and take the records with the same hash until every counted match is
    // emitted. Rows are compared again only if the count pass saw another key with that hash.
    auto position           = static_cast<size_type>(state & ~hash_graph_compare_rows);
    auto const compare_rows = (state & hash_graph_compare_rows) != 0;
    auto const key          = hash_graph_record{records[position].first, index};
    for (; output < output_end; ++position) {
      auto const record = records[position];
      if (record.first == key.first && (!compare_rows || equal(key, record))) {
        left_indices[output]  = index + left_index_offset;
        right_indices[output] = record.second;
        ++output;
      }
    }
  }
}

template <typename Hasher>
void launch_hash_graph_build_count_kernel(size_type num_rows,
                                          bitmask_type const* valid_rows,
                                          size_type* counts,
                                          hash_value_type* hashes,
                                          cuda::std::uint32_t num_buckets,
                                          Hasher hasher,
                                          cuda::stream_ref stream)
{
  auto const config = grid_1d{num_rows, hash_graph_block_size};
  hash_graph_build_count_kernel<<<config.num_blocks,
                                  config.num_threads_per_block,
                                  0,
                                  stream.get()>>>(
    num_rows, valid_rows, counts, hashes, num_buckets, hasher);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename Hasher>
void launch_hash_graph_build_fill_kernel(size_type num_rows,
                                         bitmask_type const* valid_rows,
                                         size_type* offsets,
                                         hash_graph_record* records,
                                         hash_value_type const* hashes,
                                         cuda::std::uint32_t num_buckets,
                                         Hasher hasher,
                                         cuda::stream_ref stream)
{
  auto const config = grid_1d{num_rows, hash_graph_block_size};
  hash_graph_build_fill_kernel<<<config.num_blocks,
                                 config.num_threads_per_block,
                                 0,
                                 stream.get()>>>(
    num_rows, valid_rows, offsets, records, hashes, num_buckets, hasher);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <bool IsOuter, typename Equal, typename Hasher>
void launch_hash_graph_probe_count_kernel(size_type num_rows,
                                          bitmask_type const* valid_rows,
                                          hash_graph_probe_state* probe_states,
                                          size_type* match_counts,
                                          cuda::std::uint64_t* total_matches,
                                          cuda::std::uint32_t* matched_rows,
                                          cuda::std::uint64_t* matched_build_rows,
                                          hash_graph_ref graph,
                                          Equal equal,
                                          Hasher hasher,
                                          cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_graph_block_size};
  // Keep reduction registers and shared memory out of the probes that need no total, such as
  // retrieval. A device-side null check still reserves those resources for every probe.
  dispatch_bool(total_matches != nullptr, [&](auto compute_sum) {
    hash_graph_probe_count_kernel<IsOuter, decltype(compute_sum)::value>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(num_rows,
                                                                             valid_rows,
                                                                             probe_states,
                                                                             match_counts,
                                                                             total_matches,
                                                                             matched_rows,
                                                                             matched_build_rows,
                                                                             graph,
                                                                             equal,
                                                                             hasher);
  });
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <bool IsOuter, typename Equal>
void launch_hash_graph_retrieve_kernel(size_type num_rows,
                                       hash_graph_probe_state const* probe_states,
                                       cuda::std::int64_t const* output_offsets,
                                       hash_graph_record const* records,
                                       size_type left_index_offset,
                                       size_type* left_indices,
                                       size_type* right_indices,
                                       Equal equal,
                                       cuda::stream_ref stream)
{
  auto const config = grid_1d{num_rows, hash_graph_block_size};
  hash_graph_retrieve_kernel<IsOuter>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(num_rows,
                                                                           probe_states,
                                                                           output_offsets,
                                                                           records,
                                                                           left_index_offset,
                                                                           left_indices,
                                                                           right_indices,
                                                                           equal);
  CUDF_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief Computes the output offset of each probe row from its match count
 *
 * @return One offset per probe row plus the end offset, and the total number of output pairs
 */
std::pair<cuda::device_buffer<cuda::std::int64_t>, std::size_t> hash_graph_match_offsets(
  device_span<size_type const> match_counts, cuda::stream_ref stream);

}  // namespace cudf::detail
