/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/scan.h>

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Basic type expected for iterators passed to `make_strings_column` that represent string
 * data in device memory.
 */
using string_index_pair = cuda::std::pair<char const*, size_type>;

/**
 * @brief Create a strings-type column from iterators of pointer/size pairs
 *
 * @tparam IndexPairIterator iterator over type `pair<char const*,size_type>` values
 *
 * @param begin First string row (inclusive)
 * @param end Last string row (exclusive)
 * @param stream CUDA stream used for device memory operations
 * @param mr  Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
template <typename IndexPairIterator>
std::unique_ptr<column> make_strings_column(IndexPairIterator begin,
                                            IndexPairIterator end,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  size_type strings_count = cuda::std::distance(begin, end);
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  // build offsets column from the strings sizes
  auto offsets_transformer =
    cuda::proclaim_return_type<size_type>([] __device__(string_index_pair item) -> size_type {
      return (item.first != nullptr ? static_cast<size_type>(item.second) : size_type{0});
    });
  auto offsets_transformer_itr = cuda::transform_iterator(begin, offsets_transformer);
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);

  // create null mask
  auto validator = [] __device__(string_index_pair const item) { return item.first != nullptr; };
  auto new_nulls = cudf::detail::valid_if(begin, end, validator, stream, mr);
  auto const null_count = new_nulls.second;
  auto null_mask        = (null_count > 0)
                            ? std::move(new_nulls.first)
                            : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr);

  // build chars column
  auto chars_data =
    make_chars_buffer(offsets_column->view(), bytes, begin, strings_count, stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             chars_data.release(),
                             null_count,
                             std::move(null_mask));
}

/**
 * @brief Builds the offsets child of a strings column without synchronizing
 *
 * @tparam OutputType Offset type of the output offsets column
 * @tparam IndexPairIterator iterator over type `pair<char const*,size_type>` values
 *
 * @param begin Iterator to the first string row
 * @param strings_count Number of rows
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The offsets column
 */
template <typename OutputType, typename IndexPairIterator>
std::unique_ptr<column> make_offsets_child_column_async(IndexPairIterator begin,
                                                        size_type strings_count,
                                                        cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr)
{
  auto offsets = make_numeric_column(
    data_type{type_to_id<OutputType>()}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto const sizes = cuda::proclaim_return_type<OutputType>(
    [strings_count, begin] __device__(size_type idx) -> OutputType {
      return idx < strings_count ? string_index_pair{begin[idx]}.second : OutputType{0};
    });
  auto const input_it = cudf::detail::make_counting_transform_iterator(0, sizes);
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         input_it,
                         input_it + strings_count + 1,
                         offsets->mutable_view().template data<OutputType>(),
                         OutputType{0});
  return offsets;
}

/**
 * @brief Builds the null mask, valid count and chars size of a strings column in one pass
 *
 * This is `cudf::detail::valid_if_kernel` with the size sum fused in, so it reads the input once;
 * fixes to either kernel likely apply to the other.
 *
 * @tparam block_size The number of threads in the block
 * @tparam IndexPairIterator iterator over type `pair<char const*,size_type>` values
 *
 * @param[out] null_mask The output bitmask
 * @param[in] begin Iterator to the first string row
 * @param[in] strings_count Number of rows
 * @param[out] valid_count The count of set bits in the output bitmask
 * @param[out] chars_size The total size of the strings in bytes
 */
template <size_type block_size, typename IndexPairIterator>
CUDF_KERNEL void strings_validity_and_size_kernel(bitmask_type* null_mask,
                                                  IndexPairIterator begin,
                                                  size_type strings_count,
                                                  size_type* valid_count,
                                                  int64_t* chars_size)
{
  constexpr size_type leader_lane{0};
  auto const lane_id{threadIdx.x % cudf::detail::warp_size};
  auto i            = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  size_type warp_valid_count{0};
  int64_t thread_chars_size{0};

  auto active_mask = __ballot_sync(0xFFFF'FFFFu, i < strings_count);
  while (i < strings_count) {
    auto const pair           = string_index_pair{begin[i]};
    bitmask_type const ballot = __ballot_sync(active_mask, pair.first != nullptr);
    if (lane_id == leader_lane) {
      null_mask[cudf::word_index(i)] = ballot;
      warp_valid_count += __popc(ballot);
    }
    thread_chars_size += pair.second;
    i += stride;
    active_mask = __ballot_sync(active_mask, i < strings_count);
  }

  auto const block_valid_count =
    cudf::detail::single_lane_block_sum_reduce<block_size, leader_lane>(warp_valid_count);
  using block_reduce = cub::BlockReduce<int64_t, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  auto const block_chars_size = block_reduce(temp_storage).Sum(thread_chars_size);
  if (threadIdx.x == 0) {
    atomicAdd(valid_count, block_valid_count);
    cuda::atomic_ref<int64_t, cuda::thread_scope_device> ref{*chars_size};
    ref.fetch_add(block_chars_size, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Creates a batch of strings columns from iterators of pointer/size pairs
 *
 * The output sizes of all columns are synchronized once instead of once per column.
 *
 * @tparam IndexPairIterator iterator over type `pair<char const*,size_type>` values
 *
 * @param input Per-column iterator to the first string row and number of rows
 * @param input_consumed Called with a column index once no work enqueued afterwards reads its
 *        input, so callers can release the input early, stream-ordered on `stream`, and keep the
 *        peak memory usage low
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return The strings columns, in the order of `input`
 */
template <typename IndexPairIterator>
std::vector<std::unique_ptr<column>> make_strings_column_batch(
  std::vector<std::pair<IndexPairIterator, size_type>> const& input,
  std::function<void(std::size_t)> const& input_consumed,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_columns = input.size();

  std::vector<cuda::device_buffer<cudf::bitmask_type>> null_masks;
  null_masks.reserve(num_columns);

  auto d_chars_sizes = cudf::detail::make_zeroed_device_uvector_async<int64_t>(
    num_columns, stream, cudf::get_current_device_resource_ref());
  auto d_valid_counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    num_columns, stream, cudf::get_current_device_resource_ref());

  // Only the sizes are needed before the sync. Offsets are built per column afterwards, so they
  // never coexist with the inputs of all columns.
  for (std::size_t idx = 0; idx < num_columns; ++idx) {
    auto const string_pairs = input[idx].first;
    auto const string_count = input[idx].second;
    null_masks.emplace_back(
      cudf::create_null_mask(string_count, mask_state::UNINITIALIZED, stream, mr));

    if (string_count == 0) { continue; }

    constexpr size_type block_size{256};
    auto const grid =
      cudf::detail::grid_1d{static_cast<thread_index_type>(string_count), block_size};
    strings_validity_and_size_kernel<block_size>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
        null_masks.back().data(),
        string_pairs,
        string_count,
        d_valid_counts.data() + idx,
        d_chars_sizes.data() + idx);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto const chars_sizes  = cudf::detail::make_std_vector_async(d_chars_sizes, stream);
  auto const valid_counts = cudf::detail::make_std_vector_async(d_valid_counts, stream);

  // Except for other stream syncs in `CUB` that we cannot control,
  // this should be the only stream sync we need in the entire API.
  stream.sync();

  auto const threshold = cudf::strings::get_offset64_threshold();
  CUDF_EXPECTS(
    cudf::strings::is_large_strings_enabled() ||
      std::none_of(chars_sizes.begin(),
                   chars_sizes.end(),
                   [threshold](auto const chars_size) { return chars_size >= threshold; }),
    "Size of output exceeds the column size limit",
    std::overflow_error);

  std::vector<std::unique_ptr<column>> output(num_columns);
  for (std::size_t idx = 0; idx < num_columns; ++idx) {
    auto const string_pairs  = input[idx].first;
    auto const strings_count = input[idx].second;
    if (strings_count == 0) {
      output[idx] = make_empty_column(type_id::STRING);
      if (input_consumed) { input_consumed(idx); }
      continue;
    }

    auto const chars_size  = chars_sizes[idx];
    auto const valid_count = valid_counts[idx];

    auto offsets =
      chars_size >= threshold
        ? make_offsets_child_column_async<int64_t>(string_pairs, strings_count, stream, mr)
        : make_offsets_child_column_async<size_type>(string_pairs, strings_count, stream, mr);
    auto chars_data =
      make_chars_buffer(offsets->view(), chars_size, string_pairs, strings_count, stream, mr);
    if (input_consumed) { input_consumed(idx); }

    auto const null_count = strings_count - valid_count;
    output[idx]           = make_strings_column(
      strings_count,
      std::move(offsets),
      chars_data.release(),
      null_count,
      null_count ? std::move(null_masks[idx])
                           : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
  }

  return output;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
