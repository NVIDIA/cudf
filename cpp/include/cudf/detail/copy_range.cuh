/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cub/cub.cuh>
#include <cuda/stream>
#include <cuda_runtime.h>

#include <memory>

namespace cudf::detail {
template <cudf::size_type block_size,
          typename SourceValueIterator,
          typename SourceValidityIterator,
          typename T,
          bool has_validity>
CUDF_KERNEL void copy_range_kernel(SourceValueIterator source_value_begin,
                                   SourceValidityIterator source_validity_begin,
                                   cudf::mutable_column_device_view target,
                                   cudf::size_type target_begin,
                                   cudf::size_type target_end,
                                   cudf::size_type* __restrict__ const null_count)
{
  using cudf::detail::warp_size;

  static_assert(block_size <= 1024, "copy_range_kernel assumes block_size is not larger than 1024");
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "copy_range_kernel assumes bitmask element size in bits == warp size");

  constexpr cudf::size_type leader_lane{0};
  int const lane_id = threadIdx.x % warp_size;

  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const warp_id = tid / warp_size;

  cudf::size_type const offset         = target.offset();
  cudf::size_type const begin_mask_idx = cudf::word_index(offset + target_begin);
  cudf::size_type const end_mask_idx   = cudf::word_index(offset + target_end);

  cudf::size_type mask_idx             = begin_mask_idx + warp_id;
  cudf::size_type const masks_per_grid = cudf::detail::grid_1d::grid_stride() / warp_size;

  cudf::size_type target_offset = begin_mask_idx * warp_size - (offset + target_begin);
  cudf::size_type source_idx    = tid + target_offset;

  cudf::size_type warp_null_change{0};

  while (mask_idx <= end_mask_idx) {
    cudf::size_type index = mask_idx * warp_size + lane_id - offset;
    bool in_range         = (index >= target_begin && index < target_end);

    // write data
    if (in_range) target.element<T>(index) = *(source_value_begin + source_idx);

    if (has_validity) {  // update bitmask
      bool const valid      = in_range && *(source_validity_begin + source_idx);
      int const active_mask = __ballot_sync(0xFFFF'FFFFu, in_range);
      int const valid_mask  = __ballot_sync(0xFFFF'FFFFu, valid);
      int const warp_mask   = active_mask & valid_mask;

      cudf::bitmask_type old_mask = target.get_mask_word(mask_idx);
      if (lane_id == leader_lane) {
        cudf::bitmask_type new_mask = (old_mask & ~active_mask) | warp_mask;
        target.set_mask_word(mask_idx, new_mask);
        warp_null_change += __popc(active_mask & old_mask) - __popc(active_mask & new_mask);
      }
    }

    source_idx += cudf::detail::grid_1d::grid_stride();
    mask_idx += masks_per_grid;
  }

  if (has_validity) {
    auto block_null_change =
      cudf::detail::single_lane_block_sum_reduce<block_size, leader_lane>(warp_null_change);
    if (threadIdx.x == 0) {  // if the first thread in a block
      atomicAdd(null_count, block_null_change);
    }
  }
}

/**
 * @brief Internal API to copy a range of values from source iterators to a
 * target column.
 *
 * The elements indicated by the indices [@p target_begin, @p target_end) were
 * replaced with the elements retrieved from source iterators;
 * *(@p source_value_begin + idx) if *(@p source_validity_begin + idx) is true,
 * invalidate otherwise (where idx = [0, @p target_end - @p target_begin)).
 * @p target is modified in place.
 *
 * @tparam SourceValueIterator Iterator for retrieving source values
 * @tparam SourceValidityIterator Iterator for retrieving source validities
 * @param source_value_begin Start of source value iterator
 * @param source_validity_begin Start of source validity iterator
 * @param target the column to copy into
 * @param target_begin The starting index of the target range (inclusive)
 * @param target_end The index of the last element in the target range
 * (exclusive)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename SourceValueIterator, typename SourceValidityIterator>
void copy_range(SourceValueIterator source_value_begin,
                SourceValidityIterator source_validity_begin,
                mutable_column_view& target,
                size_type target_begin,
                size_type target_end,
                cuda::stream_ref stream)
{
  CUDF_EXPECTS((target_begin <= target_end) && (target_begin >= 0) &&
                 (target_begin < target.size()) && (target_end <= target.size()),
               "Range is out of bounds.");
  using T = typename std::iterator_traits<SourceValueIterator>::value_type;

  // this code assumes that source and target have the same type.
  CUDF_EXPECTS(type_id_matches_device_storage_type<T>(target.type().id()), "data type mismatch");

  auto warp_aligned_begin_lower_bound = cudf::util::round_down_safe(target_begin, warp_size);
  auto warp_aligned_end_upper_bound   = cudf::util::round_up_safe<int64_t>(target_end, warp_size);
  auto num_items = warp_aligned_end_upper_bound - warp_aligned_begin_lower_bound;

  constexpr size_type block_size{256};

  auto grid = cudf::detail::grid_1d{num_items, block_size, 1};

  if (target.nullable()) {
    cudf::detail::device_scalar<size_type> null_count(
      target.null_count(), stream, cudf::get_current_device_resource_ref());

    auto kernel =
      copy_range_kernel<block_size, SourceValueIterator, SourceValidityIterator, T, true>;
    kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(
      source_value_begin,
      source_validity_begin,
      *mutable_column_device_view::create(target, stream),
      target_begin,
      target_end,
      null_count.data());

    target.set_null_count(null_count.value(stream));
  } else {
    auto kernel =
      copy_range_kernel<block_size, SourceValueIterator, SourceValidityIterator, T, false>;
    kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(
      source_value_begin,
      source_validity_begin,
      *mutable_column_device_view::create(target, stream),
      target_begin,
      target_end,
      nullptr);
  }

  CUDF_CHECK_CUDA(stream.get());
}

/**
 * @copydoc cudf::copy_range_in_place
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void copy_range_in_place(column_view const& source,
                         mutable_column_view& target,
                         size_type source_begin,
                         size_type source_end,
                         size_type target_begin,
                         cuda::stream_ref stream);

/**
 * @copydoc cudf::copy_range
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return std::unique_ptr<column> The result target column
 */
std::unique_ptr<column> copy_range(column_view const& source,
                                   column_view const& target,
                                   size_type source_begin,
                                   size_type source_end,
                                   size_type target_begin,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
