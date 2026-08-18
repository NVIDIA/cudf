/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/iterator>

namespace cudf {
namespace detail {
/**
 * @brief Generate a bitmask where every bit is set for which a predicate is
 * `true` over the elements in `[begin, begin + size)`.
 *
 * Bit `i` in the output mask will be set if `p(*(begin+i)) == true`.
 *
 * @tparam block_size The number of threads in the block
 * @param[out] output The output bitmask
 * @param[in] begin The beginning of the sequence of elements
 * @param[in] size The number of elements
 * @param[in] p The predicate to apply to each element
 * @param[out] valid_count The count of set bits in the output bitmask
 */
template <size_type block_size, typename InputIterator, typename Predicate>
CUDF_KERNEL void valid_if_kernel(
  bitmask_type* output, InputIterator begin, size_type size, Predicate p, size_type* valid_count)
{
  constexpr size_type leader_lane{0};
  auto const lane_id{threadIdx.x % warp_size};
  auto i            = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  size_type warp_valid_count{0};

  auto active_mask = __ballot_sync(0xFFFF'FFFFu, i < size);
  while (i < size) {
    bitmask_type ballot = __ballot_sync(active_mask, p(*(begin + i)));
    if (lane_id == leader_lane) {
      output[cudf::word_index(i)] = ballot;
      warp_valid_count += __popc(ballot);
    }
    i += stride;
    active_mask = __ballot_sync(active_mask, i < size);
  }

  size_type block_count = single_lane_block_sum_reduce<block_size, leader_lane>(warp_valid_count);
  if (threadIdx.x == 0) { atomicAdd(valid_count, block_count); }
}

/**
 * @brief Generate a bitmask where every bit is set for which a predicate is
 * `true` over the elements in `[begin,end)`.
 *
 * Bit `i` in the output mask will be set if `p(*(begin+i)) == true`.
 *
 * If `distance(begin,end) == 0`, returns an empty `rmm::device_buffer`.
 *
 * @throws cudf::logic_error if `(begin > end)`
 *
 * @param begin The beginning of the sequence
 * @param end The end of the sequence
 * @param p The predicate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param resources Memory resources used for the returned bitmask and temporary count scalar
 * @return A pair containing a `device_buffer` with the new bitmask and its null count
 */
template <typename InputIterator, typename Predicate>
std::pair<rmm::device_buffer, size_type> valid_if(InputIterator begin,
                                                  InputIterator end,
                                                  Predicate p,
                                                  rmm::cuda_stream_view stream,
                                                  cudf::memory_resources resources)
{
  CUDF_EXPECTS(begin <= end, "Invalid range.");

  size_type size = cuda::std::distance(begin, end);

  auto null_mask =
    cudf::create_null_mask(size, mask_state::UNINITIALIZED, stream, resources.get_output_mr());

  size_type null_count{0};
  if (size > 0) {
    cudf::detail::device_scalar<size_type> valid_count{0, stream, resources.get_temporary_mr()};

    constexpr size_type block_size{256};
    grid_1d grid{size, block_size};

    valid_if_kernel<block_size><<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      static_cast<bitmask_type*>(null_mask.data()), begin, size, p, valid_count.data());

    null_count = size - valid_count.value(stream);
  }
  return std::pair(std::move(null_mask), null_count);
}

/**
 * @brief Populates a set of bitmasks by applying a binary predicate to two
*         input ranges.

 * Given a set of bitmasks, `masks`, the state of bit `j` in mask `i` is
 * determined by `p( *(begin1 + i), *(begin2 + j))`. If the predicate evaluates
 * to true, the bit is set to `1`. If false, set to `0`.
 *
 * Example Arguments:
 * begin1:        zero-based counting iterator,
 * begin2:        zero-based counting iterator,
 * p:             [](size_type col, size_type row){ return col == row; }
 * masks:         [[b00...], [b00...], [b00...]]
 * mask_count:    3
 * mask_num_bits: 2
 * valid_counts:  [0, 0, 0]
 *
 * Example Results:
 * masks:         [[b10...], [b01...], [b00...]]
 * valid_counts:  [1, 1, 0]
 *
 * @note If any mask in `masks` is `nullptr`, that mask will be ignored.
 *
 * @param begin1        LHS arguments to binary predicate. ex: column/mask idx
 * @param begin2        RHS arguments to binary predicate. ex: row/bit idx
 * @param p             Predicate: `bit = p(begin1 + mask_idx, begin2 + bit_idx)`
 * @param masks         Masks for which bits will be obtained and assigned.
 * @param mask_count    The number of `masks`.
 * @param mask_num_bits The number of bits to assign for each mask. If this
 *                      number is smaller than the total number of bits, the
 *                      remaining bits may not be initialized.
 * @param valid_counts  Used to obtain the total number of valid bits for each
 *                      mask.
 */
template <typename InputIterator1,
          typename InputIterator2,
          typename BinaryPredicate,
          int32_t block_size>
CUDF_KERNEL void valid_if_n_kernel(InputIterator1 begin1,
                                   InputIterator2 begin2,
                                   BinaryPredicate p,
                                   bitmask_type* masks[],
                                   size_type mask_count,
                                   size_type mask_num_bits,
                                   size_type* valid_counts)
{
  for (size_type mask_idx = 0; mask_idx < mask_count; mask_idx++) {
    auto const mask = masks[mask_idx];
    if (mask == nullptr) { continue; }

    auto block_offset     = blockIdx.x * blockDim.x;
    auto warp_valid_count = static_cast<size_type>(0);

    while (block_offset < mask_num_bits) {
      auto const thread_idx    = block_offset + threadIdx.x;
      auto const thread_active = thread_idx < mask_num_bits;
      auto const arg_1         = *(begin1 + mask_idx);
      auto const arg_2         = *(begin2 + thread_idx);
      auto const bit_is_valid  = thread_active && p(arg_1, arg_2);
      auto const warp_validity = __ballot_sync(0xffff'ffffu, bit_is_valid);
      auto const mask_idx      = word_index(thread_idx);

      if (thread_active && threadIdx.x % warp_size == 0) { mask[mask_idx] = warp_validity; }

      warp_valid_count += __popc(warp_validity);
      block_offset += blockDim.x * gridDim.x;
    }

    auto block_valid_count = single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);

    if (threadIdx.x == 0) { atomicAdd(valid_counts + mask_idx, block_valid_count); }
  }
}

}  // namespace detail
}  // namespace cudf
