/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <rolling/detail/rolling_jit.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation_udf.cuh generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/operation_udf.cuh>

struct rolling_udf_ptx {
  template <typename OutType, typename InType>
  static OutType operate(InType const* in_col, cudf::size_type start, cudf::size_type count)
  {
    OutType ret;
    GENERIC_ROLLING_OP(&ret, 0, 0, 0, 0, &in_col[start], count, sizeof(InType));
    return ret;
  }
};

struct rolling_udf_cuda {
  template <typename OutType, typename InType>
  static OutType operate(InType const* in_col, cudf::size_type start, cudf::size_type count)
  {
    OutType ret;
    GENERIC_ROLLING_OP(&ret, in_col, start, count);
    return ret;
  }
};

namespace cudf {
namespace rolling {
namespace jit {

template <typename InType,
          typename OutType,
          class agg_op,
          typename PrecedingWindowType,
          typename FollowingWindowType>
__device__ void rolling_window_kernel(cudf::size_type nrows,
                                      void const* __restrict__ p_in_col,
                                      cudf::bitmask_type const* __restrict__ const in_col_valid,
                                      void* __restrict__ p_out_col,
                                      cudf::bitmask_type* __restrict__ out_col_valid,
                                      cudf::size_type* __restrict__ output_valid_count,
                                      detail::window_wrapper_base b_preceding_window_begin,
                                      detail::window_wrapper_base b_following_window_begin,
                                      cudf::size_type min_periods)
{
  auto i                                           = cudf::detail::grid_1d::global_thread_id();
  auto const stride                                = cudf::detail::grid_1d::grid_stride();
  PrecedingWindowType const preceding_window_begin = b_preceding_window_begin;
  FollowingWindowType const following_window_begin = b_following_window_begin;
  auto const* const in_col                         = static_cast<InType const*>(p_in_col);
  auto* const out_col                              = static_cast<OutType*>(p_out_col);

  cudf::size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffff'ffffu, i < nrows);
  while (i < nrows) {
    int64_t const preceding_window = preceding_window_begin[i];
    int64_t const following_window = following_window_begin[i];

    // compute bounds
    auto const start = static_cast<cudf::size_type>(
      min(static_cast<int64_t>(nrows), max(int64_t{0}, i - preceding_window + 1)));
    auto const end = static_cast<cudf::size_type>(
      min(static_cast<int64_t>(nrows), max(int64_t{0}, i + following_window + 1)));
    auto const start_index = min(start, end);
    auto const end_index   = max(start, end);

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.
    cudf::size_type count = end_index - start_index;
    OutType val           = agg_op::template operate<OutType, InType>(in_col, start_index, count);

    // check if we have enough input samples
    bool const output_is_valid = (count >= min_periods);

    // set the mask
    unsigned int const result_mask = __ballot_sync(active_threads, output_is_valid);

    // store the output value, one per thread
    if (output_is_valid) { out_col[i] = val; }

    // only one thread writes the mask
    if (0 == cudf::intra_word_index(i)) {
      out_col_valid[cudf::word_index(i)] = result_mask;
      warp_valid_count += __popc(result_mask);
    }

    // process next element
    i += stride;
    active_threads = __ballot_sync(active_threads, i < nrows);
  }

  // TODO: likely faster to do a single_lane_block_reduce and a single
  // atomic per block but that requires jit-compiling single_lane_block_reduce...
  if (0 == cudf::intra_word_index(threadIdx.x)) { atomicAdd(output_valid_count, warp_valid_count); }
}

}  // namespace jit
}  // namespace rolling
}  // namespace cudf

extern "C" __global__ void cudf_kernel_entry(
  cudf::size_type nrows,
  void const* __restrict__ in_col,
  cudf::bitmask_type const* __restrict__ in_col_valid,
  void* __restrict__ out_col,
  cudf::bitmask_type* __restrict__ out_col_valid,
  cudf::size_type* __restrict__ output_valid_count,
  cudf::detail::window_wrapper_base preceding_window_begin,
  cudf::detail::window_wrapper_base following_window_begin,
  cudf::size_type min_periods)
{
  CUDF_KERNEL_INSTANCE(nrows,
                       in_col,
                       in_col_valid,
                       out_col,
                       out_col_valid,
                       output_valid_count,
                       preceding_window_begin,
                       following_window_begin,
                       min_periods);
}
