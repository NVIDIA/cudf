/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_reduce.cuh>
#include <cuda/devices>
#include <cuda/std/type_traits>

namespace cudf {
namespace reduction {
namespace detail {
/**
 * @brief A custom tuning policy for the block reductions for one-pass variance computation.
 * @tparam State type of the state being reduced over.
 * @tparam BinaryOp type of the binary operator used in the reduction.
 *
 * @note This policy preserves CUB's architecture-specific tuning, changing only the
 * collective used to combine expensive intermediate states for variance.
 */
template <typename State, typename BinaryOp>
struct variance_reduce_policy {
  CUDF_HOST_DEVICE constexpr cub::ReducePolicy operator()(cuda::compute_capability cc) const
  {
    // TODO: Update to public CUB facilities using policy transformers once that exists.
    auto policy = cub::detail::reduce::
      policy_selector_from_types<State, cuda::std::make_unsigned_t<size_type>, BinaryOp>{}(cc);
    // Combining the states for variance is quite expensive. By using raking block reduce,
    // we use significantly less collective FP64 arithmetic than CUB's default policy.
    policy.multi_tile.reduce_algorithm  = cub::BLOCK_REDUCE_RAKING;
    policy.single_tile.reduce_algorithm = cub::BLOCK_REDUCE_RAKING;
    return policy;
  }
};

/**
 * @brief A custom tuning policy for segmented one-pass variance computation.
 * @tparam State Type of the state being reduced over.
 * @tparam Offset Type of the segment offsets.
 * @tparam BinaryOp Type of the binary operator used in the reduction.
 *
 * @note Preserves CUB's architecture-specific tuning, changing only the block
 * collective used to combine expensive intermediate states for variance.
 */
template <typename State, typename Offset, typename BinaryOp>
struct variance_segmented_reduce_policy {
  CUDF_HOST_DEVICE constexpr cub::SegmentedReducePolicy operator()(
    cuda::compute_capability cc) const
  {
    // TODO: Update to public CUB facilities using policy transformers once that exists.
    auto policy =
      cub::detail::segmented_reduce::policy_selector_from_types<State, Offset, BinaryOp>{}(cc);
    policy.large_reduce.reduce_algorithm = cub::BLOCK_REDUCE_RAKING;
    return policy;
  }
};

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
