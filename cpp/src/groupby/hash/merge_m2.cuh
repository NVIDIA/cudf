/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/traits.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/std/cmath>
#include <cuda/std/limits>

namespace cudf::groupby::detail::hash {

template <typename T>
concept count_type = cudf::is_integral_not_bool<T>() || cudf::is_floating_point<T>();

template <count_type T>
struct m2_state {
  T count;
  double mean;
  double m2;
};

struct m2_value {
  double value;
  bool valid;

  template <count_type T>
  __device__ operator m2_state<T>() const
  {
    if (!valid) { return {T{0}, 0.0, 0.0}; }
    return {T{1},
            value,
            cuda::std::isfinite(value) ? 0.0 : cuda::std::numeric_limits<double>::quiet_NaN()};
  }
};

template <count_type T>
struct merge_m2 {
  // Merging the intermediate states is expensive: raking avoids the overlapping partial reductions
  // performed by the default warp-based block algorithm, and benchmarks as more performant.
  // Although this operator is commutative, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
  // benchmarked as slower on B200.
  static constexpr auto block_reduce_algorithm = cub::BLOCK_REDUCE_RAKING;

  __device__ m2_state<T> operator()(m2_state<T> const& a, m2_state<T> const& b) const
  {
    // Avoid inf * 0 for empty partials and retain non-finite singleton M2.
    if (a.count == 0) { return b; }
    if (b.count == 0) { return a; }
    // Make non-finite results independent of the order in which states are merged.
    // Adding the means preserves +/-inf when finite or same-sign infinities combine, and yields
    // NaN for opposite-sign infinities or NaN inputs.
    if (!cuda::std::isfinite(a.mean) || !cuda::std::isfinite(b.mean)) {
      auto constexpr nan = cuda::std::numeric_limits<double>::quiet_NaN();
      return {a.count + b.count, a.mean + b.mean, nan};
    }
    auto const count = a.count + b.count;
    auto const delta = b.mean - a.mean;
    // This is the same merge operator as for whole-column reductions of M2. A rounded
    // reciprocal avoids both full division and a count-dependent lookup, including when
    // merging partials across chunks.
    auto const rcp     = __drcp_rn(static_cast<double>(count));
    auto const delta_n = delta * rcp;
    // An overflowed delta would make the sign of the merged mean depend on the merge order.
    auto const mean = cuda::std::isfinite(delta)
                        ? a.mean + delta_n * b.count
                        : a.mean * (a.count * rcp) + b.mean * (b.count * rcp);
    return {count, mean, a.m2 + b.m2 + delta * delta_n * a.count * b.count};
  }

  // Thread-local folds see individual values; collective reductions see full states. This
  // is a simplification of the above merge for the case where the additional state
  // b.count == 1.
  __device__ m2_state<T> operator()(m2_state<T> const& a, m2_value const& b) const
  {
    if (!b.valid) { return a; }
    if (a.count == 0) { return static_cast<m2_state<T>>(b); }
    auto const delta = b.value - a.mean;
    // Preserve the general merge's non-finite/overflow behavior.
    if (!cuda::std::isfinite(delta)) { return (*this)(a, static_cast<m2_state<T>>(b)); }
    auto const count   = a.count + T{1};
    auto const delta_n = delta * __drcp_rn(static_cast<double>(count));
    // As well as avoiding some work compared to the full merge, this specialization means
    // avoids the problem that subtracting the updated mean from b.value would lose small
    // deviations when the rounded mean equals b.value.
    return {count, a.mean + delta_n, a.m2 + delta * delta_n * a.count};
  }
};

}  // namespace cudf::groupby::detail::hash
