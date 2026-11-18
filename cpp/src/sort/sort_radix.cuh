/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <cuda/std/cmath>
#include <cuda/std/tuple>

namespace cudf {
namespace detail {

/**
 * @brief Radix sort key for a floating-point value
 *
 * The bias `s` is the most significant part of the key and is non-zero only for NaN values.
 */
template <typename F>
struct float_pair {
  size_type s;  // index bias for sorting nan
  F f;          // actual float value to sort
};

/**
 * @brief Decomposes a float_pair into its radix sort key parts for cub
 */
template <typename F>
struct float_decomposer {
  __device__ cuda::std::tuple<size_type&, F&> operator()(float_pair<F>& key) const
  {
    return {key.s, key.f};
  }
};

/**
 * @brief Builds the float_pair radix sort key for row `idx`
 */
template <typename F>
struct float_to_pair_fn {
  F const* fs;
  size_type size;  // number of rows
  bool ascending;  // sort direction
  __device__ float_pair<F> operator()(size_type idx) const
  {
    auto const f = fs[idx];
    // NaNs compare greater than all other values: a non-zero bias makes them larger than any
    // non-NaN. The bias is ordered by row index in the sort direction so equal NaNs remain stable.
    auto const s = cuda::std::isnan(f) * (ascending ? idx + 1 : size - idx);
    return float_pair<F>{s, f};
  }
};

}  // namespace detail
}  // namespace cudf
