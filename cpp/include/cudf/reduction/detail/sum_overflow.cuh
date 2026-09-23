/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>

#include <cuda/numeric>

namespace cudf {
namespace reduction::detail {

/**
 * @brief Running accumulator for a sum that detects signed-integer overflow.
 *
 * `wraps` is the net number of times the running sum has stepped outside [MIN, MAX].
 * A final `wraps == 0` means the true sum fits in `DeviceType`, i.e. no overflow.
 */
template <typename DeviceType>
struct sum_overflow_result {
  DeviceType sum;
  cudf::size_type wraps;

  CUDF_HOST_DEVICE sum_overflow_result() : sum{0}, wraps{0} {}
  CUDF_HOST_DEVICE sum_overflow_result(DeviceType s, cudf::size_type w) : sum{s}, wraps{w} {}
};

/// @brief Associative combine: wrap the sums and track the net carry direction.
template <typename DeviceType>
struct overflow_sum_op {
  __device__ sum_overflow_result<DeviceType> operator()(
    sum_overflow_result<DeviceType> const& lhs, sum_overflow_result<DeviceType> const& rhs) const
  {
    auto const r     = cuda::add_overflow<DeviceType>(lhs.sum, rhs.sum);
    auto const carry = r.overflow ? (rhs.sum > DeviceType{0} ? 1 : -1) : 0;
    return sum_overflow_result<DeviceType>{r.value, lhs.wraps + rhs.wraps + carry};
  }
};

/// @brief Maps a value to a zero-wrap accumulator.
template <typename DeviceType>
struct to_sum_overflow {
  __device__ sum_overflow_result<DeviceType> operator()(DeviceType value) const
  {
    return sum_overflow_result<DeviceType>{value, 0};
  }
};

/// @brief Maps a row index to an accumulator, treating nulls as a zero contribution.
template <typename DeviceType>
struct null_aware_to_sum_overflow {
  cudf::column_device_view dcol;

  CUDF_HOST_DEVICE null_aware_to_sum_overflow(cudf::column_device_view const& d) : dcol{d} {}

  __device__ sum_overflow_result<DeviceType> operator()(cudf::size_type idx) const
  {
    return dcol.is_valid(idx) ? sum_overflow_result<DeviceType>{dcol.element<DeviceType>(idx), 0}
                              : sum_overflow_result<DeviceType>{DeviceType{0}, 0};
  }
};

}  // namespace reduction::detail
}  // namespace cudf
