/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>

#include <cuda/numeric>
#include <cuda/std/type_traits>

namespace cudf {
namespace reduction::detail {

/**
 * @brief Type of the wrap counter in the accumulator for an `Overflow` output child.
 *
 * `FLAG` (`bool`) keeps the 32-bit counter so its accumulator size is unchanged; `COUNT`
 * (`int64_t`) reports the count as INT64, so it accumulates in 64 bits.
 */
template <typename Overflow>
using sum_overflow_wraps_t =
  cuda::std::conditional_t<cuda::std::is_same_v<Overflow, bool>, cudf::size_type, int64_t>;

/**
 * @brief Running accumulator for a sum that detects signed-integer overflow.
 *
 * `sum` is the running sum modulo 2^W and `wraps` is the net number of times it has stepped
 * outside [MIN, MAX], where W is the bit width of `DeviceType`. The exact total is
 * `wraps * 2^W + sum`, and a final `wraps == 0` means the true sum fits in `DeviceType`.
 */
template <typename DeviceType, typename Wraps>
struct sum_overflow_result {
  DeviceType sum;
  Wraps wraps;

  CUDF_HOST_DEVICE sum_overflow_result() : sum{0}, wraps{0} {}
  CUDF_HOST_DEVICE sum_overflow_result(DeviceType s, Wraps w) : sum{s}, wraps{w} {}
};

/// @brief Associative combine: wrap the sums and track the net carry direction.
template <typename DeviceType, typename Wraps>
struct overflow_sum_op {
  __device__ sum_overflow_result<DeviceType, Wraps> operator()(
    sum_overflow_result<DeviceType, Wraps> const& lhs,
    sum_overflow_result<DeviceType, Wraps> const& rhs) const
  {
    auto const r     = cuda::add_overflow<DeviceType>(lhs.sum, rhs.sum);
    auto const carry = r.overflow ? (rhs.sum > DeviceType{0} ? Wraps{1} : Wraps{-1}) : Wraps{0};
    return sum_overflow_result<DeviceType, Wraps>{r.value, lhs.wraps + rhs.wraps + carry};
  }
};

/// @brief Maps a value to a zero-wrap accumulator.
template <typename DeviceType, typename Wraps>
struct to_sum_overflow {
  __device__ sum_overflow_result<DeviceType, Wraps> operator()(DeviceType value) const
  {
    return sum_overflow_result<DeviceType, Wraps>{value, 0};
  }
};

/// @brief Maps a row index to an accumulator, treating nulls as a zero contribution.
template <typename DeviceType, typename Wraps>
struct null_aware_to_sum_overflow {
  cudf::column_device_view dcol;

  CUDF_HOST_DEVICE null_aware_to_sum_overflow(cudf::column_device_view const& d) : dcol{d} {}

  __device__ sum_overflow_result<DeviceType, Wraps> operator()(cudf::size_type idx) const
  {
    return dcol.is_valid(idx)
             ? sum_overflow_result<DeviceType, Wraps>{dcol.element<DeviceType>(idx), 0}
             : sum_overflow_result<DeviceType, Wraps>{DeviceType{0}, 0};
  }
};

/**
 * @brief Converts a net wrap count into the value of the SUM_OVERFLOW `overflow` child.
 *
 * @tparam Overflow `bool` for `sum_overflow_output::FLAG`, `int64_t` for
 * `sum_overflow_output::COUNT`
 */
template <typename Overflow, typename Wraps>
CUDF_HOST_DEVICE inline Overflow wraps_to_overflow(Wraps wraps)
{
  if constexpr (cuda::std::is_same_v<Overflow, bool>) {
    return wraps != 0;
  } else {
    return static_cast<Overflow>(wraps);
  }
}

}  // namespace reduction::detail
}  // namespace cudf
