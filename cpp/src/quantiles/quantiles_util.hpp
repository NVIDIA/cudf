/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/assert.cuh>
#include <cudf/fixed_point/conv.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/cmath>
#include <cuda/std/functional>

namespace cudf {
namespace detail {

namespace interpolate {
template <typename Result, typename T>
CUDF_HOST_DEVICE inline Result linear(T lhs, T rhs, double frac)
{
  // TODO: safe operation to avoid overflow/underflow
  // double can fully represent int8-32 value range.
  // Since the fraction part of double is 52 bits,
  // double cannot fully represent int64.
  // Underflow may occur when converting int64 to double
  // detail: https://github.com/NVIDIA/cudf/issues/1417

  auto dlhs             = convert_to_floating<double>(lhs);
  auto drhs             = convert_to_floating<double>(rhs);
  double one_minus_frac = 1.0 - frac;
  return static_cast<Result>(one_minus_frac * dlhs + frac * drhs);
}

template <typename Result, typename T>
CUDF_HOST_DEVICE inline Result midpoint(T lhs, T rhs)
{
  // TODO: try std::midpoint (C++20) if available
  auto dlhs = convert_to_floating<double>(lhs);
  auto drhs = convert_to_floating<double>(rhs);
  return static_cast<Result>(dlhs / 2 + drhs / 2);
}

template <typename Result>
CUDF_HOST_DEVICE inline Result midpoint(int64_t lhs, int64_t rhs)
{
  // caring to avoid integer overflow and underflow between int64_t and Result( double )
  int64_t half = lhs / 2 + rhs / 2;
  int64_t rest = lhs % 2 + rhs % 2;
  return static_cast<Result>(static_cast<Result>(half) + static_cast<Result>(rest) * 0.5);
}

template <>
CUDF_HOST_DEVICE inline int64_t midpoint(int64_t lhs, int64_t rhs)
{
  // caring to avoid integer overflow
  int64_t half   = lhs / 2 + rhs / 2;
  int64_t rest   = lhs % 2 + rhs % 2;
  int64_t result = half;

  // rounding toward zero
  result += (half >= 0 && rest != 0) ? rest / 2 : 0;
  result += (half < 0 && rest != 0) ? 1 : 0;

  return result;
}

}  // namespace interpolate

struct quantile_index {
  size_type lower;
  size_type higher;
  size_type nearest;
  size_type nearest_half_up;
  double fraction;

  CUDF_HOST_DEVICE inline quantile_index(size_type count, double quantile)
  {
    quantile = cuda::std::min(cuda::std::max(quantile, 0.0), 1.0);

    double val      = quantile * (count - 1);
    lower           = cuda::std::floor(val);
    higher          = static_cast<size_type>(cuda::std::ceil(val));
    nearest         = static_cast<size_type>(cuda::std::nearbyint(val));
    nearest_half_up = static_cast<size_type>(cuda::std::round(val));
    fraction        = val - lower;
  }
};

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
/* @brief computes a quantile value.
 *
 * Computes a value for a quantile by interpolating between two values on either
 * side of the desired quantile.
 *
 * `get_value` must have signature: `T <T>(size_type)` where T can be
 * `static_cast` to `Result`.
 *
 * @param[in] get_value Gets the value at a given index in range [0, size].
 * @param[in] size      Number of values indexed by `get_value`.
 * @param[in] q         Desired quantile in range [0, 1].
 * @param[in] interp    Strategy used to interpolate between the two values
 *                      on either side of the desired quantile.
 *
 * @returns Value of the desired quantile.
 */
template <typename Result, typename ValueAccessor>
CUDF_HOST_DEVICE inline Result select_quantile(ValueAccessor get_value,
                                               size_type size,
                                               double q,
                                               interpolation interp)
{
  if (size < 2) { return get_value(0); }

  quantile_index idx(size, q);

  switch (interp) {
    case interpolation::LINEAR:
      return interpolate::linear<Result>(get_value(idx.lower), get_value(idx.higher), idx.fraction);

    case interpolation::MIDPOINT:
      return interpolate::midpoint<Result>(get_value(idx.lower), get_value(idx.higher));

    case interpolation::LOWER: return static_cast<Result>(get_value(idx.lower));

    case interpolation::HIGHER: return static_cast<Result>(get_value(idx.higher));

    case interpolation::NEAREST: return static_cast<Result>(get_value(idx.nearest));

    case interpolation::NEAREST_HALF_UP: return static_cast<Result>(get_value(idx.nearest_half_up));

    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Invalid interpolation operation for quantiles.");
#else
      CUDF_UNREACHABLE("Invalid interpolation operation for quantiles");
#endif
    }
  }
}

template <typename Result, typename Iterator>
CUDF_HOST_DEVICE inline Result select_quantile_data(Iterator begin,
                                                    size_type size,
                                                    double q,
                                                    interpolation interp)
{
  if (size == 0) return static_cast<Result>(*begin);

  quantile_index idx(size, q);

  switch (interp) {
    case interpolation::LOWER: return static_cast<Result>(*(begin + idx.lower));

    case interpolation::HIGHER: return static_cast<Result>(*(begin + idx.higher));

    case interpolation::NEAREST: return static_cast<Result>(*(begin + idx.nearest));

    case interpolation::NEAREST_HALF_UP: return static_cast<Result>(*(begin + idx.nearest_half_up));

    case interpolation::LINEAR:
      return interpolate::linear<Result>(*(begin + idx.lower), *(begin + idx.higher), idx.fraction);

    case interpolation::MIDPOINT:
      return interpolate::midpoint<Result>(*(begin + idx.lower), *(begin + idx.higher));
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Invalid interpolation operation for quantiles.");
#else
      CUDF_UNREACHABLE("Invalid interpolation operation for quantiles");
#endif
    }
  }
}

template <typename Iterator>
CUDF_HOST_DEVICE inline bool select_quantile_validity(Iterator begin,
                                                      size_type size,
                                                      double q,
                                                      interpolation interp)
{
  quantile_index idx(size, q);

  switch (interp) {
    case interpolation::HIGHER: return *(begin + idx.higher);

    case interpolation::LOWER: return *(begin + idx.lower);

    case interpolation::NEAREST: return *(begin + idx.nearest);

    case interpolation::NEAREST_HALF_UP: return *(begin + idx.nearest_half_up);

    case interpolation::LINEAR:
    case interpolation::MIDPOINT: return *(begin + idx.lower) and *(begin + idx.higher);
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Invalid interpolation operation for quantiles.");
#else
      CUDF_UNREACHABLE("Invalid interpolation operation for quantiles");
#endif
    }
  }
}

}  // namespace detail
}  // namespace cudf
