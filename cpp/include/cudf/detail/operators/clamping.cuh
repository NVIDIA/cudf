/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/** @brief Identifies which replacement, if any, a clamped value needs. */
enum class clamp_bound : int8_t { NONE = 0, LOWER = 1, UPPER = 2 };

/**
 * @brief Tests enabled bounds, giving the lower bound precedence.
 *
 * @tparam T Value type.
 * @param a Input value.
 * @param lower Lower bound, or null to disable it.
 * @param upper Upper bound, or null to disable it.
 */
template <typename T>
__device__ clamp_bound clamp_position(T a,
                                      cuda::std::optional<T> lower,
                                      cuda::std::optional<T> upper)
  requires(!nullable<T>)
{
  if (lower.has_value() && a < *lower) { return clamp_bound::LOWER; }
  if (upper.has_value() && *upper < a) { return clamp_bound::UPPER; }
  return clamp_bound::NONE;
}

/**
 * @brief Replaces values outside enabled bounds, preserving in-range values.
 *
 * Replacement values are accessed only if their corresponding bound is crossed.
 *
 * @tparam T Value type.
 * @param a Input value.
 * @param lower Lower bound, or null to disable it.
 * @param lower_replacement Replacement for values below the lower bound; valid when selected.
 * @param upper Upper bound, or null to disable it.
 * @param upper_replacement Replacement for values above the upper bound; valid when selected.
 */
template <typename T>
__device__ T clamp(T a,
                   cuda::std::optional<T> lower,
                   cuda::std::optional<T> lower_replacement,
                   cuda::std::optional<T> upper,
                   cuda::std::optional<T> upper_replacement)
  requires(!nullable<T>)
{
  switch (clamp_position(a, lower, upper)) {
    case clamp_bound::LOWER: return *lower_replacement;
    case clamp_bound::UPPER: return *upper_replacement;
    default: return a;
  }
}

template <typename T>
__device__ cuda::std::optional<T> clamp(cuda::std::optional<T> a,
                                        cuda::std::optional<T> lower,
                                        cuda::std::optional<T> lower_replacement,
                                        cuda::std::optional<T> upper,
                                        cuda::std::optional<T> upper_replacement)
  requires(!nullable<T>)
{
  if (!a.has_value()) { return cuda::std::nullopt; }
  return clamp(*a, lower, lower_replacement, upper, upper_replacement);
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
