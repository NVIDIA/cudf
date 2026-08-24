/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/operators/operators.cuh>
#include <cudf/errc.hpp>

#include <cuda/std/expected>

/**
 * @brief Calculates the integer Lehmer mean of two integers using checked arithmetic.
 * The Lehmer mean is defined as `(a^2 + b^2) / (a + b)`.
 * @param a The first integer.
 * @param b The second integer.
 * @return The Lehmer mean of the two integers, or an error code if an overflow occurs during the
 * calculation.
 *
 */
__device__ cuda::std::expected<int32_t, cudf::errc> lehmer_mean(int32_t a, int32_t b)
{
  auto a2 = cudf::detail::ops::mul_overflow(a, a);
  if (!a2) return cuda::std::unexpected(a2.error());

  auto b2 = cudf::detail::ops::mul_overflow(b, b);
  if (!b2) return cuda::std::unexpected(b2.error());

  auto a_b_sum = cudf::detail::ops::add_overflow(a, b);
  if (!a_b_sum) return cuda::std::unexpected(a_b_sum.error());

  auto a2_b2_sum = cudf::detail::ops::add_overflow(a2.value(), b2.value());
  if (!a2_b2_sum) return cuda::std::unexpected(a2_b2_sum.error());

  return cudf::detail::ops::div_overflow(a2_b2_sum.value(), a_b_sum.value());
}

extern "C" __device__ int transform(int32_t* out, int32_t a, int32_t b)
{
  auto result = lehmer_mean(a, b);
  if (!result) return static_cast<int>(result.error());
  *out = result.value();
  return 0;
}
