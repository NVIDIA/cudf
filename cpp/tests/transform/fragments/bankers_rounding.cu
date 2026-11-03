/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/fixed_point/fixed_point.hpp>

template <typename Decimal>
__device__ Decimal bankers_round(Decimal x)
{
  using Rep = typename Decimal::rep;

  auto const scale = x.scale();
  auto const value = x.value();

  if (scale >= numeric::scale_type{0}) { return x; }

  Rep factor = 1;
  for (int32_t i = 0; i < -static_cast<int32_t>(scale); ++i) {
    factor *= 10;
  }

  Rep q = value / factor;
  Rep r = value % factor;

  if (r < 0) r = -r;

  auto const half = factor / 2;

  bool round_up = false;

  if (r > half) {
    round_up = true;
  } else if (r == half) {
    // tie: round to even
    round_up = (q % 2) != 0;
  }

  if (round_up) { q += value >= 0 ? Rep{1} : Rep{-1}; }

  return Decimal{q, numeric::scale_type{0}};
}

extern "C" __device__ int transform(numeric::decimal128* out, numeric::decimal128 in)
{
  *out = bankers_round(in);
  return 0;
}
