/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/errc.hpp>

extern "C" __device__ cudf::errc asymmetric_squared_loss(double* out,
                                                         double actual,
                                                         double target,
                                                         double penalty)
{
  auto const difference         = actual - target;
  auto const squared_difference = difference * difference;
  *out = actual >= target ? squared_difference : penalty * squared_difference;
  return cudf::errc::SUCCESS;
}
