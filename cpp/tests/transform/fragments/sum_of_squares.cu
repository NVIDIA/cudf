/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

__device__ float sum_of_squares(float a, float b) { return a * a + b * b; }

extern "C" __device__ int transform(float* out, float a, float b)
{
  *out = sum_of_squares(a, b);
  return 0;
}
