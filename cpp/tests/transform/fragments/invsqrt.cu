/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

__device__ float invsqrt(float a) { return 1.0F / sqrtf(a); }

extern "C" __device__ int transform(float* out, float a)
{
  *out = invsqrt(a);
  return 0;
}
