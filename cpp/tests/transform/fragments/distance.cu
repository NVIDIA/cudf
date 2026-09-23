/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda/std/cmath>

__device__ float distance(float x1, float y1, float x2, float y2)
{
  return cuda::std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

extern "C" __device__ int transform(float* out, float x1, float y1, float x2, float y2)
{
  *out = distance(x1, y1, x2, y2);
  return 0;
}
