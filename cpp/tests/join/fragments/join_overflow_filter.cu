/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Join filter reading a per-pair `limit` extra input. The BOOL8 output is its physical type.
extern "C" __device__ int transform(unsigned char* output, int a, int b, int limit)
{
  if (a < limit) { return 1; }  // cudf::errc::ARITHMETIC_OVERFLOW
  if (b == 0) { return 2; }     // cudf::errc::DIVISION_BY_ZERO
  *output = a >= b;
  return 0;
}
