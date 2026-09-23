/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/std/cstdint>

__device__ uint8_t to_upper(uint8_t input)
{
  if (input > 96 && input < 123) {
    return input - 32;
  } else {
    return input;
  }
}

extern "C" __device__ int transform(uint8_t* output, uint8_t input)
{
  *output = to_upper(input);
  return 0;
}
