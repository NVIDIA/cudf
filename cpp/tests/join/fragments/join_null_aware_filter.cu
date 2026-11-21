/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/std/optional>

// Null-aware join filter with a scalar `offset` extra input. A null `b` rejects the pair.
extern "C" __device__ int transform(cuda::std::optional<unsigned char>* output,
                                    cuda::std::optional<int> a,
                                    cuda::std::optional<int> b,
                                    cuda::std::optional<int> offset)
{
  if (!b.has_value()) {
    *output = cuda::std::nullopt;
    return 0;
  }
  *output = !a.has_value() || *a + *offset >= *b;
  return 0;
}
