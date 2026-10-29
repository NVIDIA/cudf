/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace parquet_test {
struct offset_gen {
  int const group_size;
  __device__ int operator()(int i) { return i * group_size; }
};

struct bool_gen {
  __device__ bool operator()(int i) { return i % 2 == 0; }
};
}  // namespace parquet_test
