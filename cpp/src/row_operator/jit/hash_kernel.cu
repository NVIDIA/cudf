/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_operator/jit/hash_kernel.cuh"

#pragma nv_hdrstop

#include <cudf/detail/kernel_instance.cuh>

extern "C" {

__global__ void cudf_kernel_entry(cudf_abi::column_device_view const* columns,
                                  cudf_abi::size_type num_rows,
                                  cudf_abi::uint64_t seed,
                                  bool has_nulls,
                                  void* low,
                                  void* high)
{
  CUDF_KERNEL_INSTANCE(columns, num_rows, seed, has_nulls, low, high);
}
}
