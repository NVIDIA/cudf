/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <row_operator/jit/row_function_dispatch.cuh>

#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

extern "C" __global__ void cudf_kernel_entry(cudf::table_device_view table,
                                             cudf::size_type num_rows,
                                             uint32_t seed,
                                             cudf::hash_value_type* output)
{
  auto const stride = cudf::detail::grid_1d::grid_stride();

  for (auto row = cudf::detail::grid_1d::global_thread_id(); row < num_rows; row += stride) {
    output[row] = cudf_row_hash(table.begin(), row, seed);
  }
}
