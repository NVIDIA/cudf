/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#pragma nv_hdrstop

#include <cudf/detail/kernel_instance.cuh>

namespace cudf::detail {

using transform_udf_type = CUDF_TRANSFORM_UDF_TYPE;

extern "C" __device__ transform_udf_type cudf_transform_udf;

template <typename T0, typename T1, typename T2, typename Output>
__device__ void ternary_op_kernel(column_device_view_core const* inputs,
                                  mutable_column_device_view_core const* output,
                                  size_type n,
                                  error_int* error_code)
{
  auto const start  = grid_1d::global_thread_id();
  auto const stride = grid_1d::grid_stride();

  for (auto row = start; row < n; row += stride) {
    auto const error = cudf_transform_udf(
#if CUDF_TRANSFORM_UDF_INDEXED
      row,
#endif
      &output->element<Output>(row),
      inputs[0].element<T0>(row),
      inputs[1].element<T1>(row),
      inputs[2].element<T2>(row));
    if (error != errc::SUCCESS) { atomicMax(error_code, static_cast<error_int>(error)); }
  }
}

extern "C" __global__ void cudf_kernel_entry(column_device_view_core const* inputs,
                                             mutable_column_device_view_core const* output,
                                             size_type n,
                                             error_int* error_code)
{
  CUDF_KERNEL_INSTANCE(inputs, output, n, error_code);
}

}  // namespace cudf::detail
