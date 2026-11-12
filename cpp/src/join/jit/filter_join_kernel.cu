/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#include <jit/column_accessor.cuh>
#include <jit/type_list.cuh>
#include <join/jit/filter_join_kernel.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation_udf.cuh generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
// These headers select the kernel instance and declare the UDF signature.
#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/operation_udf.cuh>
// clang-format on

using transform_type = CUDF_UDF_TYPE;
extern "C" __device__ transform_type cudf_transform_udf;

extern "C" __global__ void cudf_kernel_entry(
  cudf::size_type num_rows,
  cudf::size_type const* __restrict__ left_indices,
  cudf::size_type const* __restrict__ right_indices,
  cudf::column_device_view_core const* __restrict__ columns,
  bool* __restrict__ predicate_results,
  cudf::error_int* __restrict__ max_error)
{
  CUDF_KERNEL_INSTANCE(num_rows,
                       left_indices,
                       right_indices,
                       columns,
                       predicate_results,
                       max_error,
                       [](cudf::size_type left_index, cudf::size_type right_index, auto... args) {
#if CUDF_UDF_INDEXED
                         return cudf_transform_udf(left_index, right_index, args...);
#else
      return cudf_transform_udf(args...);
#endif
                       });
}
