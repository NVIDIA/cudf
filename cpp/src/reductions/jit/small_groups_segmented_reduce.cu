/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#include <jit/column_accessor.cuh>
#include <jit/column_device_view_wrappers.cuh>
#include <jit/type_list.cuh>
#include <reductions/jit/small_groups_segmented_reduce.cuh>

#pragma nv_hdrstop  // Generated kernel configuration and UDF declarations vary per compilation.

// clang-format off
#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/reduce_udf.cuh>
// clang-format on

namespace cudf {

extern "C" __global__ void cudf_kernel_entry(
  size_type num_segments,
  size_type const* __restrict__ offsets,
  column_device_view_core const* __restrict__ inputs,
  mutable_column_device_view_core const* __restrict__ outputs,
  errc* __restrict__ segment_errors)
{
  CUDF_KERNEL_INSTANCE(num_segments,
                       offsets,
                       inputs,
                       outputs,
                       segment_errors,
                       CUDF_DISPATCH_REDUCE_INIT,
                       CUDF_DISPATCH_REDUCE_FILTER,
                       CUDF_DISPATCH_REDUCE_UPDATE,
                       CUDF_DISPATCH_REDUCE_MERGE,
                       CUDF_DISPATCH_REDUCE_FINALIZE);
}

}  // namespace cudf
