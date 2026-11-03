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
#include <reductions/jit/segmented_reduce.cuh>

#pragma nv_hdrstop  // Generated kernel configuration and UDF declarations vary per compilation.

// clang-format off
#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/reduce_udf.cuh>
// clang-format on

namespace cudf {

extern "C" __global__ void cudf_kernel_entry(
  jit::reduce::phase phase,
  size_type num_segments,
  size_type const* __restrict__ offsets,
  column_device_view_core const* __restrict__ inputs,
  mutable_column_device_view_core const* __restrict__ outputs,
  size_type blocks_per_segment,
  void const* __restrict__ partials,
  void* __restrict__ results,
  void* __restrict__ scratch,
  errc* __restrict__ segment_errors,
  bool finalize_inline)
{
  CUDF_KERNEL_INSTANCE(phase,
                       num_segments,
                       offsets,
                       inputs,
                       outputs,
                       blocks_per_segment,
                       static_cast<CUDF_DISPATCH_REDUCE_STATE const*>(partials),
                       static_cast<CUDF_DISPATCH_REDUCE_STATE*>(results),
                       static_cast<CUDF_DISPATCH_REDUCE_STATE*>(scratch),
                       segment_errors,
                       finalize_inline,
                       CUDF_DISPATCH_REDUCE_INIT,
                       CUDF_DISPATCH_REDUCE_FILTER,
                       CUDF_DISPATCH_REDUCE_UPDATE,
                       CUDF_DISPATCH_REDUCE_MERGE,
                       CUDF_DISPATCH_REDUCE_FINALIZE);
}

}  // namespace cudf
