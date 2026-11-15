/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/column_accessor.cuh"
#include "jit/type_list.cuh"
#include "reductions/jit/reduce_kernel.cuh"

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#pragma nv_hdrstop  // Generated kernel configuration and UDF declarations vary per compilation.

// clang-format off
#include <cudf/detail/kernel_instance.cuh>
// clang-format on

namespace cudf {

extern "C" {

using reduce_init_udf_type = CUDF_REDUCE_INIT_UDF_TYPE;

__device__ reduce_init_udf_type cudf_reduce_init;

using reduce_filter_udf_type = CUDF_REDUCE_FILTER_UDF_TYPE;

__device__ reduce_filter_udf_type cudf_reduce_filter;

using reduce_update_udf_type = CUDF_REDUCE_UPDATE_UDF_TYPE;

__device__ reduce_update_udf_type cudf_reduce_update;

using reduce_merge_udf_type = CUDF_REDUCE_MERGE_UDF_TYPE;

__device__ reduce_merge_udf_type cudf_reduce_merge;

using reduce_finalize_udf_type = CUDF_REDUCE_FINALIZE_UDF_TYPE;

__device__ reduce_finalize_udf_type cudf_reduce_finalize;

__global__ void cudf_kernel_entry(jit::reduce::phase phase,
                                  size_type num_rows,
                                  column_device_view_core const* __restrict__ inputs,
                                  column_device_view_core const* __restrict__ filter_inputs,
                                  mutable_column_device_view_core const* __restrict__ outputs,
                                  void const* __restrict__ partials,
                                  void* __restrict__ results,
                                  void* __restrict__ scratch,
                                  errc* __restrict__ error,
                                  bool finalize_inline)
{
  CUDF_KERNEL_INSTANCE(
    phase,
    num_rows,
    inputs,
    filter_inputs,
    outputs,
    static_cast<CUDF_DISPATCH_REDUCE_STATE const*>(partials),
    static_cast<CUDF_DISPATCH_REDUCE_STATE*>(results),
    static_cast<CUDF_DISPATCH_REDUCE_STATE*>(scratch),
    error,
    finalize_inline,
    cudf_reduce_init,
    [](size_type row, auto... args) {
#if CUDF_REDUCE_FILTER_INDEXED
      return cudf_reduce_filter(row, args...);
#else
      return cudf_reduce_filter(args...);
#endif
    },
    [](size_type row, auto... args) {
#if CUDF_REDUCE_UPDATE_INDEXED
      return cudf_reduce_update(row, args...);
#else
      return cudf_reduce_update(args...);
#endif
    },
    cudf_reduce_merge,
    cudf_reduce_finalize);
}
}

}  // namespace cudf
