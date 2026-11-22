/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#include <jit/column_accessor.cuh>
#include <jit/type_list.cuh>
#include <reductions/jit/small_groups_segmented_reduce.cuh>

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

__device__ errc cudf_all_true(size_type, bool* selected)
{
  *selected = true;
  return errc::SUCCESS;
}

__device__ errc cudf_all_true_indexed(size_type segment, size_type, bool* selected)
{
  return cudf_all_true(segment, selected);
}

static_assert(sizeof(CUDF_REDUCE_STATE) == CUDF_REDUCE_STATE_SIZE &&
                alignof(CUDF_REDUCE_STATE) == CUDF_REDUCE_STATE_ALIGNMENT,
              "Fatal error: reduction state layout mismatch");

__global__ void cudf_kernel_entry(size_type num_segments,
                                  size_type const* __restrict__ offsets,
                                  column_device_view_core const* __restrict__ inputs,
                                  column_device_view_core const* __restrict__ filter_inputs,
                                  mutable_column_device_view_core const* __restrict__ outputs,
                                  errc* __restrict__ segment_errors)
{
  CUDF_KERNEL_INSTANCE(
    num_segments,
    offsets,
    inputs,
    filter_inputs,
    outputs,
    segment_errors,
    cudf_reduce_init,
    [](size_type segment, size_type row, auto... args) {
#if CUDF_REDUCE_FILTER_INDEXED
      return cudf_reduce_filter(segment, row, args...);
#else
      return cudf_reduce_filter(segment, args...);
#endif
    },
    [](size_type segment, size_type row, auto... args) {
#if CUDF_REDUCE_UPDATE_INDEXED
      return cudf_reduce_update(segment, row, args...);
#else
      return cudf_reduce_update(segment, args...);
#endif
    },
    cudf_reduce_merge,
    cudf_reduce_finalize);
}

}  // extern "C"

}  // namespace cudf
