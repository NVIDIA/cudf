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
                    // this pragma. Each kernel instance will have a different `kernel_instance.cuh`
                    // thus we need to have this pragma before including it to avoid type mismatches
                    // in the PCH cache.

// clang-format off
#include <cudf/detail/kernel_instance.cuh>
// clang-format on

namespace cudf {

extern "C" {

using transform_udf_type = CUDF_TRANSFORM_UDF_TYPE;

/// @brief The user-defined function entry point for the kernel.
/// This is the C-ABI symbol that will be used as a stable identifier for the UDF.
/// This allows re-using the same kernel across different UDFs with different names without
/// recompiling the kernel.
__device__ transform_udf_type cudf_transform_udf;

/// @brief The entry point for the kernel.
/// This is the C-ABI function that will be used to retrieve the `CuFunction` for the kernel from
/// the compiled module. This workaround is necessary because tracking the scope-dependent C++
/// mangled name of the kernel is complex and error-prone without compiler support. We use
/// this fixed C-ABI symbol to reference the kernel and retrieve the `CuFunction` handle of the
/// kernel.
__global__ void cudf_kernel_entry(size_type num_rows,
                                  size_type const* __restrict__ left_indices,
                                  size_type const* __restrict__ right_indices,
                                  column_device_view_core const* __restrict__ columns,
                                  bool* __restrict__ predicate_results,
                                  int32_t* __restrict__ max_error)
{
  CUDF_KERNEL_INSTANCE(num_rows,
                       left_indices,
                       right_indices,
                       columns,
                       predicate_results,
                       max_error,
                       [](size_type left_index, size_type right_index, auto... args) {
#if CUDF_TRANSFORM_UDF_INDEXED
                         return cudf_transform_udf(left_index, right_index, args...);
#else
      return cudf_transform_udf(args...);
#endif
                       });
}
}

}  // namespace cudf
