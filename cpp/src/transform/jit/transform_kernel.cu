/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#include <jit/column_accessor.cuh>
#include <jit/type_list.cuh>
#include <transform/jit/transform_kernel.cuh>

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
/// It is a C-ABI function that will be used as a stable identifier for the UDF.
/// This allows re-using the same kernel across different UDFs with different names without
/// recompiling the kernel.
__device__ transform_udf_type cudf_transform_udf;

/// @brief The entry point for the kernel.
/// It is a C-ABI function that will be used to retrieve the `CuFunction` for the kernel from
/// the compiled module. This workaround is necessary because tracking the scope-dependent C++
/// mangled name of the kernel is complex and error-prone without compiler support. We use
/// this fixed C-ABI symbol to reference the kernel and retrieve the `CuFunction` handle of the
/// kernel.
__global__ void cudf_kernel_entry(size_type num_rows,
                                  bitmask_type const* __restrict__ stencil,
                                  column_device_view_core const* __restrict__ input_cols,
                                  mutable_column_device_view_core const* __restrict__ output_cols,
                                  int32_t* __restrict__ max_error)
{
  CUDF_KERNEL_INSTANCE(
    num_rows, stencil, input_cols, output_cols, max_error, [](size_type index, auto... args) {
#if CUDF_TRANSFORM_UDF_INDEXED
      return cudf_transform_udf(index, args...);
#else
      return cudf_transform_udf(args...);
#endif
    });
}
}

}  // namespace cudf
