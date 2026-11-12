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
                    // it. Each UDF will have a different operation_udf.cuh generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
// These headers select the kernel instance and declare the UDF signature.
#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/operation_udf.cuh>
// clang-format on

using transform_type = CUDF_UDF_TYPE;
extern "C" __device__ transform_type cudf_transform_udf;

// The entry point for the JIT compiled kernel. This is the C-ABI function that will be used to
// retrieve the `CuFunction` for the kernel from the compiled module. This is because we don't want
// to track the scope-dependent C++ mangled name of the kernel, and can just use a fixed name to
// retrieve the `CuFunction` of the kernel.
// A C++-mangled symbol has ambiguous and complex resolution rules, and can change based on the
// scope of the function, the types of the arguments, and other factors that will not be known until
// after compilation. By using a fixed C-ABI symbol name for the kernel entry point, we can avoid
// these issues and ensure that we can always retrieve the correct `CuFunction` for the kernel
// regardless of the context in which it was compiled or used.
extern "C" __global__ void cudf_kernel_entry(
  cudf::size_type row_size,
  cudf::bitmask_type const* __restrict__ stencil,
  cudf::column_device_view_core const* __restrict__ input_cols,
  cudf::mutable_column_device_view_core const* __restrict__ output_cols,
  cudf::error_int* __restrict__ max_error)
{
  CUDF_KERNEL_INSTANCE(row_size,
                       stencil,
                       input_cols,
                       output_cols,
                       max_error,
                       [](cudf::size_type row_index, auto... args) {
#if CUDF_UDF_INDEXED
                         return cudf_transform_udf(row_index, args...);
#else
      return cudf_transform_udf(args...);
#endif
                       });
}
