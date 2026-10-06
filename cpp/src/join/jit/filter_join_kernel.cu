/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/join/filter_join_kernel.cuh>
#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/tuple>

#include <jit/column_accessor.cuh>
#include <jit/type_list.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation_udf.cuh generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
// This header is an inlined header that defines the GENERIC_JOIN_FILTER_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/operation_udf.cuh>
// clang-format on

namespace cudf::join::jit {

template <bool has_user_data, bool is_null_aware, typename Accessors>
__device__ void filter_join_kernel(cudf::size_type num_rows,
                                   cudf::size_type const* __restrict__ left_indices,
                                   cudf::size_type const* __restrict__ right_indices,
                                   cudf::column_device_view_core const* __restrict__ columns,
                                   bool* __restrict__ predicate_results,
                                   void* __restrict__ user_data)
{
  auto function = [](auto... args) { (void)GENERIC_JOIN_FILTER_OP(args...); };
  cudf::detail::filter_join_kernel<has_user_data, is_null_aware, Accessors>(
    num_rows, left_indices, right_indices, columns, predicate_results, user_data, function);
}

}  // namespace cudf::join::jit

extern "C" __global__ void cudf_kernel_entry(
  cudf::size_type num_rows,
  cudf::size_type const* __restrict__ left_indices,
  cudf::size_type const* __restrict__ right_indices,
  cudf::column_device_view_core const* __restrict__ columns,
  bool* __restrict__ predicate_results,
  void* __restrict__ user_data)
{
  CUDF_KERNEL_INSTANCE(
    num_rows, left_indices, right_indices, columns, predicate_results, user_data);
}
