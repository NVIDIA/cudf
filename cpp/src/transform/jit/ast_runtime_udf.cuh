/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/errc.hpp>

#include <cuda/std/optional>

#include <transform/jit/ast_runtime_descriptor.hpp>

#pragma nv_hdrstop

namespace cudf::jit::ast_runtime {

template <typename T, bool Nullable>
__device__ auto load(input_descriptor const& column, cuda::std::int32_t row)
{
  auto const index = column.offset + (column.scalar ? 0 : row);
  if constexpr (Nullable) {
    if (column.null_mask != nullptr && !(column.null_mask[index / 32] & (1u << (index % 32)))) {
      return cuda::std::optional<T>{};
    }
    return cuda::std::optional<T>{static_cast<T const*>(column.data)[index]};
  } else {
    return static_cast<T const*>(column.data)[index];
  }
}

template <typename T>
__device__ void store(output_descriptor const& column,
                      cuda::std::int32_t row,
                      cuda::std::uint32_t,
                      T value)
{
  static_cast<T*>(column.data)[row] = value;
}

template <typename T>
__device__ void store(output_descriptor const& column,
                      cuda::std::int32_t row,
                      cuda::std::uint32_t active_mask,
                      cuda::std::optional<T> value)
{
  static_cast<T*>(column.data)[row] = value.value_or(T{});
  auto const validity               = __ballot_sync(active_mask, value.has_value());
  if (column.null_mask != nullptr && (row % 32) == (__ffs(active_mask) - 1)) {
    column.null_mask[row / 32] = validity;
  }
}

#include <cudf/detail/operation_udf.cuh>

}  // namespace cudf::jit::ast_runtime

// A named global expression keeps the device entry alive until the driver fragment is linked.
extern "C" __global__ void cudf_ast_runtime_entry()
{
  auto const operation = &cudf::jit::ast_runtime::transform_row;
  asm volatile("" : : "l"(operation));
}
