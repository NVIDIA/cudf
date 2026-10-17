/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/errc.hpp>
#include <cudf/utilities/bit.hpp>

#include <cuda/std/optional>

#include <jit/sync.cuh>
#include <transform/jit/ast_runtime_descriptor.hpp>

// Keep generated operations outside the reusable precompiled-header prefix.
// https://docs.nvidia.com/cuda/archive/12.9.0/nvrtc/index.html#determining-the-header-stop-point
#pragma nv_hdrstop

namespace cudf::jit::ast_runtime {

template <typename T, bool Nullable>
__device__ auto load(input_descriptor const& column, cuda::std::int32_t row)
{
  auto const index = column.offset + (column.scalar ? 0 : row);
  if constexpr (Nullable) {
    if (!cudf::bit_value_or(column.null_mask, index, true)) { return cuda::std::optional<T>{}; }
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
  // Null positions have no defined value, but maskless outputs still need initialized data.
  if (value.has_value() || column.null_mask == nullptr) {
    static_cast<T*>(column.data)[row] = value.value_or(T{});
  }
  // All active lanes must participate, including lanes whose value store was skipped.
  cudf::jit::warp_compact_validity(active_mask, column.null_mask, row, value.has_value());
}

#include <cudf/detail/operation_udf.cuh>

}  // namespace cudf::jit::ast_runtime

// Registering a global entry makes retention of this otherwise device-only fragment explicit
// across NVRTC versions before the driver is linked. This anchor is never launched.
extern "C" __global__ void cudf_ast_runtime_entry()
{
  auto const operation = &cudf::jit::ast_runtime::transform_row;
  asm volatile("" : : "l"(operation));
}
