/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/errc.hpp>

#include <transform/jit/ast_runtime_descriptor.hpp>

namespace cudf::jit::ast_runtime {

extern "C" __device__ cudf::errc transform_row(cuda::std::int32_t row,
                                               cuda::std::uint32_t active_mask,
                                               input_descriptor const* inputs,
                                               output_descriptor const* outputs);

}  // namespace cudf::jit::ast_runtime

extern "C" __global__ void cudf_kernel_entry(
  cuda::std::int32_t row_size,
  cuda::std::uint32_t const* stencil,
  void*,
  cudf::jit::ast_runtime::input_descriptor const* inputs,
  cudf::jit::ast_runtime::output_descriptor const* outputs,
  cuda::std::int32_t* max_error)
{
  auto const start  = static_cast<cuda::std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto const stride = static_cast<cuda::std::int64_t>(gridDim.x) * blockDim.x;
  // The final warp must reach the active-lane ballot together, even when its last rows are absent.
  auto const padded_rows = (static_cast<cuda::std::int64_t>(row_size) + 31) / 32 * 32;
  auto thread_error      = cudf::errc::SUCCESS;
  for (auto row = start; row < padded_rows; row += stride) {
    auto const active =
      row < row_size && (stencil == nullptr || (stencil[row / 32] & (1u << (row % 32))));
    auto const active_mask = __ballot_sync(0xffffffffu, active);
    if (!active) { continue; }
    auto const error = cudf::jit::ast_runtime::transform_row(
      static_cast<cuda::std::int32_t>(row), active_mask, inputs, outputs);
    if (static_cast<int>(error) > static_cast<int>(thread_error)) { thread_error = error; }
  }
  if (thread_error != cudf::errc::SUCCESS) { atomicMax(max_error, static_cast<int>(thread_error)); }
}
