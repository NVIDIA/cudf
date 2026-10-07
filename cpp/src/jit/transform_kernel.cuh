/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/errc.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <cuda/atomic>
#include <cuda/std/algorithm>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <jit/column_accessor.cuh>
#include <jit/sync.cuh>
#include <jit/type_list.cuh>

namespace cudf::detail {

/**
 * @brief Applies a row operation using transform input and output accessors.
 *
 * `operation` is invoked as `operation(row, arguments)`, where `arguments` is a tuple containing
 * output pointers followed by input values.
 */
template <bool IsNullAware,
          typename InputAccessors,
          typename OutputAccessors,
          typename RowOperation>
__device__ void transform_kernel(size_type row_size,
                                 bitmask_type const* __restrict__ stencil,
                                 column_device_view_core const* __restrict__ input_cols,
                                 mutable_column_device_view_core const* __restrict__ output_cols,
                                 int32_t* __restrict__ max_error,
                                 RowOperation&& operation)
{
  auto const start  = grid_1d::global_thread_id();
  auto const stride = grid_1d::grid_stride();
  auto thread_error = errc::SUCCESS;

  if constexpr (!IsNullAware) {
    for (auto row = start; row < row_size; row += stride) {
      if (stencil != nullptr && !bit_is_set(stencil, row)) { continue; }

      auto ins = InputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::element(input_cols, row)...}; });

      auto outs = OutputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::output_arg(output_cols, row)...}; });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto const row_error = operation(row, cuda::std::tuple_cat(out_ptrs, ins));

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, row, cuda::std::get<A::index>(outs)), ...);
      });

      thread_error = cuda::std::max(thread_error, row_error);
    }
  } else {
    // Keep every lane in a warp on the same loop iteration when writing validity.
    auto const warp_padded_size =
      util::round_up_safe<thread_index_type>(row_size, detail::warp_size);

    for (auto row = start; row < warp_padded_size; row += stride) {
      auto const active_mask = __ballot_sync(0xffff'ffffu, row < row_size);
      if (row >= row_size) { continue; }

      auto ins = InputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::nullable_element(input_cols, row)...}; });

      auto outs = OutputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::null_output_arg(output_cols, row)...}; });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto const row_error = operation(row, cuda::std::tuple_cat(out_ptrs, ins));

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, row, *cuda::std::get<A::index>(outs)), ...);
        (jit::warp_compact_validity<A>(
           active_mask, output_cols, row, cuda::std::get<A::index>(outs).has_value()),
         ...);
      });

      thread_error = cuda::std::max(thread_error, row_error);
    }
  }

  if (thread_error == errc::SUCCESS) { return; }

  cuda::atomic_ref ref(*max_error);
  ref.fetch_max(static_cast<int32_t>(thread_error), cuda::std::memory_order_relaxed);
}

}  // namespace cudf::detail
