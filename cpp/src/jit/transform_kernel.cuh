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
#include <cuda/std/utility>

#include <jit/column_accessor.cuh>
#include <jit/sync.cuh>
#include <jit/type_list.cuh>

namespace cudf::detail {

template <typename T>
inline constexpr bool is_fixed_width_repr_compatible =
  cudf::is_fixed_width<T>() && cudf::is_rep_layout_compatible<T>();

/**
 * @brief Applies a row operation using transform input and output accessors.
 *
 * `operation` is invoked as `operation(row, output_pointers..., input_values...)`.
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
  constexpr auto all_fixed_width_repr_compatible = []<typename... A>() {
    return (is_fixed_width_repr_compatible<typename A::element_type> && ...);
  };
  constexpr bool direct_access = InputAccessors::map(all_fixed_width_repr_compatible) &&
                                 OutputAccessors::map(all_fixed_width_repr_compatible);

  auto const start  = grid_1d::global_thread_id();
  auto const stride = grid_1d::grid_stride();
  auto thread_error = errc::SUCCESS;

  // `row_index` is intentionally kept wide to reduce register pressure.
  // The row indexes are checked against the row sizes before use.
  // Both branches expand argument packs directly to avoid concatenated tuple types and
  // `cuda::std::apply` machinery, reducing frontend work, especially for JIT compilation.
  if constexpr (!IsNullAware) {
    for (auto row_index = start; row_index < row_size; row_index += stride) {
      if (stencil != nullptr && !bit_is_set(stencil, row_index)) { continue; }

      if constexpr (direct_access) {
        InputAccessors::map([&]<typename... In> {
          OutputAccessors::map([&]<typename... Out> {
            auto const error = operation(
              row_index,
              (Out::column(output_cols).template data<typename Out::element_type>() + row_index)...,
              In::column(input_cols)
                .template data<typename In::element_type>()[In::map_index(row_index)]...);
            thread_error = cuda::std::max(thread_error, error);
          });
        });
      } else {
        OutputAccessors::map([&]<typename... Out> {
          [&](auto... outs) {
            InputAccessors::map([&]<typename... In> {
              auto const error =
                operation(row_index, &outs..., In::element(input_cols, row_index)...);
              (Out::assign(output_cols, row_index, outs), ...);

              thread_error = cuda::std::max(thread_error, error);
            });
          }(Out::output_arg(output_cols, row_index)...);
        });
      }
    }
  } else {
    // Keep every lane in a warp on the same loop iteration when writing validity.
    auto const warp_padded_size =
      util::round_up_safe<thread_index_type>(row_size, detail::warp_size);

    for (auto row_index = start; row_index < warp_padded_size; row_index += stride) {
      auto const active_mask = __ballot_sync(0xffff'ffffu, row_index < row_size);
      if (row_index >= row_size) { continue; }
      OutputAccessors::map([&]<typename... Out> {
        [&](auto... outs) {
          InputAccessors::map([&]<typename... In> {
            auto const error =
              operation(row_index, &outs..., In::nullable_element(input_cols, row_index)...);
            (Out::assign(output_cols, row_index, outs.operator*()), ...);
            (jit::warp_compact_validity<Out>(active_mask, output_cols, row_index, outs.has_value()),
             ...);
            thread_error = cuda::std::max(thread_error, error);
          });
        }(Out::null_output_arg(output_cols, row_index)...);
      });
    }
  }

  if (thread_error == errc::SUCCESS) { return; }

  cuda::atomic_ref ref(*max_error);
  ref.fetch_max(static_cast<int32_t>(thread_error), cuda::std::memory_order_relaxed);
}

}  // namespace cudf::detail
