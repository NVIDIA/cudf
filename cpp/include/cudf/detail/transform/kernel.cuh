/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/transform/column_accessor.cuh>
#include <cudf/detail/transform/column_device_view_wrappers.cuh>
#include <cudf/detail/transform/sync.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/errc.hpp>
#include <cudf/type_list.cuh>
#include <cudf/utilities/bit.hpp>

#include <cuda/atomic>
#include <cuda/std/algorithm>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace cudf::detail {
/// @brief The generic transform kernel. Supports all types and nullability combinations.
template <bool is_null_aware,
          bool has_user_data,
          typename InputAccessors,
          typename OutputAccessors,
          typename Function>
__device__ void transform_kernel(size_type row_size,
                                 bitmask_type const* __restrict__ stencil,
                                 void* __restrict__ user_data,
                                 column_device_view_core const* __restrict__ input_cols,
                                 mutable_column_device_view_core const* __restrict__ output_cols,
                                 int32_t* __restrict__ max_error,
                                 Function function)
{
  auto start        = detail::grid_1d::global_thread_id();
  auto stride       = detail::grid_1d::grid_stride();
  auto thread_error = errc::SUCCESS;

  auto operation = [&]<typename Args>(thread_index_type row, Args args) {
    // Check the actual accessor argument types, including dictionary values and optionals.
    auto func = [&](auto... a) {
      static_assert(cuda::std::is_invocable_v<Function, decltype(a)...>,
                    "Transform function is not invocable with its descriptor argument types");
      if constexpr (!cuda::std::is_void_v<decltype(function(a...))>) {
        return static_cast<cudf::errc>(function(a...));
      } else {
        (void)function(a...);
        return errc::SUCCESS;
      }
    };

    if constexpr (has_user_data) {
      return cuda::std::apply(func, cuda::std::tuple_cat(cuda::std::tuple{user_data, row}, args));
    } else {
      return cuda::std::apply(func, args);
    }
  };

  if constexpr (!is_null_aware) {
    for (auto row = start; row < row_size; row += stride) {
      if (stencil != nullptr && !bit_is_set(stencil, row)) { continue; }

      auto ins = InputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::element(input_cols, row)...}; });

      auto outs = OutputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::output_arg(output_cols, row)...}; });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto row_error = operation(row, cuda::std::tuple_cat(out_ptrs, ins));

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, row, cuda::std::get<A::index>(outs)), ...);
      });

      thread_error = cuda::std::max(thread_error, row_error);
    }
  } else {
    // Keep every lane in a warp on the same loop iteration when writing validity.
    auto warp_padded_size = util::round_up_safe<thread_index_type>(row_size, detail::warp_size);

    for (auto row = start; row < warp_padded_size; row += stride) {
      auto active_mask = __ballot_sync(0xffff'ffffu, row < row_size);
      if (row >= row_size) { continue; }

      auto ins = InputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::nullable_element(input_cols, row)...}; });

      auto outs = OutputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::null_output_arg(output_cols, row)...}; });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto row_error = operation(row, cuda::std::tuple_cat(out_ptrs, ins));

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, row, *cuda::std::get<A::index>(outs)), ...);
        (warp_compact_validity<A>(
           active_mask, output_cols, row, cuda::std::get<A::index>(outs).has_value()),
         ...);
      });

      thread_error = cuda::std::max(thread_error, row_error);
    }
  }

  // early exit if no error occurred
  if (thread_error == errc::SUCCESS) { return; }

  cuda::atomic_ref ref(*max_error);
  ref.fetch_max(static_cast<int32_t>(thread_error), cuda::std::memory_order_relaxed);
}

}  // namespace cudf::detail
