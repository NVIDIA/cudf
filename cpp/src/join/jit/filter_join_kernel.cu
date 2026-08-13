/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/tuple>

#include <jit/column_accessor.cuh>
#include <jit/span.cuh>
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

// TODO: Create a JIT-compatible header for JoinNoMatch to avoid this duplication.
// This must match the definition in cudf/join/join.hpp
constexpr cudf::size_type JoinNoMatch = cuda::std::numeric_limits<cudf::size_type>::min();

template <bool has_user_data, typename... T>
__device__ void execute_predicate_op(void* user_data,
                                     size_type row_index,
                                     cuda::std::tuple<T...> args)
{
  if constexpr (has_user_data) {
    cuda::std::apply(
      [&](auto&&... args) { (void)GENERIC_JOIN_FILTER_OP(user_data, row_index, args...); }, args);
  } else {
    cuda::std::apply([&](auto&&... args) { (void)GENERIC_JOIN_FILTER_OP(args...); }, args);
  }
}

template <bool has_user_data, bool is_null_aware, typename Accessors>
__device__ void filter_join_kernel(cudf::size_type num_rows,
                                   cudf::size_type const* __restrict__ left_indices,
                                   cudf::size_type const* __restrict__ right_indices,
                                   cudf::column_device_view_core const* __restrict__ columns,
                                   bool* __restrict__ predicate_results,
                                   void* __restrict__ user_data)
{
  auto const start  = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  for (auto i = start; i < num_rows; i += stride) {
    // Skip if either index is JoinNoMatch
    if (left_indices[i] == JoinNoMatch || right_indices[i] == JoinNoMatch) {
      predicate_results[i] = false;
      continue;
    }

    cudf::size_type const* indices[] = {left_indices, right_indices};

    // Each accessor receives both tables and both indices, and internally selects
    // the appropriate table based on whether it's a left or right accessor.
    if constexpr (is_null_aware) {
      // Null-aware path: pass optional<T> inputs, get optional<bool> result
      cuda::std::optional<bool> result{false};
      auto inputs = Accessors::map([&]<typename... A>() {
        return cuda::std::tuple{A::nullable_element(columns, indices[A::table_index][i])...};
      });
      execute_predicate_op<has_user_data>(
        user_data, i, cuda::std::tuple_cat(cuda::std::tuple{&result}, inputs));
      predicate_results[i] = result.has_value() && result.value();
    } else {
      // Non-null-aware path: if any input is null, predicate is false
      auto any_null = Accessors::map(
        [&]<typename... A>() { return (A::is_null(columns, indices[A::table_index][i]) || ...); });
      if (any_null) {
        predicate_results[i] = false;
        continue;
      }
      bool result = false;
      auto inputs = Accessors::map([&]<typename... A>() {
        return cuda::std::tuple{A::element(columns, indices[A::table_index][i])...};
      });
      execute_predicate_op<has_user_data>(
        user_data, i, cuda::std::tuple_cat(cuda::std::tuple{&result}, inputs));
      predicate_results[i] = result;
    }
  }
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
