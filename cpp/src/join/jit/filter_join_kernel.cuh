/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#include <cuda/std/algorithm.max.h>
#include <cuda/std/limits>
#include <cuda/std/optional>

namespace cudf::detail {

/**
 * @brief Applies a row operation to join candidates using column accessors.
 *
 * `operation` receives left and right row indices, the output pointer, then input values.
 * It decides whether to forward the indices to the UDF.
 */
template <bool is_null_aware, typename Accessors, typename RowOperation>
__device__ void filter_join_kernel(cudf::size_type num_rows,
                                   cudf::size_type const* __restrict__ left_indices,
                                   cudf::size_type const* __restrict__ right_indices,
                                   cudf::column_device_view_core const* __restrict__ columns,
                                   bool* __restrict__ predicate_results,
                                   error_int* __restrict__ max_error,
                                   RowOperation&& operation)
{
  // Keep this sentinel consistent with cudf/join/join.hpp without including its host APIs.
  constexpr auto JoinNoMatch = cuda::std::numeric_limits<cudf::size_type>::min();
  auto const start           = cudf::detail::grid_1d::global_thread_id();
  auto const stride          = cudf::detail::grid_1d::grid_stride();

  auto thread_error = errc::SUCCESS;

  for (auto i = start; i < num_rows; i += stride) {
    // Skip if either index is JoinNoMatch
    if (left_indices[i] == JoinNoMatch || right_indices[i] == JoinNoMatch) {
      predicate_results[i] = false;
      continue;
    }

    cudf::size_type const candidate_index = i;
    cudf::size_type const indices[]       = {left_indices[i], right_indices[i], candidate_index};

    // Table accessors use their source map; extra inputs use the candidate position.
    if constexpr (is_null_aware) {
      // Null-aware path: pass optional<T> inputs, get optional<bool> result
      cuda::std::optional<uint8_t> result{0};
      auto row_error       = Accessors::map([&]<typename... A>() {
        return operation(left_indices[i],
                         right_indices[i],
                         &result,
                         A::nullable_element(columns, indices[A::table_index])...);
      });
      thread_error         = cuda::std::max(thread_error, row_error);
      predicate_results[i] = result.value_or(0) != 0;
    } else {
      // Non-null-aware path: if any input is null, predicate is false
      auto any_null = Accessors::map(
        [&]<typename... A>() { return (A::is_null(columns, indices[A::table_index]) || ...); });
      if (any_null) {
        predicate_results[i] = false;
        continue;
      }
      uint8_t result{0};
      auto row_error       = Accessors::map([&]<typename... A>() {
        return operation(left_indices[i],
                         right_indices[i],
                         &result,
                         A::element(columns, indices[A::table_index])...);
      });
      thread_error         = cuda::std::max(thread_error, row_error);
      predicate_results[i] = result != 0;
    }
  }

  if (thread_error == errc::SUCCESS) { return; }

  atomicMax(max_error, static_cast<error_int>(thread_error));
}

}  // namespace cudf::detail
