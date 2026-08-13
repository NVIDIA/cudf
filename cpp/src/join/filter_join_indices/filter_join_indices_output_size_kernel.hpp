/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "join/filter_join_indices/filter_join_indices_kernel.hpp"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>

namespace cudf::detail {

/**
 * @brief Launches a kernel that fills the per-output contribution counts for `filter_join_indices`.
 *
 * The total output size is the sum of `output_counts`. Its layout depends on the join kind:
 * - INNER_JOIN: per input pair, `1` if the predicate passes and `0` otherwise.
 * - FULL_JOIN: per input pair, `1` for a preserved pair and `2` for a failed valid pair (which
 *   splits into `(left, JoinNoMatch)` and `(JoinNoMatch, right)`).
 * - LEFT_JOIN: per left row, the number of passing pairs (accumulated atomically). The host floors
 *   empty rows to `1` afterwards to account for the synthetic `(left, JoinNoMatch)` entry, so the
 *   buffer must be zero-initialized before the launch.
 *
 * @tparam has_nulls Indicates whether the expression may evaluate to null
 * @tparam has_complex_type Indicates whether the expression may contain complex types
 *
 * @param[in] left_table Device view of the left table
 * @param[in] right_table Device view of the right table
 * @param[in] left_indices Device span of left table indices
 * @param[in] right_indices Device span of right table indices
 * @param[in] device_expression_data Device data required to evaluate the expression
 * @param[in] config Grid configuration for kernel launch
 * @param[in] shmem_per_block Amount of shared memory to allocate per block
 * @param[in] join_kind The join kind. Must be INNER_JOIN, LEFT_JOIN, or FULL_JOIN.
 * @param[out] output_counts Per-output contribution counts described above. Sized to
 *             `left_indices.size()` for INNER_JOIN and FULL_JOIN, and to `left_table.num_rows()`
 *             (zero-initialized) for LEFT_JOIN.
 * @param[in] stream CUDA stream on which to launch the kernel
 */
template <bool has_nulls, bool has_complex_type>
void launch_filter_output_size_kernel(
  cudf::table_device_view const& left_table,
  cudf::table_device_view const& right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::detail::grid_1d const& config,
  std::size_t shmem_per_block,
  cudf::join_kind join_kind,
  cudf::size_type* output_counts,
  rmm::cuda_stream_view stream);

}  // namespace cudf::detail
