/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>

namespace cudf::detail {
/**
 * @brief Maximum number of threads per block for filter join indices kernels
 */
size_type constexpr MAX_BLOCK_SIZE = 256;

/**
 * @brief Launches the appropriate filter join indices kernel based on template parameters
 *
 * This function selects and launches the appropriate kernel implementation based on whether
 * the expression may evaluate to null and whether the expression contains complex types.
 * Non-match indices from outer joins are always included without predicate evaluation.
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
 * @param[out] output_flags Device array to mark valid index pairs
 * @param[in] stream CUDA stream on which to launch the kernel
 */
template <bool has_nulls, bool has_complex_type>
void launch_filter_gather_map_kernel(
  cudf::table_device_view const& left_table,
  cudf::table_device_view const& right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::detail::grid_1d const& config,
  std::size_t shmem_per_block,
  bool* output_flags,
  rmm::cuda_stream_view stream);

}  // namespace cudf::detail
