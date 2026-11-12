/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <cuda/stream>

namespace cudf::detail {
/**
 * @brief Maximum number of threads per block for compute column kernels
 */
size_type constexpr MAX_BLOCK_SIZE = 128;

/**
 * @brief Launches the appropriate compute column kernel based on template parameters
 *
 * This function selects and launches the appropriate kernel implementation based on whether
 * the input contains nulls and whether the input contains complex types.
 *
 * @tparam HasNull Indicates whether the input contains any null value.
 * @tparam HasComplexType Indicates whether the input may contain complex types
 *
 * @param table_device Device view of the input table
 * @param device_expression_data Device data required to evaluate the expression
 * @param mutable_output_device Device view of the output column to be populated
 * @param config Grid configuration for kernel launch
 * @param shmem_per_block Amount of shared memory to allocate per block
 * @param stream CUDA stream on which to launch the kernel
 */
template <bool HasNull, bool HasComplexType>
void launch_compute_column_kernel(table_device_view const& table_device,
                                  ast::detail::expression_device_view device_expression_data,
                                  mutable_column_device_view& mutable_output_device,
                                  cudf::detail::grid_1d const& config,
                                  size_t shmem_per_block,
                                  cuda::stream_ref stream);
}  // namespace cudf::detail
