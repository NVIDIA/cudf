/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mixed_filter_join_common_utils.cuh"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @brief Performs a semi join using the combination of a hash lookup to
 * identify equal rows between one pair of tables and the evaluation of an
 * expression containing an arbitrary expression.
 *
 * This method probes the hash table with each row in the probe table using a
 * custom equality comparator that also checks that the conditional expression
 * evaluates to true between the left/right tables when a match is found
 * between probe and build rows.
 *
 * @tparam block_size The number of threads per block for this kernel
 * @tparam has_nulls Whether or not the inputs may contain nulls.
 *
 * @param[in] has_nulls If the input has nulls
 * @param[in] left_table The left table
 * @param[in] right_table The right table
 * @param[in] probe The table with which to probe the hash table for matches.
 * @param[in] build The table with which the hash table was built.
 * @param[in] equality_probe The equality comparator used when probing the hash table.
 * @param[in] set_ref The hash table device view built from `build`.
 * @param[out] left_table_keep_mask The result of the join operation with "true" element indicating
 * the corresponding index from left table is present in output
 * @param[in] device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] config Kernel launch configuration
 * @param[in] shmem_size_per_block Shared memory size per block
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 */
void launch_mixed_join_semi(bool has_nulls,
                            table_device_view left_table,
                            table_device_view right_table,
                            table_device_view probe,
                            table_device_view build,
                            row_equality const equality_probe,
                            hash_set_ref_type set_ref,
                            cudf::device_span<bool> left_table_keep_mask,
                            cudf::ast::detail::expression_device_view device_expression_data,
                            detail::grid_1d const config,
                            int64_t shmem_size_per_block,
                            cuda::stream_ref stream);

}  // namespace detail

}  // namespace cudf
