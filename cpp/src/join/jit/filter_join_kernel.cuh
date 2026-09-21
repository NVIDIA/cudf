/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>

namespace cudf::join::jit {

/**
 * @brief JIT kernel for filtering join indices based on predicate
 *
 * @tparam has_user_data Whether the predicate function requires user data
 * @tparam is_null_aware Whether the expression needs input validity as part of its computation
 * @tparam Accessors type list of accessors for columns used in the predicate
 * @param num_rows Number of rows to process
 * @param left_indices Device pointer to left table indices
 * @param right_indices Device pointer to right table indices
 * @param columns Device view of all columns involved in the predicate
 * @param predicate_results Output array for predicate evaluation results
 * @param user_data Optional user data for predicate function
 */
template <bool has_user_data, null_aware is_null_aware, typename Accessors>
CUDF_KERNEL void filter_join_kernel(cudf::size_type num_rows,
                                    cudf::size_type const* __restrict__ left_indices,
                                    cudf::size_type const* __restrict__ right_indices,
                                    cudf::column_device_view_core const* __restrict__ columns,
                                    bool* __restrict__ predicate_results,
                                    void* __restrict__ user_data);

}  // namespace cudf::join::jit
