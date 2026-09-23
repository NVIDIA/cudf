/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "join_common_utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <optional>

namespace cudf {
namespace detail {

/**
 * @brief Computes the join operation between two tables and returns the
 * output indices of left and right table as a combined table
 *
 * @param left  Table of left columns to join
 * @param right Table of right  columns to join
 * tables have been flipped, meaning the output indices should also be flipped
 * @param binary_predicate The predicate expression to evaluate
 * @param JoinKind The type of join to be performed
 * @param output_size Optional pre-computed output size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device memory
 *
 * @return Join output indices vector pair
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_join(table_view const& left,
                 table_view const& right,
                 ast::expression const& binary_predicate,
                 join_kind JoinKind,
                 std::optional<std::size_t> output_size,
                 cuda::stream_ref stream,
                 rmm::device_async_resource_ref mr);

/**
 * @brief Computes the size of a join operation between two tables without
 * materializing the result and returns the total size value.
 *
 * @param left  Table of left columns to join
 * @param right Table of right  columns to join
 * tables have been flipped, meaning the output indices should also be flipped
 * @param binary_predicate The predicate expression to evaluate
 * @param JoinKind The type of join to be performed
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device memory
 *
 * @return Join output indices vector pair
 */
std::size_t compute_conditional_join_output_size(table_view const& left,
                                                 table_view const& right,
                                                 ast::expression const& binary_predicate,
                                                 join_kind JoinKind,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
