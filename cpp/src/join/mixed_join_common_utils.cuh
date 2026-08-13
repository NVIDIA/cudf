/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/types.hpp>

namespace cudf::detail {

using row_hash = cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                            cudf::nullate::DYNAMIC>;

// // This alias is used by mixed_joins, which support only non-nested types
using row_equality = cudf::detail::row::equality::strong_index_comparator_adapter<
  cudf::detail::row::equality::device_row_comparator<false, cudf::nullate::DYNAMIC>>;

/**
 * @brief Base equality comparator for use with cuco set/multiset methods that require expression
 * evaluation.
 *
 * This class provides the common interface and attributes needed for equality
 * comparators that combine row equality checks with conditional expression evaluation.
 * It stores the expression evaluator, thread-local storage, table swap flag, and
 * the row equality comparator used for non-conditional equality checks.
 */
template <bool has_nulls>
struct expression_equality {
  __device__ expression_equality(
    cudf::ast::detail::expression_evaluator<has_nulls> const& evaluator,
    cudf::ast::detail::IntermediateDataType<has_nulls>* thread_intermediate_storage,
    bool const swap_tables,
    row_equality const& equality_probe)
    : evaluator{evaluator},
      thread_intermediate_storage{thread_intermediate_storage},
      swap_tables{swap_tables},
      equality_probe{equality_probe}
  {
  }

  cudf::ast::detail::IntermediateDataType<has_nulls>* thread_intermediate_storage;
  cudf::ast::detail::expression_evaluator<has_nulls> const& evaluator;
  bool const swap_tables;
  row_equality const& equality_probe;
};

}  // namespace cudf::detail
