/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "mixed_join_common_utils.cuh"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/types.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>

#include <cstddef>

namespace cudf::detail {

/**
 * @brief Equality comparator for cuco::static_set queries.
 *
 * This equality comparator is designed for use with cuco::static_set's APIs. A
 * probe hit indicates that the hashes of the keys are equal, at which point
 * this comparator checks whether the keys themselves are equal (using the
 * provided equality_probe) and then evaluates the conditional expression
 */
template <bool has_nulls>
struct single_expression_equality : expression_equality<has_nulls> {
  using expression_equality<has_nulls>::expression_equality;

  __device__ __forceinline__ bool operator()(size_type const left_index,
                                             size_type const right_index) const noexcept
  {
    using cudf::detail::row::lhs_index_type;
    using cudf::detail::row::rhs_index_type;

    auto output_dest = cudf::ast::detail::value_expression_result<bool, has_nulls>();
    // Two levels of checks:
    // 1. The contents of the columns involved in the equality condition are equal.
    // 2. The predicate evaluated on the relevant columns (already encoded in the evaluator)
    // evaluates to true.
    if (this->equality_probe(lhs_index_type{left_index}, rhs_index_type{right_index})) {
      // For the AST evaluator, we need to map back to left/right table semantics
      auto const left_table_idx  = this->swap_tables ? right_index : left_index;
      auto const right_table_idx = this->swap_tables ? left_index : right_index;
      this->evaluator.evaluate(output_dest,
                               static_cast<size_type>(left_table_idx),
                               static_cast<size_type>(right_table_idx),
                               0,
                               this->thread_intermediate_storage);
      return (output_dest.is_valid() && output_dest.value());
    }
    return false;
  }
};

/**
 * @brief Equality comparator that composes two row_equality comparators.
 */
struct double_row_equality_comparator {
  row_equality const equality_comparator;
  row_equality const conditional_comparator;

  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) const noexcept
  {
    using detail::row::lhs_index_type;
    using detail::row::rhs_index_type;

    return equality_comparator(lhs_index_type{lhs_row_index}, rhs_index_type{rhs_row_index}) &&
           conditional_comparator(lhs_index_type{lhs_row_index}, rhs_index_type{rhs_row_index});
  }
};

// A CUDA Cooperative Group of 1 thread for the hash set for mixed semi.
auto constexpr DEFAULT_MIXED_SEMI_JOIN_CG_SIZE = 1;

// The hash set type used by mixed_semi_join with the build_table.
using hash_set_type =
  cuco::static_set<size_type,
                   cuco::extent<size_t>,
                   cuda::thread_scope_device,
                   double_row_equality_comparator,
                   cuco::linear_probing<DEFAULT_MIXED_SEMI_JOIN_CG_SIZE, row_hash>,
                   rmm::mr::polymorphic_allocator<char>,
                   cuco::storage<1>>;

// The hash_set_ref_type used by mixed_semi_join kernels for probing.
using hash_set_ref_type = hash_set_type::ref_type<cuco::contains_tag>;

}  // namespace cudf::detail
