/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>

#include <cuco/pair.cuh>

#include <memory>
#include <utility>

namespace cudf::detail {

/**
 * @brief Equality comparator for cuco hash table probing with row-level equality.
 */
template <typename Equal>
class pair_equal {
 public:
  pair_equal(Equal check_row_equality) : _check_row_equality{std::move(check_row_equality)} {}

  __device__ __forceinline__ bool operator()(
    cuco::pair<hash_value_type, size_type> const& lhs,
    cuco::pair<hash_value_type, size_type> const& rhs) const noexcept
  {
    using detail::row::lhs_index_type;
    using detail::row::rhs_index_type;

    return lhs.first == rhs.first and
           _check_row_equality(lhs_index_type{lhs.second}, rhs_index_type{rhs.second});
  }

 private:
  Equal _check_row_equality;
};

/**
 * @brief Extracts the right-side row index from a cuco hash table slot.
 */
struct output_fn {
  __device__ constexpr cudf::size_type operator()(
    cuco::pair<hash_value_type, cudf::size_type> const& slot) const
  {
    return slot.second;
  }
};

/**
 * @brief Extracts a right-side row index and marks it as matched.
 *
 * This is used while retrieving a full join to build the right-side match set without a
 * subsequent pass over the join output.
 */
struct mark_matched_output_fn {
  size_type* right_matches;
  size_type right_table_num_rows;

  __device__ cudf::size_type operator()(
    cuco::pair<hash_value_type, cudf::size_type> const& slot) const
  {
    auto const index = slot.second;
    if (index >= 0 && index < right_table_num_rows) { right_matches[index] = 1; }
    return index;
  }
};

/**
 * @brief Equality comparator for cuco hash table probing with primitive row equality.
 */
class primitive_pair_equal {
 public:
  primitive_pair_equal(cudf::detail::row::primitive::row_equality_comparator check_row_equality)
    : _check_row_equality{std::move(check_row_equality)}
  {
  }

  __device__ __forceinline__ bool operator()(
    cuco::pair<hash_value_type, size_type> const& lhs,
    cuco::pair<hash_value_type, size_type> const& rhs) const noexcept
  {
    return lhs.first == rhs.first and _check_row_equality(lhs.second, rhs.second);
  }

 private:
  cudf::detail::row::primitive::row_equality_comparator _check_row_equality;
};

template <typename Fn>
decltype(auto) dispatch_join_comparator(
  table_view const& right_table,
  table_view const& left_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  bool has_nulls,
  null_equality compare_nulls,
  Fn&& fn)
{
  auto const left_nulls = cudf::nullate::DYNAMIC{has_nulls};

  if (cudf::detail::is_primitive_row_op_compatible(right_table)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{left_nulls, preprocessed_left};
    auto const d_equal  = cudf::detail::row::primitive::row_equality_comparator{
      left_nulls, preprocessed_left, preprocessed_right, compare_nulls};
    return std::forward<Fn>(fn)(primitive_pair_equal{d_equal}, d_hasher);
  }

  auto const d_hasher =
    cudf::detail::row::hash::row_hasher{preprocessed_left}.device_hasher(left_nulls);
  auto const row_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_left, preprocessed_right};

  if (cudf::detail::has_nested_columns(left_table)) {
    auto const d_equal = row_comparator.equal_to<true>(left_nulls, compare_nulls);
    return std::forward<Fn>(fn)(pair_equal{d_equal}, d_hasher);
  }

  auto const d_equal = row_comparator.equal_to<false>(left_nulls, compare_nulls);
  return std::forward<Fn>(fn)(pair_equal{d_equal}, d_hasher);
}

}  // namespace cudf::detail
