/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>

#include <memory>
#include <utility>

namespace cudf::detail {

template <typename Equal>
class hash_csr_equal {
 public:
  explicit hash_csr_equal(Equal check_row_equality)
    : _check_row_equality{std::move(check_row_equality)}
  {
  }

  __device__ bool operator()(hash_csr_key_type const& lhs,
                             hash_csr_key_type const& rhs) const noexcept
  {
    using detail::row::lhs_index_type;
    using detail::row::rhs_index_type;
    return _check_row_equality(lhs_index_type{lhs.row}, rhs_index_type{rhs.row});
  }

 private:
  Equal _check_row_equality;
};

class primitive_hash_csr_equal {
 public:
  explicit primitive_hash_csr_equal(
    cudf::detail::row::primitive::row_equality_comparator check_row_equality)
    : _check_row_equality{std::move(check_row_equality)}
  {
  }

  __device__ bool operator()(hash_csr_key_type const& lhs,
                             hash_csr_key_type const& rhs) const noexcept
  {
    return _check_row_equality(lhs.row, rhs.row);
  }

 private:
  cudf::detail::row::primitive::row_equality_comparator _check_row_equality;
};

template <typename Fn>
decltype(auto) dispatch_hash_csr_comparator(
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
    return std::forward<Fn>(fn)(primitive_hash_csr_equal{d_equal}, d_hasher);
  }

  auto const d_hasher =
    cudf::detail::row::hash::row_hasher{preprocessed_left}.device_hasher(left_nulls);
  auto const row_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_left, preprocessed_right};

  if (cudf::detail::has_nested_columns(left_table)) {
    auto const d_equal = row_comparator.equal_to<true>(left_nulls, compare_nulls);
    return std::forward<Fn>(fn)(hash_csr_equal{d_equal}, d_hasher);
  }

  auto const d_equal = row_comparator.equal_to<false>(left_nulls, compare_nulls);
  return std::forward<Fn>(fn)(hash_csr_equal{d_equal}, d_hasher);
}

}  // namespace cudf::detail
