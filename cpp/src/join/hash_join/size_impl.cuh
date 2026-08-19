/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"
#include "dispatch.cuh"
#include "join/join_common_utils.cuh"

#include <cudf/detail/nvtx/ranges.hpp>

namespace cudf::detail {

std::size_t get_full_join_size(
  cudf::table_view const& right_table,
  cudf::table_view const& left_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

template <join_kind Join>
std::size_t compute_join_output_size(
  table_view const& right_table,
  table_view const& left_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  cudf::null_equality nulls_equal,
  cuda::stream_ref stream)
{
  static_assert(Join == join_kind::INNER_JOIN || Join == join_kind::LEFT_JOIN);

  if (right_table.num_rows() == 0) {
    return Join == join_kind::INNER_JOIN ? 0 : left_table.num_rows();
  }

  auto const left_table_num_rows = left_table.num_rows();

  return dispatch_join_comparator(
    right_table,
    left_table,
    preprocessed_right,
    preprocessed_left,
    has_nulls,
    nulls_equal,
    [&](auto equality, auto d_hasher) {
      auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});
      if constexpr (Join == join_kind::LEFT_JOIN) {
        return hash_table.count_outer(
          iter, iter + left_table_num_rows, equality, hash_table.hash_function(), stream.get());
      } else {
        return hash_table.count(
          iter, iter + left_table_num_rows, equality, hash_table.hash_function(), stream.get());
      }
    });
}

template <typename Hasher>
template <join_kind Join>
std::size_t hash_join<Hasher>::join_size(cudf::table_view const& left,
                                         cuda::stream_ref stream) const
{
  static_assert(Join == join_kind::INNER_JOIN || Join == join_kind::LEFT_JOIN);

  CUDF_FUNC_RANGE();

  if constexpr (Join == join_kind::INNER_JOIN) {
    if (_is_empty) { return 0; }
  } else {
    if (_is_empty) { return left.num_rows(); }
  }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(left),
               "Left table has nulls while right table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_left = cudf::detail::row::equality::preprocessed_table::create(
    left, stream, cudf::get_current_device_resource_ref());

  return cudf::detail::compute_join_output_size<Join>(_right,
                                                      left,
                                                      _preprocessed_right,
                                                      preprocessed_left,
                                                      _impl->_hash_table,
                                                      _has_nulls,
                                                      _nulls_equal,
                                                      stream);
}

template <typename Hasher>
template <join_kind Join>
std::size_t hash_join<Hasher>::join_size(cudf::table_view const& left,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr) const
{
  static_assert(Join == join_kind::FULL_JOIN);

  CUDF_FUNC_RANGE();

  if (_is_empty) { return left.num_rows(); }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(left),
               "Left table has nulls while right table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_left = cudf::detail::row::equality::preprocessed_table::create(
    left, stream, cudf::get_current_device_resource_ref());

  return cudf::detail::get_full_join_size(_right,
                                          left,
                                          _preprocessed_right,
                                          preprocessed_left,
                                          _impl->_hash_table,
                                          _has_nulls,
                                          _nulls_equal,
                                          stream,
                                          mr);
}

}  // namespace cudf::detail
