/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"
#include "dispatch.cuh"
#include "hash_csr_kernels.cuh"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <cuda/std/functional>

namespace cudf::detail {

template <typename Hasher>
template <join_kind Join>
std::size_t hash_join<Hasher>::join_size(cudf::table_view const& left,
                                         rmm::cuda_stream_view stream) const
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

  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream);

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto match_counts =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(left.num_rows(), stream, temp_mr);
  auto const row_bitmask = cudf::detail::bitmask_and(left, stream, temp_mr).first;
  auto const valid_rows  = _nulls_equal == null_equality::UNEQUAL
                             ? static_cast<bitmask_type const*>(row_bitmask.data())
                             : nullptr;

  auto count_matches = [&](auto equality, auto hasher) {
    launch_hash_csr_probe_count<Join == join_kind::LEFT_JOIN>(left.num_rows(),
                                                              valid_rows,
                                                              nullptr,
                                                              match_counts.data(),
                                                              nullptr,
                                                              nullptr,
                                                              _impl->map_view(),
                                                              _impl->csr_view(),
                                                              equality,
                                                              hasher,
                                                              stream);
  };
  dispatch_join_comparator(
    _right, left, _preprocessed_right, preprocessed_left, _has_nulls, _nulls_equal, count_matches);
  auto const output_size = cudf::detail::reduce(
    match_counts.begin(), match_counts.end(), std::int64_t{0}, cuda::std::plus<>{}, stream);
  CUDF_EXPECTS(output_size >= 0, "Join output size overflowed", std::overflow_error);
  return static_cast<std::size_t>(output_size);
}

template <typename Hasher>
template <join_kind Join>
std::size_t hash_join<Hasher>::join_size(cudf::table_view const& left,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr) const
{
  static_assert(Join == join_kind::FULL_JOIN);

  CUDF_FUNC_RANGE();
  if (_is_empty) { return left.num_rows(); }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(left),
               "Left table has nulls while right table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream);
  auto match_counts =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(left.num_rows(), stream, mr);
  auto matched_slots =
    cudf::detail::make_zeroed_device_uvector_async<std::uint32_t>(_impl->capacity, stream, mr);
  auto matched_build_rows = cudf::detail::device_scalar<unsigned long long>(0, stream, mr);
  auto const row_bitmask  = cudf::detail::bitmask_and(left, stream, mr).first;
  auto const valid_rows   = _nulls_equal == null_equality::UNEQUAL
                              ? static_cast<bitmask_type const*>(row_bitmask.data())
                              : nullptr;

  auto count_matches = [&](auto equality, auto hasher) {
    launch_hash_csr_probe_count<true>(left.num_rows(),
                                      valid_rows,
                                      nullptr,
                                      match_counts.data(),
                                      matched_slots.data(),
                                      matched_build_rows.data(),
                                      _impl->map_view(),
                                      _impl->csr_view(),
                                      equality,
                                      hasher,
                                      stream);
  };
  dispatch_join_comparator(
    _right, left, _preprocessed_right, preprocessed_left, _has_nulls, _nulls_equal, count_matches);

  auto const left_output_size = cudf::detail::reduce(
    match_counts.begin(), match_counts.end(), std::int64_t{0}, cuda::std::plus<>{}, stream);
  auto const matched_right_rows = matched_build_rows.value(stream);
  CUDF_EXPECTS(left_output_size >= 0, "Join output size overflowed", std::overflow_error);
  auto const output_size = static_cast<std::uint64_t>(left_output_size) +
                           static_cast<std::uint64_t>(_right.num_rows()) - matched_right_rows;
  CUDF_EXPECTS(output_size <= std::numeric_limits<std::size_t>::max(),
               "Join output size overflowed",
               std::overflow_error);
  return static_cast<std::size_t>(output_size);
}

}  // namespace cudf::detail
