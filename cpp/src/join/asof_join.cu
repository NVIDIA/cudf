/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "asof_join.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/join/asof_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <memory>
#include <utility>

namespace cudf::experimental::detail {
namespace {

template <typename RowEqual>
struct is_group_start {
  RowEqual row_equal;

  __device__ bool operator()(size_type row) const { return row == 0 || !row_equal(row - 1, row); }
};

template <typename T>
struct backward_probe {
  column_device_view left_on;
  column_device_view right_on;
  size_type const* group_offsets;
  size_type const* group_positions;
  size_type num_groups;

  __device__ size_type operator()(size_type left_row) const
  {
    if (left_on.is_null(left_row)) { return JoinNoMatch; }

    // A null group_positions pointer means an ungrouped join: every row uses group zero.
    auto const group = group_positions ? group_positions[left_row] : size_type{0};
    if (group < 0 || group >= num_groups) { return JoinNoMatch; }

    auto const first = cuda::counting_iterator<size_type>{group_offsets[group]};
    auto const last  = cuda::counting_iterator<size_type>{group_offsets[group + 1]};
    auto const upper = cuda::std::upper_bound(
      first, last, left_on.element<T>(left_row), [this](T const& left_value, size_type right_row) {
        return !right_on.is_null(right_row) && left_value < right_on.element<T>(right_row);
      });
    if (upper == first) { return JoinNoMatch; }
    auto const candidate = *(upper - 1);
    return right_on.is_null(candidate) ? JoinNoMatch : candidate;
  }
};

struct probe_dispatch {
  column_device_view left_on;
  column_device_view right_on;
  size_type const* group_offsets;
  size_type const* group_positions;
  size_type num_groups;
  size_type* output;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref temp_mr;

  template <typename T>
    requires(cudf::is_integral_not_bool<T>() || cudf::is_chrono<T>())
  void operator()() const
  {
    thrust::transform(
      rmm::exec_policy_nosync(stream, temp_mr),
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator<size_type>{left_on.size()},
      output,
      backward_probe<T>{left_on, right_on, group_offsets, group_positions, num_groups});
  }

  template <typename T>
    requires(not cudf::is_integral_not_bool<T>() && not cudf::is_chrono<T>())
  void operator()() const
  {
    CUDF_FAIL("Unsupported as-of join ordered-key type");
  }
};

}  // namespace

void asof_join_impl::build_groups(cuda::stream_ref stream, cudf::memory_resources mr)
{
  // Ungrouped joins may have an empty _right_by with zero rows, so use _right_on's size.
  auto const num_rows = _right_on.size();
  auto const temp_mr  = mr.get_temporary_mr();

  if (num_rows == 0) {
    _right_group_offsets =
      cuda::device_buffer<size_type>{stream, mr.get_output_mr(), 1, cuda::no_init};
    CUDF_CUDA_TRY(
      cub::DeviceTransform::Fill(_right_group_offsets.begin(), 1, size_type{0}, stream));
    _num_right_groups = 0;
    return;
  }

  if (_right_by.num_columns() == 0) {
    _right_group_offsets =
      cuda::device_buffer<size_type>{stream, mr.get_output_mr(), 2, cuda::no_init};
    CUDF_CUDA_TRY(
      cub::DeviceTransform::Fill(_right_group_offsets.begin(), 1, size_type{0}, stream));
    CUDF_CUDA_TRY(
      cub::DeviceTransform::Fill(_right_group_offsets.begin() + 1, 1, num_rows, stream));
    _num_right_groups = 1;
    return;
  }

  _right_eq_preprocessed =
    cudf::detail::row::equality::preprocessed_table::create(_right_by, stream, mr);
  _right_lex_preprocessed =
    cudf::detail::row::lexicographic::preprocessed_table::create(_right_by, {}, {}, stream, mr);

  cuda::device_buffer<size_type> offsets(
    stream, temp_mr, static_cast<std::size_t>(num_rows) + 1, cuda::no_init);
  // Mark the first row and each change in adjacent by keys, then compact their indices into
  // offsets. Append num_rows so group g occupies [offsets[g], offsets[g + 1]).
  auto const has_nulls  = cudf::has_nested_nulls(_right_by);
  auto const comparator = cudf::detail::row::equality::self_comparator{_right_eq_preprocessed};
  auto const row_equal =
    comparator.equal_to<false>(nullate::DYNAMIC{has_nulls}, null_equality::EQUAL);
  cuda::device_buffer<bool> group_starts(stream, temp_mr, num_rows, cuda::no_init);
  // We benchmarked this two-step approach against DeviceSelect::If and found that
  // combining the predicate with selection was slower, especially for string keys.
  CUDF_CUDA_TRY(cub::DeviceTransform::Transform(cuda::counting_iterator<size_type>{0},
                                                group_starts.begin(),
                                                num_rows,
                                                is_group_start{row_equal},
                                                stream));

  auto mr_property = cuda::std::execution::prop{cuda::mr::get_memory_resource, temp_mr};
  auto env         = cuda::std::execution::env{stream, mr_property};
  cudf::detail::device_scalar<size_type> num_groups{0, stream, temp_mr};
  CUDF_CUDA_TRY(cub::DeviceSelect::Flagged(cuda::counting_iterator<size_type>{0},
                                           group_starts.begin(),
                                           offsets.begin(),
                                           num_groups.data(),
                                           num_rows,
                                           env));

  _num_right_groups = num_groups.value(stream);
  CUDF_CUDA_TRY(
    cub::DeviceTransform::Fill(offsets.begin() + _num_right_groups, 1, num_rows, stream));
  _right_group_offsets = cuda::device_buffer<size_type>{
    stream, mr.get_output_mr(), static_cast<std::size_t>(_num_right_groups) + 1, cuda::no_init};
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(_right_group_offsets.data(),
                                           offsets.data(),
                                           _right_group_offsets.size() * sizeof(size_type),
                                           stream));
}

cuda::device_buffer<size_type> asof_join_impl::probe(table_view const& left_by,
                                                     column_view const& left_on,
                                                     cuda::stream_ref stream,
                                                     cudf::memory_resources mr) const
{
  cuda::device_buffer<size_type> result(stream, mr.get_output_mr(), left_on.size(), cuda::no_init);
  if (left_on.is_empty()) { return result; }

  auto const temp_mr = mr.get_temporary_mr();
  if (_num_right_groups == 0) {
    thrust::fill(
      rmm::exec_policy_nosync(stream, temp_mr), result.begin(), result.end(), JoinNoMatch);
    return result;
  }

  // Grouped probes map each left row to a right group, or JoinNoMatch.
  std::unique_ptr<cuda::device_buffer<size_type>> group_positions;
  if (left_by.num_columns() > 0) {
    group_positions = std::make_unique<cuda::device_buffer<size_type>>(
      stream, temp_mr, left_on.size(), cuda::no_init);
    auto const has_nulls = cudf::has_nested_nulls(_right_by) || cudf::has_nested_nulls(left_by);
    // Binary-search group representatives for the first by key not less than the left key.
    auto left_lex = cudf::detail::row::lexicographic::preprocessed_table::create(
      left_by, {}, {}, stream, mr.as_temporary());
    auto const row_less = cudf::detail::row::lexicographic::two_table_comparator{
      _right_lex_preprocessed, std::move(left_lex)};
    auto const right_group_rows = cuda::transform_iterator(
      _right_group_offsets.begin(),
      cuda::proclaim_return_type<cudf::detail::row::lhs_index_type>(
        [] __device__(size_type row) { return cudf::detail::row::lhs_index_type{row}; }));
    thrust::lower_bound(rmm::exec_policy_nosync(stream, temp_mr),
                        right_group_rows,
                        right_group_rows + _num_right_groups,
                        cudf::detail::row::rhs_iterator(0),
                        cudf::detail::row::rhs_iterator(0) + left_on.size(),
                        group_positions->begin(),
                        row_less.less<false>(nullate::DYNAMIC{has_nulls}));

    // lower_bound gives only a candidate: reject missing/unequal groups and null grouping keys.
    auto left_eq =
      cudf::detail::row::equality::preprocessed_table::create(left_by, stream, mr.as_temporary());
    auto const row_equal =
      cudf::detail::row::equality::two_table_comparator{_right_eq_preprocessed, std::move(left_eq)};
    auto const equal =
      row_equal.equal_to<false>(nullate::DYNAMIC{has_nulls}, null_equality::UNEQUAL);
    thrust::transform(
      rmm::exec_policy_nosync(stream, temp_mr),
      group_positions->begin(),
      group_positions->end(),
      cuda::counting_iterator<size_type>{0},
      group_positions->begin(),
      [group_rows = _right_group_offsets.data(), num_groups = _num_right_groups, equal] __device__(
        size_type group, size_type left_row) {
        return group < num_groups && equal(cudf::detail::row::lhs_index_type{group_rows[group]},
                                           cudf::detail::row::rhs_index_type{left_row})
                 ? group
                 : JoinNoMatch;
      });
  }

  // Within each matched group, find the last row with right_on <= left_on.
  // Return one right-row index (or JoinNoMatch) per left row, preserving left input order.
  auto const left_on_device  = column_device_view::create(left_on, stream, temp_mr);
  auto const right_on_device = column_device_view::create(_right_on, stream, temp_mr);
  cudf::type_dispatcher(left_on.type(),
                        probe_dispatch{*left_on_device,
                                       *right_on_device,
                                       _right_group_offsets.data(),
                                       group_positions ? group_positions->data() : nullptr,
                                       _num_right_groups,
                                       result.data(),
                                       stream,
                                       temp_mr});
  return result;
}

}  // namespace cudf::experimental::detail
