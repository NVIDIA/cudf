/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "asof_join.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/join/asof_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <memory>
#include <string>
#include <utility>

namespace cudf {
namespace {

bool is_supported_on_type(data_type type)
{
  return cudf::is_integral_not_bool(type) || cudf::is_timestamp(type) || cudf::is_duration(type);
}

bool is_supported_by_type(data_type type)
{
  return type.id() == type_id::STRING || cudf::is_fixed_width(type);
}

void validate_by_types(table_view const& by)
{
  for (auto const& column : by) {
    CUDF_EXPECTS(is_supported_by_type(column.type()),
                 "As-of join grouping keys must be strings or fixed-width scalar columns");
  }
}

void validate_key_sizes(table_view const& by, column_view const& on, char const* side)
{
  CUDF_EXPECTS(by.num_columns() == 0 || by.num_rows() == on.size(),
               std::string{"As-of join "} + side +
                 " grouping and ordered keys must have the same number of rows");
}

struct right_group_index {
  rmm::device_uvector<size_type> rows;
  rmm::device_uvector<size_type> offsets;
  size_type num_groups;
};

right_group_index build_right_group_index(table_view const& right_by,
                                          size_type num_rows,
                                          cuda::stream_ref stream)
{
  auto const mr    = cudf::get_current_device_resource_ref();
  auto mr_property = cuda::std::execution::prop{cuda::mr::get_memory_resource, mr};
  auto env         = cuda::std::execution::env{cuda::stream_ref{stream.get()}, mr_property};
  rmm::device_uvector<size_type> rows(num_rows, stream, mr);
  rmm::device_uvector<size_type> offsets(static_cast<std::size_t>(num_rows) + 1, stream, mr);

  if (num_rows == 0) {
    CUDF_CUDA_TRY(cub::DeviceTransform::Fill(offsets.begin(), 1, size_type{0}, env));
    offsets.resize(1, stream);
    return {std::move(rows), std::move(offsets), 0};
  }

  if (right_by.num_columns() == 0) {
    CUDF_CUDA_TRY(cub::DeviceTransform::Fill(rows.begin(), 1, size_type{0}, env));
    CUDF_CUDA_TRY(cub::DeviceTransform::Fill(offsets.begin(), 1, size_type{0}, env));
    CUDF_CUDA_TRY(cub::DeviceTransform::Fill(offsets.begin() + 1, 1, num_rows, env));
    rows.resize(1, stream);
    offsets.resize(2, stream);
    return {std::move(rows), std::move(offsets), 1};
  }

  auto const has_nulls  = cudf::has_nested_nulls(right_by);
  auto const comparator = cudf::detail::row::equality::self_comparator{right_by, stream, mr};
  auto const row_equal =
    comparator.equal_to<false>(nullate::DYNAMIC{has_nulls}, null_equality::EQUAL);
  rmm::device_uvector<uint8_t> group_starts(num_rows, stream, mr);
  CUDF_CUDA_TRY(cub::DeviceTransform::Transform(
    cuda::counting_iterator<size_type>{0},
    group_starts.begin(),
    num_rows,
    [row_equal] __device__(size_type row) -> uint8_t {
      return row == 0 || !row_equal(row - 1, row);
    },
    env));

  cudf::detail::device_scalar<size_type> num_groups{0, stream, mr};
  auto const input  = cuda::make_zip_iterator(cuda::counting_iterator<size_type>{0},
                                             cuda::counting_iterator<size_type>{0});
  auto const output = cuda::make_zip_iterator(rows.begin(), offsets.begin());
  CUDF_CUDA_TRY(cub::DeviceSelect::Flagged(
    input, group_starts.begin(), output, num_groups.data(), num_rows, env));

  auto const host_num_groups = num_groups.value(stream);
  CUDF_CUDA_TRY(cub::DeviceTransform::Fill(offsets.begin() + host_num_groups, 1, num_rows, env));
  rows.resize(host_num_groups, stream);
  offsets.resize(static_cast<std::size_t>(host_num_groups) + 1, stream);
  return {std::move(rows), std::move(offsets), host_num_groups};
}

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

    auto const group = group_positions[left_row];
    if (group < 0 || group >= num_groups) { return JoinNoMatch; }

    auto first            = group_offsets[group];
    auto last             = group_offsets[group + 1];
    auto const left_value = left_on.element<T>(left_row);
    while (first < last) {
      auto const middle = first + (last - first) / 2;
      if (right_on.is_null(middle) || !(left_value < right_on.element<T>(middle))) {
        first = middle + 1;
      } else {
        last = middle;
      }
    }
    if (first == group_offsets[group]) { return JoinNoMatch; }
    auto const candidate = first - 1;
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

  template <typename T>
  void operator()() const
  {
    if constexpr (cudf::is_integral_not_bool<T>() || cudf::is_chrono<T>()) {
      thrust::transform(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        cuda::counting_iterator<size_type>{0},
        cuda::counting_iterator<size_type>{left_on.size()},
        output,
        backward_probe<T>{left_on, right_on, group_offsets, group_positions, num_groups});
    } else {
      CUDF_FAIL("Unsupported as-of join ordered-key type");
    }
  }
};

}  // namespace

namespace detail {

asof_join::asof_join(table_view const& right_by,
                     column_view const& right_on,
                     cuda::stream_ref stream)
  : _right_by{right_by},
    _right_on{right_on},
    _right_group_rows{0, stream},
    _right_group_offsets{0, stream}
{
  cudf::scoped_range range{"asof_join::asof_join"};
  validate_key_sizes(right_by, right_on, "right");
  CUDF_EXPECTS(is_supported_on_type(right_on.type()),
               "As-of join ordered keys must be integer, timestamp, or duration columns");
  validate_by_types(right_by);

  if (right_by.num_columns() > 0) {
    auto const mr = cudf::get_current_device_resource_ref();
    _right_eq_preprocessed =
      cudf::detail::row::equality::preprocessed_table::create(right_by, stream, mr);
    _right_lex_preprocessed =
      cudf::detail::row::lexicographic::preprocessed_table::create(right_by, {}, {}, stream);
  }

  auto group_index     = build_right_group_index(right_by, right_on.size(), stream);
  _right_group_rows    = std::move(group_index.rows);
  _right_group_offsets = std::move(group_index.offsets);
  _num_right_groups    = group_index.num_groups;
}

std::unique_ptr<rmm::device_uvector<size_type>> asof_join::join(
  table_view const& left_by,
  column_view const& left_on,
  asof_join_strategy strategy,
  bool allow_exact_matches,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"asof_join::join"};
  CUDF_EXPECTS(strategy == asof_join_strategy::BACKWARD,
               "Only backward as-of joins are currently supported");
  CUDF_EXPECTS(allow_exact_matches, "As-of joins without exact matches are currently unsupported");
  validate_key_sizes(left_by, left_on, "left");
  CUDF_EXPECTS(left_by.num_columns() == _right_by.num_columns(),
               "As-of join inputs must have the same number of grouping keys");
  CUDF_EXPECTS(cudf::have_same_types(left_by, _right_by),
               "As-of join grouping-key types must match");
  CUDF_EXPECTS(cudf::have_same_types(left_on, _right_on),
               "As-of join ordered-key types must match");
  CUDF_EXPECTS(is_supported_on_type(left_on.type()),
               "As-of join ordered keys must be integer, timestamp, or duration columns");
  validate_by_types(left_by);

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto group_positions =
    std::make_unique<rmm::device_uvector<size_type>>(left_on.size(), stream, temp_mr);

  if (_num_right_groups == 0) {
    thrust::fill(rmm::exec_policy_nosync(stream, temp_mr),
                 group_positions->begin(),
                 group_positions->end(),
                 JoinNoMatch);
  } else if (left_by.num_columns() == 0) {
    thrust::fill(rmm::exec_policy_nosync(stream, temp_mr),
                 group_positions->begin(),
                 group_positions->end(),
                 size_type{0});
  } else {
    auto const has_nulls = cudf::has_nested_nulls(_right_by) || cudf::has_nested_nulls(left_by);
    auto left_lex =
      cudf::detail::row::lexicographic::preprocessed_table::create(left_by, {}, {}, stream);
    auto const row_less = cudf::detail::row::lexicographic::two_table_comparator{
      _right_lex_preprocessed, std::move(left_lex)};
    auto const right_group_rows = cuda::transform_iterator(
      _right_group_rows.begin(),
      cuda::proclaim_return_type<detail::row::lhs_index_type>(
        [] __device__(size_type row) { return detail::row::lhs_index_type{row}; }));
    thrust::lower_bound(rmm::exec_policy_nosync(stream, temp_mr),
                        right_group_rows,
                        right_group_rows + _num_right_groups,
                        cudf::detail::row::rhs_iterator(0),
                        cudf::detail::row::rhs_iterator(0) + left_on.size(),
                        group_positions->begin(),
                        row_less.less<false>(nullate::DYNAMIC{has_nulls}));

    auto left_eq =
      cudf::detail::row::equality::preprocessed_table::create(left_by, stream, temp_mr);
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
      [group_rows = _right_group_rows.data(), num_groups = _num_right_groups, equal] __device__(
        size_type group, size_type left_row) {
        return group < num_groups && equal(detail::row::lhs_index_type{group_rows[group]},
                                           detail::row::rhs_index_type{left_row})
                 ? group
                 : JoinNoMatch;
      });
  }

  auto result = std::make_unique<rmm::device_uvector<size_type>>(left_on.size(), stream, mr);
  auto const left_on_device  = column_device_view::create(left_on, stream, temp_mr);
  auto const right_on_device = column_device_view::create(_right_on, stream, temp_mr);
  cudf::type_dispatcher(left_on.type(),
                        probe_dispatch{*left_on_device,
                                       *right_on_device,
                                       _right_group_offsets.data(),
                                       group_positions->data(),
                                       _num_right_groups,
                                       result->data(),
                                       stream});
  return result;
}

}  // namespace detail

asof_join::~asof_join() = default;

asof_join::asof_join(table_view const& right_by,
                     column_view const& right_on,
                     cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  _impl = std::make_unique<impl_type>(right_by, right_on, stream);
}

std::unique_ptr<rmm::device_uvector<size_type>> asof_join::join(
  table_view const& left_by,
  column_view const& left_on,
  asof_join_strategy strategy,
  bool allow_exact_matches,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->join(left_by, left_on, strategy, allow_exact_matches, stream, mr);
}

}  // namespace cudf
