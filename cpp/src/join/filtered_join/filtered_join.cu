/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/join_common_utils.hpp"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/extent.cuh>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/sequence.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace cudf {
namespace detail {

/**
 * @brief Returns a validity mask for rows without nulls at any nested level, or null when unused.
 */
std::pair<rmm::device_buffer, bitmask_type const*> make_filtered_join_row_bitmask(
  table_view const& input, null_equality nulls_equal, cuda::stream_ref stream)
{
  if (nulls_equal == null_equality::EQUAL || !has_nested_nulls(input)) {
    return std::pair(rmm::device_buffer{0, stream}, nullptr);
  }

  auto const nullable_columns = get_nullable_columns(input);
  // Combine multiple masks so a row is valid only when every nested level is valid. Reuse a single
  // mask directly when possible.
  if (nullable_columns.size() > 1) {
    auto row_bitmask =
      cudf::detail::bitmask_and(
        table_view{nullable_columns}, stream, cudf::get_current_device_resource_ref())
        .first;
    auto const row_bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
    return std::pair(std::move(row_bitmask), row_bitmask_ptr);
  }

  return std::pair(rmm::device_buffer{0, stream}, nullable_columns.front().null_mask());
}

namespace {

struct gather_mask {
  join_kind kind;
  device_span<bool const> flagged;
  __device__ bool operator()(size_type idx) const noexcept
  {
    return flagged[idx] == (kind == join_kind::LEFT_SEMI_JOIN);
  }
};

}  // namespace

filtered_join::row_operator_mode filtered_join::select_row_operator_mode(
  cudf::table_view const& table)
{
  if (is_primitive_row_op_compatible(table)) { return row_operator_mode::PRIMITIVE; }
  return cudf::has_nested_columns(table) ? row_operator_mode::NESTED : row_operator_mode::FLAT;
}

std::size_t filtered_join::compute_bucket_storage_size(cudf::size_type num_rows,
                                                       double load_factor,
                                                       row_operator_mode mode)
{
  if (mode == row_operator_mode::NESTED) {
    return static_cast<std::size_t>(
      cuco::make_valid_extent<nested_probing_scheme, storage_type, std::size_t>(num_rows,
                                                                                load_factor));
  }
  return static_cast<std::size_t>(
    cuco::make_valid_extent<single_probing_scheme, storage_type, std::size_t>(num_rows,
                                                                              load_factor));
}

filtered_join::filtered_join(cudf::table_view const& right,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             cuda::stream_ref stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _right_mode{select_row_operator_mode(right)},
    _bucket_storage{cuco::extent<std::size_t>{compute_bucket_storage_size(
                      right.num_rows(), checked_load_factor(load_factor), _right_mode)},
                    rmm::mr::polymorphic_allocator<char>{std::move(mr)},
                    stream.get()},
    _right{right},
    _nulls_equal{compare_nulls},
    _preprocessed_right{cudf::detail::row::equality::preprocessed_table::create(
      _right, stream, cudf::get_current_device_resource_ref())}
{
  cudf::scoped_range range{"filtered_join::filtered_join"};
  if (_right.num_rows() == 0) return;
  _bucket_storage.initialize(empty_sentinel_key, stream);
  if (_right_mode == row_operator_mode::PRIMITIVE) {
    insert_right_table_primitive(stream);
  } else if (_right_mode == row_operator_mode::NESTED) {
    insert_right_table_nested(stream);
  } else {
    insert_right_table_flat(stream);
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join::semi_anti_join(
  cudf::table_view const& left,
  join_kind kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"filtered_join::semi_anti_join"};

  auto const temp_mr           = cudf::get_current_device_resource_ref();
  auto const preprocessed_left = [&left, stream, temp_mr] {
    cudf::scoped_range range{"filtered_join::semi_anti_join::preprocessed_left"};
    return cudf::detail::row::equality::preprocessed_table::create(left, stream, temp_mr);
  }();

  auto contains_map            = rmm::device_uvector<bool>(left.num_rows(), stream, temp_mr);
  auto const contains_map_span = cudf::device_span<bool>{contains_map.data(), contains_map.size()};
  if (_right_mode == row_operator_mode::PRIMITIVE) {
    query_right_table_primitive(left, preprocessed_left, contains_map_span, stream);
  } else if (_right_mode == row_operator_mode::NESTED) {
    query_right_table_nested(left, preprocessed_left, contains_map_span, stream);
  } else {
    query_right_table_flat(left, preprocessed_left, contains_map_span, stream);
  }

  rmm::device_uvector<size_type> gather_map(left.num_rows(), stream, mr);
  auto gather_map_end = thrust::copy_if(rmm::exec_policy_nosync(stream, temp_mr),
                                        cuda::counting_iterator<size_type>{0},
                                        cuda::counting_iterator<size_type>{left.num_rows()},
                                        gather_map.begin(),
                                        gather_mask{kind, contains_map_span});
  gather_map.resize(cuda::std::distance(gather_map.begin(), gather_map_end), stream);
  return std::make_unique<rmm::device_uvector<size_type>>(std::move(gather_map));
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join::semi_join(
  cudf::table_view const& left, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  // Early return for empty right or left table
  if (_right.num_rows() == 0 || left.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }

  return semi_anti_join(left, join_kind::LEFT_SEMI_JOIN, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join::anti_join(
  cudf::table_view const& left, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  // Early return for empty left table
  if (left.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  if (_right.num_rows() == 0) {
    auto result =
      std::make_unique<rmm::device_uvector<cudf::size_type>>(left.num_rows(), stream, mr);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     result->begin(),
                     result->end());
    return result;
  }

  return semi_anti_join(left, join_kind::LEFT_ANTI_JOIN, stream, mr);
}

}  // namespace detail

filtered_join::~filtered_join() = default;

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             double load_factor,
                             cuda::stream_ref stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<cudf::detail::filtered_join>(
      build, compare_nulls, load_factor, stream, std::move(mr))}
{
}

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             cuda::stream_ref stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : filtered_join(
      build, compare_nulls, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream, std::move(mr))
{
}

std::unique_ptr<rmm::device_uvector<size_type>> filtered_join::semi_join(
  cudf::table_view const& probe, cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  return _impl->semi_join(probe, stream, mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> filtered_join::anti_join(
  cudf::table_view const& probe, cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  return _impl->anti_join(probe, stream, mr);
}

}  // namespace cudf
