/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "join/join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuda/iterator>

#include <limits>
#include <memory>
#include <utility>

namespace cudf::detail {

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type)
{
  if (left.is_empty() || right.is_empty()) { return true; }
  if ((join_kind::LEFT_JOIN == join_type) && (0 == left.num_rows())) { return true; }
  if ((join_kind::INNER_JOIN == join_type) && ((0 == left.num_rows()) || (0 == right.num_rows()))) {
    return true;
  }
  if ((join_kind::LEFT_SEMI_JOIN == join_type) && (0 == right.num_rows())) { return true; }
  if ((join_kind::LEFT_SEMI_JOIN == join_type || join_kind::LEFT_ANTI_JOIN == join_type) &&
      (0 == left.num_rows())) {
    return true;
  }
  return false;
}

void validate_hash_join_probe(table_view const& right, table_view const& left, bool has_nulls)
{
  CUDF_EXPECTS(0 != left.num_columns(), "Hash join left table is empty", std::invalid_argument);
  CUDF_EXPECTS(right.num_columns() == left.num_columns(),
               "Mismatch in number of columns to be joined on",
               std::invalid_argument);
  CUDF_EXPECTS(has_nulls || !cudf::has_nested_nulls(left),
               "Left table has nulls while right table was not hashed with null check.",
               std::invalid_argument);
  CUDF_EXPECTS(cudf::have_same_types(right, left),
               "Mismatch in joining column data types",
               cudf::data_type_error);
}

namespace {
void build_hash_join(
  cudf::table_view const& right,
  std::shared_ptr<detail::row::equality::preprocessed_table> const& preprocessed_right,
  cudf::detail::hash_table_t& hash_table,
  bool has_nested_nulls,
  null_equality nulls_equal,
  [[maybe_unused]] bitmask_type const* bitmask,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(0 != right.num_columns(), "Selected right dataset is empty", std::invalid_argument);
  CUDF_EXPECTS(0 != right.num_rows(), "Right side table has no rows", std::invalid_argument);

  auto insert_rows = [&](auto const& right, auto const& d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});

    if (nulls_equal == cudf::null_equality::EQUAL or not nullable(right)) {
      hash_table.insert(iter, iter + right.num_rows(), stream.value());
    } else {
      auto const stencil = cuda::counting_iterator<size_type>{0};
      auto const pred    = row_is_valid{bitmask};

      hash_table.insert_if(iter, iter + right.num_rows(), stencil, pred, stream.value());
    }
  };

  auto const nulls = nullate::DYNAMIC{has_nested_nulls};

  if (cudf::detail::is_primitive_row_op_compatible(right)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{nulls, preprocessed_right};

    insert_rows(right, d_hasher);
  } else {
    auto const row_hash = detail::row::hash::row_hasher{preprocessed_right};
    auto const d_hasher = row_hash.device_hasher(nulls);

    insert_rows(right, d_hasher);
  }
}
}  // namespace

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& right,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             rmm::cuda_stream_view stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : hash_join{right, has_nulls, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream, std::move(mr)}
{
}

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& right,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _has_nulls(has_nulls),
    _is_empty{right.num_rows() == 0},
    _nulls_equal{compare_nulls},
    _impl{std::make_unique<impl>(impl{typename impl::hash_table_t{
      cuco::extent{static_cast<size_t>(right.num_rows())},
      checked_load_factor(load_factor),
      cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(), cudf::JoinNoMatch}},
      {},
      {},
      {},
      {},
      rmm::mr::polymorphic_allocator<char>{std::move(mr)},
      stream.value()}})},
    _right{right},
    _preprocessed_right{cudf::detail::row::equality::preprocessed_table::create(_right, stream)}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != right.num_columns(), "Hash join right table is empty", std::invalid_argument);
  if (_is_empty) { return; }

  auto const row_bitmask =
    cudf::detail::bitmask_and(right, stream, cudf::get_current_device_resource_ref()).first;
  cudf::detail::build_hash_join(_right,
                                _preprocessed_right,
                                _impl->_hash_table,
                                _has_nulls,
                                _nulls_equal,
                                reinterpret_cast<bitmask_type const*>(row_bitmask.data()),
                                stream);
}

template hash_join<hash_join_hasher>::hash_join(
  cudf::table_view const& right,
  bool has_nulls,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  cuda::mr::any_resource<cuda::mr::device_accessible> mr);

template hash_join<hash_join_hasher>::hash_join(
  cudf::table_view const& right,
  bool has_nulls,
  cudf::null_equality compare_nulls,
  double load_factor,
  rmm::cuda_stream_view stream,
  cuda::mr::any_resource<cuda::mr::device_accessible> mr);

template <typename Hasher>
hash_join<Hasher>::~hash_join() = default;

template hash_join<hash_join_hasher>::~hash_join();

}  // namespace cudf::detail

namespace cudf {

hash_join::~hash_join() = default;

hash_join::hash_join(cudf::table_view const& right,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream,
                     cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : hash_join(right,
              nullable_join::YES,
              compare_nulls,
              cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
              stream,
              std::move(mr))
{
}

hash_join::hash_join(cudf::table_view const& right,
                     nullable_join has_nulls,
                     null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream,
                     cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<impl_type const>(
      right, has_nulls == nullable_join::YES, compare_nulls, load_factor, stream, std::move(mr))}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::inner_join(cudf::table_view const& left,
                      std::optional<std::size_t> output_size,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(left, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::left_join(cudf::table_view const& left,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(left, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join(cudf::table_view const& left,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->full_join(left, output_size, stream, mr);
}

std::size_t hash_join::inner_join_size(cudf::table_view const& left,
                                       rmm::cuda_stream_view stream) const
{
  return _impl->inner_join_size(left, stream);
}

std::size_t hash_join::left_join_size(cudf::table_view const& left,
                                      rmm::cuda_stream_view stream) const
{
  return _impl->left_join_size(left, stream);
}

std::size_t hash_join::full_join_size(cudf::table_view const& left,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_size(left, stream, mr);
}

cudf::join_match_context hash_join::inner_join_match_context(
  cudf::table_view const& left,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join_match_context(left, stream, mr);
}

cudf::join_match_context hash_join::left_join_match_context(cudf::table_view const& left,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->left_join_match_context(left, stream, mr);
}

cudf::join_match_context hash_join::full_join_match_context(cudf::table_view const& left,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_match_context(left, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::partitioned_inner_join(cudf::join_partition_context const& context,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->partitioned_inner_join(context, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::partitioned_left_join(cudf::join_partition_context const& context,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->partitioned_left_join(context, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::partitioned_full_join(cudf::join_partition_context const& context,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->partitioned_full_join(context, stream, mr);
}

}  // namespace cudf
