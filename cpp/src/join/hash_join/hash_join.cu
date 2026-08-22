/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "dispatch.cuh"
#include "hash_csr_kernels.cuh"
#include "join/join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
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

#include <rmm/device_buffer.hpp>

#include <cuda/std/bit>
#include <cuda/std/cstdint>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
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
cuda::std::uint32_t hash_csr_capacity(size_type rows, double load_factor)
{
  auto const checked   = checked_load_factor(load_factor);
  auto const requested = std::max(static_cast<long double>(rows) + 1,
                                  std::ceil(static_cast<long double>(rows) / checked));
  CUDF_EXPECTS(requested <= std::numeric_limits<cuda::std::uint32_t>::max(),
               "HashCSR table capacity is not representable",
               std::overflow_error);
  auto const capacity = cuda::std::bit_ceil(static_cast<cuda::std::uint64_t>(requested));
  CUDF_EXPECTS(capacity <= std::numeric_limits<cuda::std::uint32_t>::max(),
               "HashCSR table capacity is not representable",
               std::overflow_error);
  return static_cast<cuda::std::uint32_t>(capacity);
}
}  // namespace

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& right,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             cuda::stream_ref stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : hash_join{right, has_nulls, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream, std::move(mr)}
{
}

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& right,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             cuda::stream_ref stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _has_nulls(has_nulls),
    _is_empty{right.num_rows() == 0},
    _nulls_equal{compare_nulls},
    _right{right},
    _preprocessed_right{cudf::detail::row::equality::preprocessed_table::create(
      _right, stream, cudf::get_current_device_resource_ref())},
    _impl{std::make_unique<impl>(
      hash_csr_capacity(right.num_rows(), load_factor), right.num_rows(), stream, std::move(mr))}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != right.num_columns(), "Hash join right table is empty", std::invalid_argument);
  if (_is_empty) { return; }

  CUDF_CUDA_TRY(cudaMemsetAsync(_impl->entries.data(),
                                0xff,
                                _impl->entries.size() * sizeof(hash_table_entry_type),
                                stream.get()));
  CUDF_CUDA_TRY(cudaMemsetAsync(_impl->cumulative_ends.data(),
                                0,
                                _impl->cumulative_ends.size() * sizeof(size_type),
                                stream.get()));

  auto const temp_mr     = cudf::get_current_device_resource_ref();
  auto const row_bitmask = cudf::detail::bitmask_and(right, stream, temp_mr).first;
  auto const valid_rows  = _nulls_equal == null_equality::UNEQUAL
                             ? static_cast<bitmask_type const*>(row_bitmask.data())
                             : nullptr;
  rmm::device_uvector<build_position_type> build_positions(right.num_rows(), stream, temp_mr);
  auto build = [&](auto equality, auto hasher) {
    launch_hash_csr_build_count(right.num_rows(),
                                valid_rows,
                                build_positions.data(),
                                _impl->cumulative_ends.data(),
                                _impl->hash_table(),
                                equality,
                                hasher,
                                stream);
  };
  dispatch_join_comparator(
    right, right, _preprocessed_right, _preprocessed_right, _has_nulls, _nulls_equal, build);
  std::size_t temp_storage_bytes{};
  CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(nullptr,
                                              temp_storage_bytes,
                                              _impl->cumulative_ends.data(),
                                              _impl->cumulative_ends.data(),
                                              _impl->capacity,
                                              stream.get()));
  rmm::device_buffer temp_storage(temp_storage_bytes, stream, temp_mr);
  CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(temp_storage.data(),
                                              temp_storage_bytes,
                                              _impl->cumulative_ends.data(),
                                              _impl->cumulative_ends.data(),
                                              _impl->capacity,
                                              stream.get()));
  launch_hash_csr_build_fill(right.num_rows(),
                             build_positions.data(),
                             _impl->cumulative_ends.data(),
                             _impl->values.data(),
                             stream);
}

template hash_join<hash_join_hasher>::hash_join(
  cudf::table_view const& right,
  bool has_nulls,
  cudf::null_equality compare_nulls,
  cuda::stream_ref stream,
  cuda::mr::any_resource<cuda::mr::device_accessible> mr);

template hash_join<hash_join_hasher>::hash_join(
  cudf::table_view const& right,
  bool has_nulls,
  cudf::null_equality compare_nulls,
  double load_factor,
  cuda::stream_ref stream,
  cuda::mr::any_resource<cuda::mr::device_accessible> mr);

template <typename Hasher>
hash_join<Hasher>::~hash_join() = default;

template hash_join<hash_join_hasher>::~hash_join();

}  // namespace cudf::detail

namespace cudf {

hash_join::~hash_join() = default;

hash_join::hash_join(cudf::table_view const& right,
                     null_equality compare_nulls,
                     cuda::stream_ref stream,
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
                     cuda::stream_ref stream,
                     cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<impl_type const>(
      right, has_nulls == nullable_join::YES, compare_nulls, load_factor, stream, std::move(mr))}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::inner_join(cudf::table_view const& left,
                      std::optional<std::size_t> output_size,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(left, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::left_join(cudf::table_view const& left,
                     std::optional<std::size_t> output_size,
                     cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(left, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join(cudf::table_view const& left,
                     std::optional<std::size_t> output_size,
                     cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->full_join(left, output_size, stream, mr);
}

std::size_t hash_join::inner_join_size(cudf::table_view const& left, cuda::stream_ref stream) const
{
  return _impl->inner_join_size(left, stream);
}

std::size_t hash_join::left_join_size(cudf::table_view const& left, cuda::stream_ref stream) const
{
  return _impl->left_join_size(left, stream);
}

std::size_t hash_join::full_join_size(cudf::table_view const& left,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_size(left, stream, mr);
}

cudf::join_match_context hash_join::inner_join_match_context(
  cudf::table_view const& left, cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join_match_context(left, stream, mr);
}

cudf::join_match_context hash_join::left_join_match_context(cudf::table_view const& left,
                                                            cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->left_join_match_context(left, stream, mr);
}

cudf::join_match_context hash_join::full_join_match_context(cudf::table_view const& left,
                                                            cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_match_context(left, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::partitioned_inner_join(cudf::join_partition_context const& context,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->partitioned_inner_join(context, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::partitioned_left_join(cudf::join_partition_context const& context,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->partitioned_left_join(context, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::partitioned_full_join(cudf::join_partition_context const& context,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->partitioned_full_join(context, stream, mr);
}

}  // namespace cudf
