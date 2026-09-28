/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "dispatch.cuh"
#include "hash_csr_kernels.cuh"

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
#include <rmm/device_uvector.hpp>

#include <cuda/std/bit>
#include <cuda/std/cstdint>

#include <algorithm>
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

namespace {
bool has_list_or_string(column_view const& column)
{
  return column.type().id() == type_id::LIST || column.type().id() == type_id::STRING ||
         std::any_of(column.child_begin(), column.child_end(), has_list_or_string);
}

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
  // Avoid power-of-two rounding at the default and lower load factors. Retain the extra
  // headroom of rounded capacities at higher load factors, where linear probing is sensitive
  // to occupancy (in particular, load_factor == 1 must not produce an almost-full table).
  return static_cast<cuda::std::uint32_t>(checked <= CUCO_DESIRED_LOAD_FACTOR ? requested
                                                                              : capacity);
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

  CUDF_CUDA_TRY(cudaMemsetAsync(
    _impl->_slots.data(), 0xff, _impl->_slots.size() * sizeof(hash_table_slot_type), stream.get()));
  CUDF_CUDA_TRY(cudaMemsetAsync(
    _impl->_offsets.data(), 0, _impl->_offsets.size() * sizeof(size_type), stream.get()));

  auto const temp_mr     = cudf::get_current_device_resource_ref();
  auto const row_bitmask = _nulls_equal == null_equality::UNEQUAL
                             ? cudf::detail::bitmask_and(right, stream, temp_mr).first
                             : rmm::device_buffer{0, stream, temp_mr};
  auto const valid_rows  = _nulls_equal == null_equality::UNEQUAL
                             ? static_cast<bitmask_type const*>(row_bitmask.data())
                             : nullptr;
  // Hashing and comparing variable-width rows again can dominate construction. Cache one
  // representative index per row for keys containing lists or strings, including within structs.
  auto const cache_representatives = std::any_of(right.begin(), right.end(), has_list_or_string);
  auto representatives             = rmm::device_uvector<size_type>{
    cache_representatives ? static_cast<std::size_t>(right.num_rows()) : 0, stream, temp_mr};
  auto build = [&](auto equality, auto hasher) {
    launch_hash_csr_build_count_kernel(right.num_rows(),
                                       valid_rows,
                                       _impl->_offsets.data(),
                                       representatives.data(),
                                       _impl->hash_table(),
                                       equality,
                                       hasher,
                                       stream);
  };
  dispatch_join_comparator(
    right, right, _preprocessed_right, _preprocessed_right, _has_nulls, _nulls_equal, build);
  {
    std::size_t temp_storage_bytes{};
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(nullptr,
                                                temp_storage_bytes,
                                                _impl->_offsets.data(),
                                                _impl->_offsets.data(),
                                                _impl->_offsets.size(),
                                                stream.get()));
    rmm::device_buffer temp_storage(temp_storage_bytes, stream, temp_mr);
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(temp_storage.data(),
                                                temp_storage_bytes,
                                                _impl->_offsets.data(),
                                                _impl->_offsets.data(),
                                                _impl->_offsets.size(),
                                                stream.get()));
  }
  // The output array is not needed until the scan workspace has been released.
  _impl->_values.resize(right.num_rows(), stream);
  // Reuse each cumulative end as a scatter cursor. Once all rows in a group have been
  // scattered, its cursor is the group's exclusive begin. No per-row positions are retained.
  if (cache_representatives) {
    launch_hash_csr_build_fill_cached_kernel(right.num_rows(),
                                             representatives.data(),
                                             _impl->_offsets.data(),
                                             _impl->_values.data(),
                                             stream);
    return;
  }
  auto fill = [&](auto equality, auto hasher) {
    launch_hash_csr_build_fill_kernel(right.num_rows(),
                                      valid_rows,
                                      _impl->_offsets.data(),
                                      _impl->_values.data(),
                                      _impl->hash_table(),
                                      equality,
                                      hasher,
                                      stream);
  };
  dispatch_join_comparator(
    right, right, _preprocessed_right, _preprocessed_right, _has_nulls, _nulls_equal, fill);
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
