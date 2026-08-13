/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "retrieve_impl.cuh"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

#include <memory>

namespace cudf::detail {

namespace {
std::size_t compute_left_join_complement_size(cudf::device_span<size_type const> right_indices,
                                              size_type left_table_row_count,
                                              size_type right_table_row_count,
                                              rmm::cuda_stream_view stream)
{
  if (left_table_row_count == 0) { return right_table_row_count; }

  auto invalid_index_map =
    std::make_unique<rmm::device_uvector<size_type>>(right_table_row_count, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    invalid_index_map->begin(),
    invalid_index_map->end(),
    int32_t{1});

  valid_range<size_type> valid(0, right_table_row_count);

  thrust::scatter_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::make_constant_iterator(0),
                     cuda::make_constant_iterator(0) + right_indices.size(),
                     right_indices.begin(),
                     right_indices.begin(),
                     invalid_index_map->begin(),
                     valid);

  return cudf::detail::count_if(
    invalid_index_map->begin(), invalid_index_map->end(), cuda::std::identity{}, stream);
}
}  // namespace

std::size_t get_full_join_size(
  cudf::table_view const& right_table,
  cudf::table_view const& left_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::size_t join_size = compute_join_output_size<join_kind::LEFT_JOIN>(right_table,
                                                                         left_table,
                                                                         preprocessed_right,
                                                                         preprocessed_left,
                                                                         hash_table,
                                                                         has_nulls,
                                                                         compare_nulls,
                                                                         stream);

  if (join_size == 0) { return join_size; }

  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const out_build_begin =
    cuda::make_transform_output_iterator(right_indices->begin(), output_fn{});

  retrieve_left_join_build_indices(right_table,
                                   left_table,
                                   preprocessed_right,
                                   preprocessed_left,
                                   hash_table,
                                   has_nulls,
                                   compare_nulls,
                                   out_build_begin,
                                   stream);

  auto const left_table_row_count  = left_table.num_rows();
  auto const right_table_row_count = right_table.num_rows();

  return join_size + compute_left_join_complement_size(
                       *right_indices, left_table_row_count, right_table_row_count, stream);
}

}  // namespace cudf::detail
