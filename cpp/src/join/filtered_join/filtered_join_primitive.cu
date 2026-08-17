/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "filtered_join_common.cuh"

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuco/operator.hpp>
#include <cuco/static_set_ref.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuda/stream>

#include <memory>

namespace cudf::detail {

namespace {

using primitive_row_comparator = cudf::detail::row::primitive::row_equality_comparator;
using primitive_row_hasher =
  cudf::detail::row::primitive::row_hasher<cudf::hashing::detail::default_hash>;

}  // namespace

void filtered_join::insert_right_table_primitive(cuda::stream_ref stream)
{
  auto const comparator = primitive_row_comparator{
    nullate::DYNAMIC{true}, _preprocessed_right, _preprocessed_right, _nulls_equal};
  cuco::static_set_ref set_ref{empty_sentinel_key,
                               insertion_adapter{comparator},
                               single_probing_scheme{},
                               cuco::thread_scope_device,
                               _bucket_storage.ref()};
  auto const hasher = primitive_row_hasher{nullate::DYNAMIC{true}, _preprocessed_right};
  auto const iter   = cudf::detail::make_counting_transform_iterator(
    size_type{0}, key_pair_fn<lhs_index_type, primitive_row_hasher>{hasher});
  insert_right_table<single_probing_scheme::cg_size>(
    iter, set_ref.rebind_operators(cuco::insert), stream);
}

void filtered_join::query_right_table_primitive(
  cudf::table_view const& left,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  cudf::device_span<bool> contains_map,
  cuda::stream_ref stream)
{
  auto const comparator = primitive_row_comparator{
    nullate::DYNAMIC{true}, _preprocessed_right, preprocessed_left, _nulls_equal};
  cuco::static_set_ref set_ref{empty_sentinel_key,
                               comparator_adapter{comparator},
                               single_probing_scheme{},
                               cuco::thread_scope_device,
                               _bucket_storage.ref()};
  auto const hasher = primitive_row_hasher{nullate::DYNAMIC{true}, preprocessed_left};
  auto const iter   = cudf::detail::make_counting_transform_iterator(
    size_type{0}, key_pair_fn<rhs_index_type, primitive_row_hasher>{hasher});
  query_right_table<single_probing_scheme::cg_size>(
    left, iter, set_ref.rebind_operators(cuco::op::contains), contains_map, stream);
}

}  // namespace cudf::detail
