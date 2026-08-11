/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "filtered_join_common.cuh"

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuco/operator.hpp>
#include <cuco/static_set_ref.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <memory>

namespace cudf::detail {

void filtered_join::insert_right_table_flat(rmm::cuda_stream_view stream)
{
  auto const comparator =
    cudf::detail::row::equality::self_comparator{_preprocessed_right}.equal_to<false>(
      nullate::YES{},
      _nulls_equal,
      cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
  cuco::static_set_ref set_ref{empty_sentinel_key,
                               insertion_adapter{comparator},
                               single_probing_scheme{},
                               cuco::thread_scope_device,
                               _bucket_storage.ref()};
  auto hasher =
    cudf::detail::row::hash::row_hasher{_preprocessed_right}.device_hasher(nullate::YES{});
  auto const iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, key_pair_fn<lhs_index_type, decltype(hasher)>{hasher});
  insert_right_table<single_probing_scheme::cg_size>(
    iter, set_ref.rebind_operators(cuco::insert), stream);
}

void filtered_join::query_right_table_flat(
  cudf::table_view const& left,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  cudf::device_span<bool> contains_map,
  rmm::cuda_stream_view stream)
{
  auto const comparator =
    cudf::detail::row::equality::two_table_comparator{_preprocessed_right, preprocessed_left}
      .equal_to<false>(nullate::YES{},
                       _nulls_equal,
                       cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
  cuco::static_set_ref set_ref{empty_sentinel_key,
                               comparator_adapter{comparator},
                               single_probing_scheme{},
                               cuco::thread_scope_device,
                               _bucket_storage.ref()};
  auto hasher =
    cudf::detail::row::hash::row_hasher{preprocessed_left}.device_hasher(nullate::YES{});
  auto const iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, key_pair_fn<rhs_index_type, decltype(hasher)>{hasher});
  query_right_table<single_probing_scheme::cg_size>(
    left, iter, set_ref.rebind_operators(cuco::op::contains), contains_map, stream);
}

}  // namespace cudf::detail
