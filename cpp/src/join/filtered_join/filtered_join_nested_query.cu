/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "filtered_join_common.cuh"
#include "hash/murmurhash3_x86_32.cuh"

#include <cudf/column/column.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuco/operator.hpp>
#include <cuco/static_set_ref.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <memory>

namespace cudf::detail {

void distinct_filtered_join::query_right_table_nested(
  cudf::table_view const& left,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  cudf::device_span<bool> contains_map,
  rmm::cuda_stream_view stream)
{
  auto const comparator =
    cudf::detail::row::equality::two_table_comparator{_preprocessed_right, preprocessed_left}
      .equal_to<true>(nullate::YES{},
                      _nulls_equal,
                      cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
  cuco::static_set_ref set_ref{empty_sentinel_key,
                               comparator_adapter{comparator},
                               nested_probing_scheme{},
                               cuco::thread_scope_device,
                               _bucket_storage.ref()};
  auto const hashes =
    cudf::hashing::detail::murmurhash3_x86_32(preprocessed_left,
                                              left.num_rows(),
                                              cudf::DEFAULT_HASH_SEED,
                                              stream,
                                              cudf::get_current_device_resource_ref());
  auto const hasher = precomputed_hash{hashes->view().data<hash_value_type>()};
  auto const iter   = cudf::detail::make_counting_transform_iterator(
    size_type{0}, key_pair_fn<join_rhs_index_type, decltype(hasher)>{hasher});
  query_right_table<nested_probing_scheme::cg_size>(
    left, iter, set_ref.rebind_operators(cuco::op::contains), contains_map, stream);
}

}  // namespace cudf::detail
