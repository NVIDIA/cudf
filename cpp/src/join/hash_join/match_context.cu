/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "dispatch.cuh"
#include "join/join_common_utils.cuh"
#include "partitioned_count_kernels.hpp"

#include <cudf/detail/iterator.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/fill.h>
#include <thrust/transform.h>

namespace cudf::detail {

std::unique_ptr<rmm::device_uvector<size_type>> make_join_match_counts(
  table_view const& right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_right,
  cudf::detail::hash_table_t const& hash_table,
  bool is_empty,
  bool has_nulls,
  null_equality compare_nulls,
  join_kind join,
  table_view const& left,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto match_counts = std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);

  if (is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 match_counts->begin(),
                 match_counts->end(),
                 join == join_kind::INNER_JOIN ? 0 : 1);
    return match_counts;
  }

  CUDF_EXPECTS(has_nulls || !cudf::has_nested_nulls(left),
               "Left table has nulls while right table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_left = cudf::detail::row::equality::preprocessed_table::create(
    left, stream, cudf::get_current_device_resource_ref());
  auto const left_table_num_rows = left.num_rows();

  auto count_matches = [&](auto equality, auto d_hasher) {
    // Precompute left keys: {hash(row_idx), row_idx} for each left row.
    auto const n = static_cast<thread_index_type>(left_table_num_rows);
    rmm::device_uvector<probe_key_type> left_keys(n, stream);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(left_table_num_rows),
                      left_keys.begin(),
                      pair_fn{d_hasher});

    auto const ref = hash_table.ref(cuco::op::count)
                       .rebind_key_eq(equality)
                       .rebind_hash_function(hash_table.hash_function());
    if (join == join_kind::INNER_JOIN) {
      launch_partitioned_count<false>(left_keys.data(), n, match_counts->begin(), ref, stream);
    } else {
      // IsOuter=true handles the clamp (zero → 1) for LEFT/FULL joins internally.
      launch_partitioned_count<true>(left_keys.data(), n, match_counts->begin(), ref, stream);
    }
  };

  dispatch_join_comparator(
    right, left, preprocessed_right, preprocessed_left, has_nulls, compare_nulls, count_matches);

  return match_counts;
}

template <typename Hasher>
std::unique_ptr<rmm::device_uvector<size_type>> hash_join<Hasher>::make_match_counts(
  join_kind join,
  cudf::table_view const& left,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  return make_join_match_counts(_right,
                                _preprocessed_right,
                                _impl->_hash_table,
                                _is_empty,
                                _has_nulls,
                                _nulls_equal,
                                join,
                                left,
                                stream,
                                mr);
}

template std::unique_ptr<rmm::device_uvector<size_type>>
hash_join<hash_join_hasher>::make_match_counts(join_kind,
                                               cudf::table_view const&,
                                               cuda::stream_ref,
                                               rmm::device_async_resource_ref) const;

}  // namespace cudf::detail
