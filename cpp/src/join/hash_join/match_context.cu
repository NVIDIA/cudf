/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "dispatch.cuh"
#include "hash_csr_kernels.cuh"

#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

namespace cudf::detail {

template <typename Hasher>
std::unique_ptr<rmm::device_uvector<size_type>> hash_join<Hasher>::make_match_counts(
  join_kind join,
  table_view const& left,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  auto match_counts = std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);

  if (_is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 match_counts->begin(),
                 match_counts->end(),
                 join == join_kind::INNER_JOIN ? 0 : 1);
    return match_counts;
  }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(left),
               "Left table has nulls while right table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream);
  auto const temp_mr     = cudf::get_current_device_resource_ref();
  auto const row_bitmask = cudf::detail::bitmask_and(left, stream, temp_mr).first;
  auto const valid_rows  = _nulls_equal == null_equality::UNEQUAL
                             ? static_cast<bitmask_type const*>(row_bitmask.data())
                             : nullptr;

  auto count_matches = [&](auto equality, auto hasher) {
    if (join == join_kind::INNER_JOIN) {
      launch_hash_csr_probe_count<false>(left.num_rows(),
                                         valid_rows,
                                         nullptr,
                                         match_counts->data(),
                                         nullptr,
                                         nullptr,
                                         _impl->map_view(),
                                         _impl->csr_view(),
                                         equality,
                                         hasher,
                                         stream);
    } else {
      launch_hash_csr_probe_count<true>(left.num_rows(),
                                        valid_rows,
                                        nullptr,
                                        match_counts->data(),
                                        nullptr,
                                        nullptr,
                                        _impl->map_view(),
                                        _impl->csr_view(),
                                        equality,
                                        hasher,
                                        stream);
    }
  };

  dispatch_join_comparator(
    _right, left, _preprocessed_right, preprocessed_left, _has_nulls, _nulls_equal, count_matches);

  return match_counts;
}

template std::unique_ptr<rmm::device_uvector<size_type>>
hash_join<hash_join_hasher>::make_match_counts(join_kind,
                                               cudf::table_view const&,
                                               cuda::stream_ref,
                                               rmm::device_async_resource_ref) const;

}  // namespace cudf::detail
