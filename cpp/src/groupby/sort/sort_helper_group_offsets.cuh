/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common_utils.cuh"
#include "stream_compaction/stream_compaction_common.cuh"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/transform.h>

namespace cudf::groupby::detail::sort {

size_type compute_nested_group_offsets(table_view const& keys,
                                       size_type const* sorted_order,
                                       size_type size,
                                       rmm::device_uvector<size_type>& group_offsets,
                                       rmm::cuda_stream_view stream);

template <bool HasNested>
size_type compute_group_offsets(table_view const& keys,
                                size_type const* sorted_order,
                                size_type size,
                                rmm::device_uvector<size_type>& group_offsets,
                                rmm::cuda_stream_view stream)
{
  auto const comparator  = cudf::detail::row::equality::self_comparator{keys, stream};
  auto const d_key_equal = comparator.equal_to<HasNested>(
    cudf::nullate::DYNAMIC{cudf::has_nested_nulls(keys)}, null_equality::EQUAL);
  // Using a temporary buffer for intermediate transform results from the iterator containing
  // the comparator speeds up compile-time significantly without much degradation in
  // runtime performance over using the comparator directly in thrust::unique_copy.
  auto result       = rmm::device_uvector<bool>(size, stream);
  auto const itr    = cuda::counting_iterator<size_type>{0};
  auto const row_eq = permuted_row_equality_comparator(d_key_equal, sorted_order);
  auto const ufn    = cudf::detail::unique_copy_fn<decltype(itr), decltype(row_eq)>{
    itr, duplicate_keep_option::KEEP_FIRST, row_eq, size - 1};
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    itr,
                    itr + size,
                    result.begin(),
                    ufn);
  auto const result_end = cudf::detail::copy_if(
    itr, itr + size, result.begin(), group_offsets.begin(), cuda::std::identity{}, stream);
  return cuda::std::distance(group_offsets.begin(), result_end);
}

}  // namespace cudf::groupby::detail::sort
