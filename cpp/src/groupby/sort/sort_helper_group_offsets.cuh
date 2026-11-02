/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf::groupby::detail::sort {

/// Compacts materialized group-boundary flags through one shared device-algorithm launcher.
size_type compact_group_offsets(bool* flags,
                                size_type size,
                                size_type* offsets,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr);

size_type compute_nested_group_offsets(table_view const& keys,
                                       size_type const* sorted_order,
                                       size_type size,
                                       rmm::device_uvector<size_type>& group_offsets,
                                       cuda::stream_ref stream);

template <bool HasNested>
size_type compute_group_offsets(table_view const& keys,
                                size_type const* sorted_order,
                                size_type size,
                                rmm::device_uvector<size_type>& group_offsets,
                                cuda::stream_ref stream)
{
  auto const temp_mr     = cudf::get_current_device_resource_ref();
  auto const comparator  = cudf::detail::row::equality::self_comparator{keys, stream, temp_mr};
  auto const d_key_equal = comparator.equal_to<HasNested>(
    cudf::nullate::DYNAMIC{cudf::has_nested_nulls(keys)}, null_equality::EQUAL);
  // Using a temporary buffer for intermediate transform results from the iterator containing
  // the comparator speeds up compile-time significantly without much degradation in
  // runtime performance over using the comparator directly in thrust::unique_copy.
  auto result    = cuda::device_buffer<bool>(stream, temp_mr, size, cuda::no_init);
  auto const itr = cuda::counting_iterator<size_type>{0};
  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    itr,
                    itr + size,
                    result.begin(),
                    [d_key_equal, sorted_order] __device__(size_type row) {
                      return row == 0 || !d_key_equal(sorted_order[row], sorted_order[row - 1]);
                    });
  return compact_group_offsets(result.data(), size, group_offsets.data(), stream, temp_mr);
}

}  // namespace cudf::groupby::detail::sort
