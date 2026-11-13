/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/primitive_common.cuh>
#include <cudf/detail/row_operator/primitive_equality.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

#include <vector>

namespace cudf::groupby::detail {

/// Compacts materialized group-boundary flags through one shared device-algorithm launcher.
size_type compact_group_offsets(bool* flags,
                                size_type size,
                                size_type* offsets,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr);

size_type compute_nested_group_offsets(table_view const& keys,
                                       device_span<size_type> group_offsets,
                                       cuda::stream_ref stream);

/**
 * @brief Views the timestamp, duration, and 32- and 64-bit fixed-point columns of `keys` as
 * integers of the same width.
 *
 * Two values of these types are equal exactly when their integer representations are, so the
 * returned columns can be compared with the primitive row comparator.
 */
std::vector<column_view> as_integer_keys(table_view const& keys);

/// Writes the rows where `d_key_equal(row, row - 1)` is false to `group_offsets`.
template <typename KeyEqual>
size_type mark_group_offsets(size_type size,
                             KeyEqual d_key_equal,
                             device_span<size_type> group_offsets,
                             cuda::stream_ref stream,
                             cudf::memory_resources mr)
{
  // Using a temporary buffer for intermediate transform results from the iterator containing
  // the comparator speeds up compile-time significantly without much degradation in
  // runtime performance over using the comparator directly in thrust::unique_copy.
  auto result    = cuda::device_buffer<bool>(stream, mr.get_temporary_mr(), size, cuda::no_init);
  auto const itr = cuda::counting_iterator<size_type>{0};
  thrust::transform(
    rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
    itr,
    itr + size,
    result.begin(),
    [d_key_equal] __device__(size_type row) { return row == 0 || !d_key_equal(row, row - 1); });
  return compact_group_offsets(
    result.data(), size, group_offsets.data(), stream, mr.as_temporary());
}

/// Writes the first row of each run of equal adjacent `keys` rows to `group_offsets`.
template <bool HasNested>
size_type compute_group_offsets(table_view const& keys,
                                device_span<size_type> group_offsets,
                                cuda::stream_ref stream)
{
  auto const size           = keys.num_rows();
  cudf::memory_resources mr = cudf::get_current_device_resource_ref();
  auto const temp_mr        = mr.get_temporary_mr();
  if constexpr (!HasNested) {
    auto const integer_keys = table_view{as_integer_keys(keys)};
    if (cudf::detail::is_primitive_row_op_compatible(integer_keys)) {
      auto const preprocessed =
        cudf::detail::row::equality::preprocessed_table::create(integer_keys, stream, temp_mr);
      auto const d_key_equal = cudf::detail::row::primitive::row_equality_comparator{
        cudf::nullate::DYNAMIC{cudf::has_nulls(integer_keys)},
        preprocessed,
        preprocessed,
        null_equality::EQUAL};
      return mark_group_offsets(size, d_key_equal, group_offsets, stream, mr);
    }
  }
  auto const comparator  = cudf::detail::row::equality::self_comparator{keys, stream, temp_mr};
  auto const d_key_equal = comparator.equal_to<HasNested>(
    cudf::nullate::DYNAMIC{cudf::has_nested_nulls(keys)}, null_equality::EQUAL);
  return mark_group_offsets(size, d_key_equal, group_offsets, stream, mr);
}

}  // namespace cudf::groupby::detail
