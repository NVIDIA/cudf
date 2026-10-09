/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_reductions.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <memory>

namespace cudf {
namespace groupby {
namespace detail {
/**
 * @brief Purge null entries in grouped values, and adjust group offsets.
 *
 * @param values Grouped values to be purged
 * @param offsets Offsets of groups' starting points
 * @param num_groups Number of groups
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Memory resources used for temporary allocations and the returned columns
 * @return Pair of null-eliminated grouped values and corresponding offsets
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> purge_null_entries(
  column_view const& values,
  column_view const& offsets,
  size_type num_groups,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  auto values_device_view = column_device_view::create(values, stream, mr.get_temporary_mr());

  auto not_null_pred = [d_value = *values_device_view] __device__(auto i) -> bool {
    return d_value.is_valid_nocheck(i);
  };

  // Purge null entries in grouped values.
  auto null_purged_entries =
    cudf::detail::copy_if(table_view{{values}}, not_null_pred, stream, mr)->release();

  auto null_purged_values = std::move(null_purged_entries.front());
  null_purged_values->set_null_mask(
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr.get_output_mr()), 0);

  // Recalculate offsets after null entries are purged.
  rmm::device_uvector<size_type> null_purged_sizes(num_groups, stream, mr.get_temporary_mr());

  thrust::transform(
    rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{num_groups},
    null_purged_sizes.begin(),
    [d_offsets = offsets.template begin<int32_t>(), not_null_pred] __device__(auto i) {
      return thrust::count_if(thrust::seq,
                              cuda::counting_iterator<size_type>{d_offsets[i]},
                              cuda::counting_iterator<size_type>{d_offsets[i + 1]},
                              not_null_pred);
    });

  auto null_purged_offsets = std::get<0>(cudf::detail::make_offsets_child_column(
    null_purged_sizes.cbegin(), null_purged_sizes.cend(), stream, mr));

  return std::pair(std::move(null_purged_values), std::move(null_purged_offsets));
}

std::unique_ptr<column> group_collect(column_view const& values,
                                      cudf::device_span<size_type const> group_offsets,
                                      size_type num_groups,
                                      null_policy null_handling,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr)
{
  // If column of grouped values contains null elements, and null_policy == EXCLUDE,
  // those elements must be filtered out, and offsets recomputed.
  if (null_handling == null_policy::EXCLUDE && values.has_nulls()) {
    auto [child_column, purged_offsets] = cudf::groupby::detail::purge_null_entries(
      values, column_view{group_offsets}, num_groups, stream, mr);
    return make_lists_column(
      num_groups,
      std::move(purged_offsets),
      std::move(child_column),
      0,
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr.get_output_mr()));
  }
  return group_collect(std::make_unique<cudf::column>(values, stream, mr.get_output_mr()),
                       group_offsets,
                       num_groups,
                       null_handling,
                       stream,
                       mr);
}

std::unique_ptr<column> group_collect(std::unique_ptr<column> values,
                                      cudf::device_span<size_type const> group_offsets,
                                      size_type num_groups,
                                      null_policy null_handling,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr)
{
  if (null_handling == null_policy::EXCLUDE && values->has_nulls()) {
    return group_collect(values->view(), group_offsets, num_groups, null_handling, stream, mr);
  }
  // The caller is done with the grouped values, so they become the list child without a copy.
  auto offsets_column =
    std::make_unique<column>(column_view{group_offsets}, stream, mr.get_output_mr());
  return make_lists_column(
    num_groups,
    std::move(offsets_column),
    std::move(values),
    0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr.get_output_mr()));
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
