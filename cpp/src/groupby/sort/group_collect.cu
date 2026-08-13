/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/iterator>
#include <thrust/copy.h>
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
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Pair of null-eliminated grouped values and corresponding offsets
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> purge_null_entries(
  column_view const& values,
  column_view const& offsets,
  size_type num_groups,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto values_device_view = column_device_view::create(values, stream);

  auto not_null_pred = [d_value = *values_device_view] __device__(auto i) -> bool {
    return d_value.is_valid_nocheck(i);
  };

  // Purge null entries in grouped values.
  auto null_purged_entries =
    cudf::detail::copy_if(table_view{{values}}, not_null_pred, stream, mr)->release();

  auto null_purged_values = std::move(null_purged_entries.front());
  null_purged_values->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);

  // Recalculate offsets after null entries are purged.
  rmm::device_uvector<size_type> null_purged_sizes(num_groups, stream);

  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{num_groups},
    null_purged_sizes.begin(),
    [d_offsets = offsets.template begin<size_type>(), not_null_pred] __device__(auto i) {
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
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto [child_column,
        offsets_column] = [null_handling, num_groups, &values, &group_offsets, stream, mr] {
    auto offsets_column = make_numeric_column(
      data_type(type_to_id<size_type>()), num_groups + 1, mask_state::UNALLOCATED, stream, mr);

    thrust::copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 group_offsets.begin(),
                 group_offsets.end(),
                 offsets_column->mutable_view().template begin<size_type>());

    // If column of grouped values contains null elements, and null_policy == EXCLUDE,
    // those elements must be filtered out, and offsets recomputed.
    if (null_handling == null_policy::EXCLUDE && values.has_nulls()) {
      return cudf::groupby::detail::purge_null_entries(
        values, offsets_column->view(), num_groups, stream, mr);
    } else {
      return std::pair(std::make_unique<cudf::column>(values, stream, mr),
                       std::move(offsets_column));
    }
  }();

  return make_lists_column(num_groups,
                           std::move(offsets_column),
                           std::move(child_column),
                           0,
                           rmm::device_buffer{0, stream, mr});
}
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
