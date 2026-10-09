/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_scan_util.cuh"
#include "reductions/nested_types_extrema_utils.cuh"

#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/structs/utilities.hpp>

namespace cudf::groupby::detail {

std::unique_ptr<column> group_nested_minmax_scan(
  column_view const& values,
  cudf::device_span<cudf::size_type const> group_labels,
  bool is_min,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (values.is_empty()) { return cudf::empty_like(values); }

  // Create a gather map containing indices of the prefix min/max elements within each group.
  auto gather_map = rmm::device_uvector<size_type>(values.size(), stream);

  // The generated operation carries the MIN/MAX choice in its state, so one scan
  // instantiation serves both without changing the comparator specialization.
  auto const binop_generator =
    cudf::reduction::detail::arg_minmax_binop_generator::create(values, is_min, stream);
  thrust::inclusive_scan_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    group_labels.begin(),
    group_labels.end(),
    cuda::counting_iterator<size_type>{0},
    gather_map.begin(),
    cuda::std::equal_to{},
    binop_generator.binop());

  // Typically, gathering a sliced input requires get_sliced_child. Groupby internal APIs never
  // pass sliced views here, so child_begin and child_end are sufficient.
  auto scanned_children =
    cudf::detail::gather(
      table_view(std::vector<column_view>{values.child_begin(), values.child_end()}),
      gather_map,
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      mr)
      ->release();

  // Push root struct nulls down to the gathered children.
  if (values.has_nulls()) {
    for (std::unique_ptr<column>& child : scanned_children) {
      child = structs::detail::superimpose_and_sanitize_nulls(
        values.null_mask(), values.null_count(), std::move(child), stream, mr);
    }
  }

  return create_structs_hierarchy(values.size(),
                                  std::move(scanned_children),
                                  values.null_count(),
                                  cudf::detail::copy_bitmask(values, stream, mr),
                                  stream,
                                  mr);
}

}  // namespace cudf::groupby::detail
