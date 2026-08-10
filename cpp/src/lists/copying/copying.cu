/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_range.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <iostream>

namespace cudf {
namespace lists {
namespace detail {

// New lists column from a subset of a lists_column_view
std::unique_ptr<cudf::column> copy_slice(lists_column_view const& lists,
                                         size_type start,
                                         size_type end,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if (lists.is_empty() or start == end) { return cudf::empty_like(lists.parent()); }
  if (end < 0 || end > lists.size()) end = lists.size();
  CUDF_EXPECTS(((start >= 0) && (start < end)), "Invalid slice range.");
  auto lists_count   = end - start;
  auto offsets_count = lists_count + 1;  // num_offsets always 1 more than num_lists

  // Account for the offset of the view:
  start += lists.offset();
  end += lists.offset();

  // Offsets at the beginning and end of the slice:
  auto offsets_data = lists.offsets().data<int32_t>();
  auto start_offset = cudf::detail::get_value<int32_t>(lists.offsets(), start, stream);
  auto end_offset   = cudf::detail::get_value<int32_t>(lists.offsets(), end, stream);

  rmm::device_uvector<int32_t> out_offsets(offsets_count, stream);

  // Compute the offsets column of the result:
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    offsets_data + start,
    offsets_data + end + 1,  // size of offsets column is 1 greater than slice length
    out_offsets.data(),
    [start_offset] __device__(cudf::size_type i) { return i - start_offset; });
  auto offsets = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                offsets_count,
                                                out_offsets.release(),
                                                rmm::device_buffer{},
                                                0);

  // Compute the child column of the result.
  // If the child of this lists column is itself a lists column, we call copy_slice() on it.
  // Otherwise, it is a column of the leaf type, so we call slice() on it and copy the resulting
  // view into a cudf::column:
  auto child =
    (lists.child().type() == cudf::data_type{type_id::LIST})
      ? copy_slice(lists_column_view(lists.child()), start_offset, end_offset, stream, mr)
      : std::make_unique<cudf::column>(
          cudf::detail::slice(lists.child(), {start_offset, end_offset}, stream).front(),
          stream,
          mr);

  // Compute the null mask of the result:
  auto null_mask = cudf::detail::copy_bitmask(lists.null_mask(), start, end, stream, mr);

  auto null_count = cudf::detail::null_count(
    static_cast<bitmask_type const*>(null_mask.data()), 0, end - start, stream);

  return make_lists_column(
    lists_count, std::move(offsets), std::move(child), null_count, std::move(null_mask));
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
